# app/problems/handover.py
# =============================================================================
# RespiraCare-ICU — Shift Handover Information Degradation
#
# Models the most dangerous moment in any ICU — the shift transition.
# At hour 6, the incoming team has incomplete information about some patients.
# Some vitals are missing, some are approximated with noise.
#
# This is purely an observation-layer modification.
# The underlying patient state is always accurate — only what the agent
# SEES is degraded. This tests whether the agent handles uncertainty
# correctly rather than acting on stale or missing data overconfidently.
#
# ward.py calls this module in this order each step:
#   1. is_handover_active(hour)          → bool
#   2. apply_degradation(patients, hour) → List[patient_ids_degraded]
#   3. get_handover_stats()              → called by grader at episode end
#
# Design:
#   - Degradation applies to exactly HANDOVER_DEGRADATION_RATE (20%) of patients
#   - Which patients are degraded is seeded — same seed = same degraded set
#   - Degradation lasts exactly 1 hour (hour 6 only, clears at hour 7)
#   - The degradation itself is applied inside build_observation() on
#     PatientStateMachine using the degraded=True flag
#   - This module tracks WHICH patients are degraded and logs it for graders
# =============================================================================

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Set, Dict

from app import config


# =============================================================================
# HANDOVER RECORD
# =============================================================================

@dataclass
class HandoverEvent:
    """
    Record of a single shift handover event.
    Tracks which patients had degraded observations and for how long.
    """
    hour: int
    degraded_patient_ids: List[str] = field(default_factory=list)
    resolved: bool = False
    resolve_hour: int = -1


# =============================================================================
# HANDOVER COORDINATOR
# =============================================================================

class HandoverCoordinator:
    """
    Manages shift handover observation degradation across an episode.

    At hour HANDOVER_HOUR (6), selects 20% of patients randomly (seeded)
    and marks them for degraded observation. The PatientStateMachine's
    build_observation() method applies the actual noise — this coordinator
    tells ward.py which patients to degrade.

    Degradation clears automatically at hour 7.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self._rng = random.Random(seed + 4000)   # offset to avoid collision
        self.current_hour = 0

        # Patient IDs currently receiving degraded observations
        self._degraded_ids: Set[str] = set()

        # Full handover event log
        self._events: List[HandoverEvent] = []

        # Whether handover has been triggered this episode
        self._handover_triggered = False

    # =========================================================================
    # HANDOVER STATE
    # =========================================================================

    def is_handover_active(self, hour: int) -> bool:
        """
        Returns True if shift handover is currently in effect.
        Handover is active exactly at HANDOVER_HOUR (hour 6).
        """
        return hour == config.HANDOVER_HOUR

    def get_degraded_patient_ids(self) -> Set[str]:
        """
        Returns the set of patient IDs currently receiving degraded
        observations. Empty when handover is not active.
        """
        return self._degraded_ids.copy()

    def is_patient_degraded(self, patient_id: str) -> bool:
        """True if this patient's observation is currently degraded."""
        return patient_id in self._degraded_ids

    # =========================================================================
    # APPLY DEGRADATION
    # Called by ward.py when building the observation at HANDOVER_HOUR.
    # =========================================================================

    def apply_degradation(self, patients: list, hour: int) -> List[str]:
        """
        Select which patients receive degraded observations this hour.

        Selection is:
          - Seeded (reproducible with same seed)
          - Exactly floor(N * HANDOVER_DEGRADATION_RATE) patients
          - Only called once per episode (at HANDOVER_HOUR)

        Returns list of degraded patient IDs.
        """
        if not self.is_handover_active(hour):
            return []

        if self._handover_triggered:
            # Already triggered — return existing degraded set
            return list(self._degraded_ids)

        # Select patients to degrade
        active_patients = [p for p in patients if not p.is_discharged]
        n_to_degrade = max(
            1,
            int(len(active_patients) * config.HANDOVER_DEGRADATION_RATE)
        )

        # Seeded selection — reproducible
        selected = self._rng.sample(active_patients, k=n_to_degrade)
        degraded_ids = [p.patient_id for p in selected]

        self._degraded_ids = set(degraded_ids)
        self._handover_triggered = True

        # Log the event
        event = HandoverEvent(
            hour=hour,
            degraded_patient_ids=degraded_ids,
        )
        self._events.append(event)

        return degraded_ids

    def clear_degradation(self, hour: int) -> None:
        """
        Clear handover degradation. Called by ward.py at hour 7.
        Marks the handover event as resolved.
        """
        if self._degraded_ids:
            self._degraded_ids.clear()
            for event in self._events:
                if not event.resolved:
                    event.resolved = True
                    event.resolve_hour = hour

    # =========================================================================
    # ADVANCE HOUR
    # =========================================================================

    def advance_hour(self, patients: list) -> List[str]:
        """
        Tick the handover coordinator forward one hour.
        Automatically clears degradation at hour after HANDOVER_HOUR.

        Returns list of event strings for ward log.
        """
        self.current_hour += 1
        events_log = []

        # Trigger handover at the handover hour
        if self.current_hour == config.HANDOVER_HOUR:
            degraded = self.apply_degradation(patients, self.current_hour)
            if degraded:
                events_log.append(
                    f"handover:degradation_active:"
                    f"{','.join(degraded)}"
                )

        # Clear degradation one hour after handover
        elif self.current_hour == config.HANDOVER_HOUR + 1:
            if self._degraded_ids:
                self.clear_degradation(self.current_hour)
                events_log.append("handover:degradation_cleared")

        return events_log

    # =========================================================================
    # GRADER STATISTICS
    # =========================================================================

    def get_handover_stats(self) -> dict:
        """
        Returns handover event statistics for grader access.
        Graders use this to understand how many patients had degraded
        observations and whether the agent handled it appropriately.
        """
        total_degraded = sum(
            len(e.degraded_patient_ids) for e in self._events
        )

        events_detail = []
        for event in self._events:
            events_detail.append({
                "hour": event.hour,
                "degraded_patients": event.degraded_patient_ids,
                "n_degraded": len(event.degraded_patient_ids),
                "resolved": event.resolved,
                "resolve_hour": event.resolve_hour if event.resolved else None,
                "duration_hours": (
                    (event.resolve_hour - event.hour)
                    if event.resolved else None
                ),
            })

        return {
            "handover_triggered": self._handover_triggered,
            "handover_hour": config.HANDOVER_HOUR,
            "total_patients_degraded": total_degraded,
            "events": events_detail,
        }