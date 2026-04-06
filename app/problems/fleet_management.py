# app/problems/fleet_management.py
# =============================================================================
# RespiraCare-ICU — Respiratory Equipment Fleet Manager
#
# Tracks which equipment each patient is occupying, validates whether an
# action is possible given current availability, manages incoming patient
# alerts, and raises a penalty flag when a patient arrives with no equipment
# ready.
#
# ward.py calls this module's methods in this order each step:
#   1. validate_action()      — before applying any patient action
#   2. record_action()        — after a valid action is applied
#   3. check_incoming()       — at the start of each hour tick
#   4. advance_hour()         — at the end of each hour tick
# =============================================================================

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from app.models import (
    SupportLevel,
    ActionType,
    IncomingAlert,
    Severity,
)
from app import config


# =============================================================================
# INTERNAL DATA STRUCTURES
# =============================================================================

@dataclass
class IncomingPatientRecord:
    """
    Internal record of a patient being transferred into the ICU.
    Distinct from IncomingAlert (which is agent-visible) — this holds
    the ground truth arrival state used by the simulator.
    """
    alert_id: str
    eta_hours: int                          # Hours remaining until arrival
    support_needed: SupportLevel
    severity: Severity
    sofa_estimate: Optional[int]
    announced: bool = False                 # Whether alert has been sent to agent
    arrived: bool = False                   # Whether patient has landed in the ward
    hour_announced: int = 0
    hour_arrived: int = 0
    penalty_applied: bool = False           # Whether no-vent penalty was triggered


@dataclass
class EquipmentState:
    """
    Tracks the real-time equipment inventory and per-patient assignments.
    """
    total_ventilators: int = config.WARD_VENTILATORS
    total_bipap: int = config.WARD_BIPAP_UNITS
    total_hfnc: int = config.WARD_HFNC_UNITS

    # patient_id → SupportLevel currently assigned
    patient_assignments: Dict[str, SupportLevel] = field(default_factory=dict)

    # Running count of no-vent penalties this episode
    no_vent_penalty_count: int = 0

    @property
    def ventilators_in_use(self) -> int:
        return sum(
            1 for s in self.patient_assignments.values()
            if s == SupportLevel.FULL_VENTILATOR
        )

    @property
    def bipap_in_use(self) -> int:
        return sum(
            1 for s in self.patient_assignments.values()
            if s == SupportLevel.BIPAP
        )

    @property
    def hfnc_in_use(self) -> int:
        return sum(
            1 for s in self.patient_assignments.values()
            if s == SupportLevel.HFNC
        )

    @property
    def available_ventilators(self) -> int:
        return max(0, self.total_ventilators - self.ventilators_in_use)

    @property
    def available_bipap(self) -> int:
        return max(0, self.total_bipap - self.bipap_in_use)

    @property
    def available_hfnc(self) -> int:
        return max(0, self.total_hfnc - self.hfnc_in_use)


# =============================================================================
# FLEET MANAGER
# =============================================================================

class FleetManager:
    """
    Manages the ward's respiratory equipment inventory across an episode.

    Responsibilities:
      - Track which patient holds which equipment
      - Validate equipment availability before actions
      - Update assignments when patients change support levels
      - Generate IncomingAlert objects for the agent
      - Detect and penalise equipment shortfalls at patient arrival
    """

    def __init__(self, task_id: int, seed: int):
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed + 1000)   # offset to avoid collision with patient seeds

        self.equipment = EquipmentState()
        self.incoming_records: List[IncomingPatientRecord] = []
        self.current_hour = 0

        # Penalty events logged this episode — ward.py reads these
        self.pending_penalties: List[str] = []

        # Build the incoming patient schedule for this task
        self._schedule = self._build_incoming_schedule()

    # =========================================================================
    # INITIALISATION
    # =========================================================================

    def initialise_from_patients(self, patients) -> None:
        """
        Register each patient's starting support level so the equipment
        inventory starts in the correct state.
        Called once by ward.py after patients are generated.
        """
        self.equipment.patient_assignments.clear()
        for patient in patients:
            if patient.support_level != SupportLevel.ROOM_AIR:
                self.equipment.patient_assignments[patient.patient_id] = (
                    patient.support_level
                )

    def _build_incoming_schedule(self) -> List[IncomingPatientRecord]:
        """
        Define when incoming patients arrive for each task.
        Returns a list of IncomingPatientRecord objects.
        Alerts are announced 2 hours before arrival.
        """
        records = []

        if self.task_id == 1:
            # No incoming patients in Task 1
            pass

        elif self.task_id == 2:
            # 3 trauma patients arriving at hour 3 — announced at hour 1
            for i in range(3):
                records.append(IncomingPatientRecord(
                    alert_id=f"INC-T2-{i+1:02d}",
                    eta_hours=3,        # Will arrive at hour 3
                    support_needed=SupportLevel.FULL_VENTILATOR,
                    severity=Severity.HIGH,
                    sofa_estimate=self._rng.randint(10, 16),
                ))

        elif self.task_id == 3:
            # 4 patients from mass casualty event — announced at hour 3, arrive hour 5
            for i in range(4):
                records.append(IncomingPatientRecord(
                    alert_id=f"INC-T3-MCI-{i+1:02d}",
                    eta_hours=5,        # Will arrive at hour 5
                    support_needed=SupportLevel.FULL_VENTILATOR,
                    severity=Severity.HIGH,
                    sofa_estimate=self._rng.randint(11, 18),
                ))

        return records

    # =========================================================================
    # ACTION VALIDATION
    # =========================================================================

    def validate_action(
        self,
        patient_id: str,
        action_type: ActionType,
        current_support: SupportLevel,
    ) -> tuple[bool, str]:
        """
        Check whether an action is possible given current equipment inventory.

        Returns:
            (is_valid: bool, reason: str)
        """
        # Actions that require a FREE ventilator
        if action_type == ActionType.ESCALATE_TO_FULL_VENT:
            if current_support == SupportLevel.FULL_VENTILATOR:
                return False, "already_on_full_vent"
            if self.equipment.available_ventilators < 1:
                return False, "no_ventilator_available"

        # Actions that require a FREE BiPAP unit
        elif action_type == ActionType.STEP_DOWN_TO_BIPAP:
            if current_support != SupportLevel.FULL_VENTILATOR:
                return False, "not_on_full_vent"
            if self.equipment.available_bipap < 1:
                return False, "no_bipap_available"

        # Actions that require a FREE HFNC unit
        elif action_type == ActionType.STEP_DOWN_TO_HFNC:
            if current_support not in (SupportLevel.FULL_VENTILATOR, SupportLevel.BIPAP):
                return False, "not_on_vent_or_bipap"
            if self.equipment.available_hfnc < 1:
                return False, "no_hfnc_available"

        # Extubation frees equipment — always valid if patient state allows it
        # (patient.py checks state — fleet manager only checks equipment)
        elif action_type == ActionType.EXTUBATE:
            pass

        # SBT — no equipment change, always valid from fleet perspective
        elif action_type == ActionType.ATTEMPT_SBT:
            pass

        return True, "ok"

    # =========================================================================
    # RECORD ACTION OUTCOME
    # =========================================================================

    def record_action(
        self,
        patient_id: str,
        action_type: ActionType,
        new_support: SupportLevel,
    ) -> None:
        """
        Update equipment assignments after an action is successfully applied.
        Called by ward.py after patient.apply_action() succeeds.
        """
        old_support = self.equipment.patient_assignments.get(patient_id)

        if new_support == SupportLevel.ROOM_AIR:
            # Patient freed all equipment
            self.equipment.patient_assignments.pop(patient_id, None)
        elif new_support != old_support:
            # Support level changed — update assignment
            self.equipment.patient_assignments[patient_id] = new_support

    # =========================================================================
    # INCOMING PATIENT MANAGEMENT
    # =========================================================================

    def get_active_alerts(self) -> List[IncomingAlert]:
        """
        Returns agent-visible IncomingAlert objects for patients announced
        but not yet arrived. Called by ward.py when building Observation.
        """
        alerts = []
        for record in self.incoming_records:
            if record.announced and not record.arrived:
                hours_remaining = record.eta_hours - self.current_hour
                if hours_remaining > 0:
                    alerts.append(IncomingAlert(
                        alert_id=record.alert_id,
                        eta_hours=hours_remaining,
                        support_needed=record.support_needed,
                        sofa_estimate=record.sofa_estimate,
                        severity_estimate=record.severity,
                    ))
        return alerts

    def advance_hour(self) -> List[str]:
        """
        Called at the end of each step by ward.py.
        Announces incoming patients 2 hours before arrival.
        Checks arrivals and raises penalty if no equipment is ready.

        Returns a list of event strings for the ward log.
        """
        events = []
        self.current_hour += 1

        for record in self.incoming_records:
            if record.arrived:
                continue

            hours_until_arrival = record.eta_hours - self.current_hour

            # Announce 2 hours before arrival
            if hours_until_arrival == 2 and not record.announced:
                record.announced = True
                record.hour_announced = self.current_hour
                events.append(f"fleet:incoming_announced:{record.alert_id}")

            # Patient arrives
            if hours_until_arrival <= 0 and not record.arrived:
                record.arrived = True
                record.hour_arrived = self.current_hour

                # Check equipment availability
                if record.support_needed == SupportLevel.FULL_VENTILATOR:
                    if self.equipment.available_ventilators < 1:
                        # No ventilator ready — penalty
                        self.equipment.no_vent_penalty_count += 1
                        record.penalty_applied = True
                        self.pending_penalties.append(
                            f"no_vent_for_incoming:{record.alert_id}"
                        )
                        events.append(
                            f"fleet:no_vent_penalty:{record.alert_id}"
                        )
                    else:
                        # Assign ventilator to incoming patient
                        new_id = f"INC-{record.alert_id}"
                        self.equipment.patient_assignments[new_id] = (
                            SupportLevel.FULL_VENTILATOR
                        )
                        events.append(
                            f"fleet:incoming_arrived_ventilated:{record.alert_id}"
                        )

        return events

    # =========================================================================
    # ACCESSORS — used by ward.py when building Observation
    # =========================================================================

    def pop_pending_penalties(self) -> List[str]:
        """
        Returns and clears the pending penalty list.
        ward.py calls this each step to apply penalties to the reward.
        """
        penalties = self.pending_penalties.copy()
        self.pending_penalties.clear()
        return penalties

    @property
    def available_ventilators(self) -> int:
        return self.equipment.available_ventilators

    @property
    def available_bipap(self) -> int:
        return self.equipment.available_bipap

    @property
    def available_hfnc(self) -> int:
        return self.equipment.available_hfnc

    def get_summary(self) -> dict:
        """
        Returns a summary dict for the /state endpoint and grader access.
        """
        return {
            "available_ventilators": self.available_ventilators,
            "available_bipap": self.available_bipap,
            "available_hfnc": self.available_hfnc,
            "ventilators_in_use": self.equipment.ventilators_in_use,
            "bipap_in_use": self.equipment.bipap_in_use,
            "hfnc_in_use": self.equipment.hfnc_in_use,
            "no_vent_penalty_count": self.equipment.no_vent_penalty_count,
            "incoming_announced": sum(
                1 for r in self.incoming_records if r.announced and not r.arrived
            ),
            "incoming_total": len(self.incoming_records),
        }