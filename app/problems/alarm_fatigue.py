# app/problems/alarm_fatigue.py
# =============================================================================
# RespiraCare-ICU — Alarm Fatigue Simulator
#
# Generates a realistic ICU alarm feed each hour for every ventilated patient.
# 87% of alarms are false positives — staff become desensitised and miss real
# deterioration signals buried in noise. The agent must learn to distinguish
# real alarms from false positives without seeing the is_real flag directly.
#
# ward.py calls this module in this order each step:
#   1. generate_alarms(patients, hour)  — produces this hour's alarm feed
#   2. record_agent_response()          — logs what the agent did per alarm
#   3. get_active_alarms()              — returns agent-visible AlarmEvent list
#   4. get_alarm_accuracy_stats()       — called by grader at episode end
#
# The is_real field is stored in InternalAlarmRecord (hidden from agent).
# The agent only sees AlarmEvent which does not contain is_real.
# =============================================================================

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from app.models import (
    AlarmEvent,
    AlarmType,
    InternalAlarmRecord,
    PatientState,
    SupportLevel,
)
from app import config


# =============================================================================
# ALARM FATIGUE SIMULATOR
# =============================================================================

class AlarmFatigueSimulator:
    """
    Generates and tracks the ICU alarm feed across an entire episode.

    Key design decisions:
      - Alarm generation is seeded for reproducibility
      - Real alarms are tied to actual patient deterioration flags
      - Cry-wolf pattern: repeated false alarms raise the FP rate further
      - The grader uses internal records to score agent alarm responses
    """

    def __init__(self, seed: int):
        self.seed = seed
        self._rng = random.Random(seed + 2000)   # offset to avoid collision

        # Internal full records (includes is_real) — never sent to agent
        self._internal_records: Dict[str, InternalAlarmRecord] = {}

        # Active alarm IDs visible to the agent this hour
        self._active_alarm_ids: List[str] = []

        # Per-patient consecutive false alarm counter
        # alarm_type string → count since last real alarm
        self._consecutive_false: Dict[str, Dict[str, int]] = {}

        # Alarm ID counter for unique IDs
        self._alarm_counter = 0

        # Load alarm pattern definitions
        self._patterns = self._load_patterns()

    # =========================================================================
    # INITIALISATION
    # =========================================================================

    def _load_patterns(self) -> dict:
        """
        Load alarm type definitions from clinical_data/alarm_patterns.json.
        Falls back to hardcoded defaults if file is missing.
        """
        patterns_path = Path("clinical_data/alarm_patterns.json")
        if patterns_path.exists():
            with open(patterns_path, "r") as f:
                return json.load(f)

        # Fallback — mirrors the JSON file content
        return {
            "false_positive_rate": config.ALARM_FALSE_POSITIVE_RATE,
            "alarm_types": [
                {"id": "spo2_low",      "name": "SpO2 below threshold"},
                {"id": "hr_high",       "name": "Heart rate high"},
                {"id": "rr_high",       "name": "Respiratory rate high"},
                {"id": "peak_pressure", "name": "Peak inspiratory pressure"},
                {"id": "low_tidal_vol", "name": "Low tidal volume"},
                {"id": "apnea",         "name": "Apnea alarm"},
                {"id": "battery",       "name": "Battery/power alarm"},
            ],
            "cry_wolf_pattern": {
                "consecutive_false_threshold": config.CRY_WOLF_CONSECUTIVE_THRESHOLD,
                "subsequent_false_positive_rate": config.CRY_WOLF_SUBSEQUENT_FP_RATE,
            },
            "alarms_per_patient_per_hour_range": [
                config.ALARMS_PER_PATIENT_MIN,
                config.ALARMS_PER_PATIENT_MAX,
            ],
        }

    def initialise_patient(self, patient_id: str) -> None:
        """
        Set up per-patient alarm tracking. Called by ward.py for each patient.
        """
        self._consecutive_false[patient_id] = {t.value: 0 for t in AlarmType}

    # =========================================================================
    # REAL ALARM DETECTION
    # Checks actual patient vitals to determine if an alarm is genuinely real.
    # =========================================================================

    def _is_alarm_real(self, patient, alarm_type: AlarmType) -> bool:
        """
        Determine whether an alarm is a genuine clinical event based on
        the patient's actual (internal) vital signs.

        This is the ground truth — never exposed to the agent.
        """
        if alarm_type == AlarmType.SPO2_LOW:
            return patient.spo2 < 90.0

        elif alarm_type == AlarmType.HR_HIGH:
            return patient.heart_rate > 120.0

        elif alarm_type == AlarmType.RR_HIGH:
            return patient.resp_rate > 28.0

        elif alarm_type == AlarmType.PEAK_PRESSURE:
            return patient.peep > 12.0

        elif alarm_type == AlarmType.LOW_TIDAL_VOL:
            # Low tidal volume is real when patient is unstable
            return patient.state == PatientState.INTUBATED_UNSTABLE

        elif alarm_type == AlarmType.APNEA:
            # Apnea is real only in severe instability
            return (
                patient.state == PatientState.INTUBATED_UNSTABLE
                and patient.spo2 < 88.0
            )

        elif alarm_type == AlarmType.BATTERY:
            # Battery alarms are almost always false positives
            return False

        return False

    # =========================================================================
    # ALARM GENERATION — called each hour
    # =========================================================================

    def _next_alarm_id(self) -> str:
        self._alarm_counter += 1
        return f"ALM-{self._alarm_counter:04d}"

    def _pick_alarm_type(self) -> AlarmType:
        """Sample a random alarm type weighted by clinical frequency."""
        # SpO2 and HR alarms are most common in real ICUs
        weights = [0.30, 0.22, 0.18, 0.12, 0.10, 0.05, 0.03]
        alarm_types = list(AlarmType)
        return self._rng.choices(alarm_types, weights=weights, k=1)[0]

    def _compute_false_positive_rate(
        self, patient_id: str, alarm_type: AlarmType
    ) -> float:
        """
        Compute the effective false positive rate for this patient and alarm type.
        Applies the cry-wolf pattern: if consecutive false alarms exceed the
        threshold, the FP rate rises to the cry-wolf rate.
        """
        consecutive = self._consecutive_false.get(patient_id, {}).get(
            alarm_type.value, 0
        )
        cry_wolf = self._patterns["cry_wolf_pattern"]

        if consecutive >= cry_wolf["consecutive_false_threshold"]:
            return cry_wolf["subsequent_false_positive_rate"]

        return self._patterns["false_positive_rate"]

    def generate_alarms(self, patients: list, hour: int) -> None:
        """
        Generate this hour's alarm feed for all ventilated patients.
        Stores full InternalAlarmRecord for each alarm (with is_real).
        Updates _active_alarm_ids for the agent-visible feed.

        Called by ward.py at the start of each step.
        """
        self._active_alarm_ids.clear()
        alarm_range = self._patterns["alarms_per_patient_per_hour_range"]

        for patient in patients:
            # Only ventilated patients generate monitor alarms
            if patient.support_level not in (
                SupportLevel.FULL_VENTILATOR, SupportLevel.BIPAP
            ):
                continue

            # Discharged patients generate no alarms
            if patient.is_discharged:
                continue

            # Ensure patient is tracked
            if patient.patient_id not in self._consecutive_false:
                self.initialise_patient(patient.patient_id)

            # How many alarms this patient generates this hour
            num_alarms = self._rng.randint(alarm_range[0], alarm_range[1])

            for _ in range(num_alarms):
                alarm_type = self._pick_alarm_type()
                alarm_id = self._next_alarm_id()

                # Is this alarm real? Check actual vitals first.
                vitals_indicate_real = self._is_alarm_real(patient, alarm_type)

                # Apply false positive rate
                fp_rate = self._compute_false_positive_rate(
                    patient.patient_id, alarm_type
                )

                if vitals_indicate_real:
                    # Vitals confirm a real problem — override FP rate
                    is_real = True
                else:
                    # Vitals are fine — this is a false positive
                    is_real = False

                # Even when vitals look fine, occasionally a real event slips through
                # (equipment sensitivity, transient desaturation not in snapshot)
                if not is_real and self._rng.random() > fp_rate:
                    is_real = True

                # Update consecutive false alarm counter
                consec = self._consecutive_false[patient.patient_id]
                if is_real:
                    consec[alarm_type.value] = 0   # Reset on real alarm
                else:
                    consec[alarm_type.value] = consec.get(alarm_type.value, 0) + 1

                # Get consecutive count for agent observation
                consecutive_count = consec.get(alarm_type.value, 0)

                # Store internal record (with is_real — hidden from agent)
                record = InternalAlarmRecord(
                    alarm_id=alarm_id,
                    patient_id=patient.patient_id,
                    alarm_type=alarm_type,
                    hour_generated=hour,
                    is_real=is_real,
                    consecutive_same_type=consecutive_count,
                )
                self._internal_records[alarm_id] = record
                self._active_alarm_ids.append(alarm_id)

    # =========================================================================
    # AGENT-VISIBLE ALARM FEED
    # =========================================================================

    def get_active_alarms(self) -> List[AlarmEvent]:
        """
        Returns the agent-visible alarm feed — AlarmEvent objects without
        the is_real field. This is what goes into Observation.active_alarms.
        """
        events = []
        for alarm_id in self._active_alarm_ids:
            record = self._internal_records.get(alarm_id)
            if record is None:
                continue
            events.append(AlarmEvent(
                alarm_id=record.alarm_id,
                patient_id=record.patient_id,
                alarm_type=record.alarm_type,
                hour_generated=record.hour_generated,
                consecutive_same_type=record.consecutive_same_type,
            ))
        return events

    # =========================================================================
    # RECORD AGENT RESPONSE
    # Called by ward.py when agent uses respond_to_alarm or suppress_alarm.
    # =========================================================================

    def record_agent_response(
        self,
        patient_id: str,
        action_type,          # ActionType — avoid circular import
        hour: int,
    ) -> Tuple[float, str]:
        """
        Record the agent's response to alarms for a given patient this hour.
        Returns (reward_delta, event_string).

        Logic:
          respond_to_alarm + is_real=True   → correct response (TP) → reward
          suppress_alarm   + is_real=False  → correct suppression (TN) → small reward
          respond_to_alarm + is_real=False  → false escalation (FP) → small penalty
          suppress_alarm   + is_real=True   → missed real alarm (FN) → large penalty
        """
        from app.models import ActionType as AT

        # Find alarms for this patient that haven't been responded to yet
        patient_alarms = [
            r for r in self._internal_records.values()
            if r.patient_id == patient_id
            and r.alarm_id in self._active_alarm_ids
            and r.agent_responded is None
        ]

        if not patient_alarms:
            return 0.0, "no_active_alarm_for_patient"

        total_reward = 0.0
        events = []

        for record in patient_alarms:
            record.hour_responded = hour

            if action_type == AT.RESPOND_TO_ALARM:
                record.agent_responded = True
                if record.is_real:
                    # True positive — correctly responded to real alarm
                    total_reward += config.REWARD_REAL_ALARM_CORRECTLY_ACTIONED
                    events.append(f"alarm_tp:{record.alarm_id}")
                else:
                    # False positive — wasted RT time on non-event
                    total_reward += config.PENALTY_FALSE_ALARM_ESCALATED
                    events.append(f"alarm_fp:{record.alarm_id}")

            elif action_type == AT.SUPPRESS_ALARM:
                record.agent_responded = False
                if not record.is_real:
                    # True negative — correctly suppressed a false alarm
                    total_reward += config.REWARD_FALSE_ALARM_CORRECTLY_SUPPRESSED
                    events.append(f"alarm_tn:{record.alarm_id}")
                else:
                    # False negative — missed a real alarm — dangerous
                    total_reward += config.PENALTY_REAL_ALARM_IGNORED
                    events.append(f"alarm_fn:{record.alarm_id}")

        return total_reward, "|".join(events)

    # =========================================================================
    # GRADER STATISTICS
    # Called by graders at end of episode.
    # =========================================================================

    def get_alarm_accuracy_stats(self) -> dict:
        """
        Compute alarm response statistics across the entire episode.
        Used by shift_grader and crisis_grader to score alarm accuracy.

        Returns a dict with TP, FP, TN, FN counts and derived metrics.
        """
        tp = fp = tn = fn = unanswered_real = unanswered_false = 0

        for record in self._internal_records.values():
            if record.agent_responded is None:
                # Agent did not respond to this alarm at all
                if record.is_real:
                    unanswered_real += 1    # Treated as missed (FN)
                    fn += 1
                else:
                    unanswered_false += 1   # Treated as passively suppressed (TN)
                    tn += 1
            elif record.agent_responded and record.is_real:
                tp += 1
            elif record.agent_responded and not record.is_real:
                fp += 1
            elif not record.agent_responded and not record.is_real:
                tn += 1
            elif not record.agent_responded and record.is_real:
                fn += 1

        total_real = tp + fn
        total_false = fp + tn

        sensitivity = tp / total_real if total_real > 0 else 1.0
        specificity = tn / total_false if total_false > 0 else 1.0

        # Grader metric: sensitivity weighted heavier than specificity
        # Missing a real alarm is much worse than acting on a false one
        accuracy_score = (sensitivity * 0.70) + (specificity * 0.30)
        accuracy_score = round(min(1.0, max(0.0, accuracy_score)), 4)

        return {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "unanswered_real": unanswered_real,
            "unanswered_false": unanswered_false,
            "total_alarms": len(self._internal_records),
            "total_real": total_real,
            "total_false": total_false,
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "accuracy_score": accuracy_score,
        }

    def get_all_internal_records(self) -> Dict[str, InternalAlarmRecord]:
        """Returns full internal alarm records — for grader access only."""
        return self._internal_records