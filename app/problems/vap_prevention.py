# app/problems/vap_prevention.py
# =============================================================================
# RespiraCare-ICU — VAP Prevention Tracker
#
# Tracks VAP bundle compliance per patient across the episode and manages
# the delayed infection consequence model.
#
# The core mechanic that makes this clinically realistic:
#   - Missing the bundle adds risk NOW (vap_risk += 0.03 per hour)
#   - The infection itself arrives LATER (VAP_ONSET_DELAY_HOURS = 2)
#   - The penalty fires when VAP develops — not when the bundle was missed
#   - This teaches the agent that present shortcuts have future costs
#
# ward.py calls this module in this order each step:
#   1. record_bundle_action(patient_id)  — when enforce_vap_bundle is applied
#   2. advance_hour(patients)            — ticks risk and fires pending events
#   3. get_compliance_stats()            — called by grader at episode end
#
# VAP risk lives on the PatientStateMachine (patient.py manages _accrue_vap_risk
# and _check_vap_onset). This module provides the ward-level coordination layer:
# it aggregates compliance data, fires ward-level penalties, and gives the
# grader a clean interface to compliance history.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from app.models import SupportLevel, PatientState
from app import config


# =============================================================================
# PER-PATIENT COMPLIANCE RECORD
# =============================================================================

@dataclass
class PatientComplianceRecord:
    """
    Tracks the full VAP bundle compliance history for one patient
    across the entire episode.
    """
    patient_id: str

    # Hours bundle was enforced
    compliant_hours: List[int] = field(default_factory=list)

    # Hours bundle was missed while patient was at risk
    missed_hours: List[int] = field(default_factory=list)

    # Hours patient was eligible for VAP risk (on vent >= 48h)
    eligible_hours: List[int] = field(default_factory=list)

    # Whether VAP has developed for this patient
    vap_developed: bool = False

    # Hour VAP developed (for grader timeline)
    vap_onset_hour: int = -1

    # Risk value at the moment VAP triggered
    vap_risk_at_trigger: float = 0.0


# =============================================================================
# VAP PREVENTION COORDINATOR
# =============================================================================

class VAPPreventionCoordinator:
    """
    Ward-level coordinator for VAP bundle compliance and infection tracking.

    Sits above patient.py's individual VAP mechanics — this module handles:
      - Aggregating compliance records across all patients
      - Logging penalty events when VAP develops
      - Giving graders a clean compliance summary
      - Providing per-patient overdue warnings for the agent

    The actual vap_risk accumulation and PendingVAPEvent firing lives in
    PatientStateMachine.advance_hour() — this coordinator observes those
    outcomes and records them for grading.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.current_hour = 0

        # Per-patient compliance records
        self._records: Dict[str, PatientComplianceRecord] = {}

        # Penalty events pending collection by ward.py
        self._pending_penalties: List[str] = []

        # Episode-level VAP incident count
        self.total_vap_incidents = 0

    # =========================================================================
    # INITIALISATION
    # =========================================================================

    def initialise_from_patients(self, patients: list) -> None:
        """
        Create a compliance record for each patient.
        Called once by ward.py after ward generation.
        """
        self._records.clear()
        for patient in patients:
            self._records[patient.patient_id] = PatientComplianceRecord(
                patient_id=patient.patient_id
            )

    # =========================================================================
    # BUNDLE ACTION RECORDING
    # Called by ward.py when agent applies enforce_vap_bundle to a patient.
    # =========================================================================

    def record_bundle_action(self, patient_id: str) -> None:
        """
        Mark this hour as compliant for the given patient.
        Resets the hours_since_last_bundle counter on the patient object
        is handled by patient.py — this just updates the compliance log.
        """
        record = self._records.get(patient_id)
        if record is None:
            return
        if self.current_hour not in record.compliant_hours:
            record.compliant_hours.append(self.current_hour)

    # =========================================================================
    # ADVANCE HOUR
    # Called by ward.py at the end of each step after all actions are applied.
    # =========================================================================

    def advance_hour(self, patients: list) -> List[str]:
        events = []

        for patient in patients:
            if patient.is_discharged:
                continue

            if patient.support_level != SupportLevel.FULL_VENTILATOR:
                continue

            record = self._records.get(patient.patient_id)
            if record is None:
                continue

            is_eligible = patient.hours_on_vent >= config.VAP_ELIGIBLE_AFTER_HOURS

            if is_eligible:
                record.eligible_hours.append(self.current_hour)

                # Use the coordinator's own record to determine compliance.
                # record_bundle_action() appends self.current_hour when the
                # agent applies enforce_vap_bundle — check that instead of
                # relying on patient.hours_since_last_bundle.
                bundle_done_this_hour = self.current_hour in record.compliant_hours

                if not bundle_done_this_hour:
                    record.missed_hours.append(self.current_hour)

            # Detect new VAP development
            if (
                patient.has_vap
                and not record.vap_developed
                and patient.state == PatientState.INTUBATED_WITH_VAP
            ):
                record.vap_developed = True
                record.vap_onset_hour = self.current_hour
                record.vap_risk_at_trigger = patient.vap_risk
                self.total_vap_incidents += 1

                self._pending_penalties.append(
                    f"vap_developed:{patient.patient_id}:hour_{self.current_hour}"
                )
                events.append(
                    f"vap:infection_developed:{patient.patient_id}"
                )

        # Increment AFTER processing so record_bundle_action hour matches
        self.current_hour += 1
        return events

    # =========================================================================
    # OVERDUE WARNINGS
    # Used by ward.py to annotate observations with at-risk patients.
    # =========================================================================

    def get_overdue_patients(self, patients: list) -> List[str]:
        """
        Returns a list of patient IDs whose VAP bundle is overdue (>= 4 hours
        since last enforcement). Used to help highlight at-risk patients.
        """
        overdue = []
        for patient in patients:
            if patient.support_level != SupportLevel.FULL_VENTILATOR:
                continue
            if patient.is_discharged:
                continue
            if patient.hours_on_vent >= config.VAP_ELIGIBLE_AFTER_HOURS:
                if patient.vap_bundle_overdue:
                    overdue.append(patient.patient_id)
        return overdue

    # =========================================================================
    # PENALTY COLLECTION
    # =========================================================================

    def pop_pending_penalties(self) -> List[str]:
        """
        Returns and clears the pending penalty list.
        ward.py calls this each step to apply VAP penalties to the reward.
        """
        penalties = self._pending_penalties.copy()
        self._pending_penalties.clear()
        return penalties

    # =========================================================================
    # GRADER STATISTICS
    # Called by graders at episode end.
    # =========================================================================

    def get_compliance_stats(self) -> dict:
        """
        Compute VAP bundle compliance statistics across the entire episode.
        Returns a grader-ready dict with per-patient and aggregate metrics.
        """
        total_eligible = 0
        total_compliant = 0
        total_missed = 0
        vap_incidents = []
        per_patient = {}

        for pid, record in self._records.items():
            eligible = len(record.eligible_hours)
            compliant = len(record.compliant_hours)
            missed = len(record.missed_hours)

            total_eligible += eligible
            total_compliant += compliant
            total_missed += missed

            compliance_rate = (
                round(compliant / eligible, 4) if eligible > 0 else 1.0
            )

            per_patient[pid] = {
                "eligible_hours": eligible,
                "compliant_hours": compliant,
                "missed_hours": missed,
                "compliance_rate": compliance_rate,
                "vap_developed": record.vap_developed,
                "vap_onset_hour": record.vap_onset_hour if record.vap_developed else None,
            }

            if record.vap_developed:
                vap_incidents.append({
                    "patient_id": pid,
                    "onset_hour": record.vap_onset_hour,
                    "risk_at_trigger": record.vap_risk_at_trigger,
                })

        # Overall compliance rate across all patients
        overall_rate = (
            round(total_compliant / total_eligible, 4)
            if total_eligible > 0 else 1.0
        )

        # Grader score: compliance rate penalised by VAP incidence
        # Each VAP incident reduces the score by 0.15 (capped at 0)
        vap_penalty = len(vap_incidents) * 0.15
        compliance_score = round(
            max(0.0, overall_rate - vap_penalty), 4
        )

        return {
            "overall_compliance_rate": overall_rate,
            "total_eligible_hours": total_eligible,
            "total_compliant_hours": total_compliant,
            "total_missed_hours": total_missed,
            "total_vap_incidents": len(vap_incidents),
            "vap_incidents": vap_incidents,
            "compliance_score": compliance_score,
            "per_patient": per_patient,
        }

    def get_patient_record(self, patient_id: str) -> PatientComplianceRecord:
        """Direct access to a patient's compliance record — for grader use."""
        return self._records.get(patient_id)