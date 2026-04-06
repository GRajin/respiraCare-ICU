# app/problems/ethical_triage.py
# =============================================================================
# RespiraCare-ICU — Ethical Triage Engine
#
# Models the crisis standard of care decision that occurs when two patients
# simultaneously need a ventilator and only one can be allocated.
#
# The framework is deterministic and based on published clinical guidelines:
#   - Lower SOFA score = better prognosis = gets priority
#   - Tie-breaker: patient who has been waiting longer wins
#   - Secondary tie-breaker: lower patient ID (deterministic, no arbitrariness)
#
# ward.py calls this module in this order each step:
#   1. check_triage_needed(patients, available_vents) → EthicalTriageCase | None
#   2. record_agent_decision(case_id, chosen_patient_id)
#   3. score_decision(case_id) → (reward_delta, is_correct)
#   4. get_triage_stats() → called by grader at episode end
#
# The correct answer is always deterministic — no subjective judgment.
# The grader can reproduce the correct answer from SOFA scores alone.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.models import (
    EthicalTriageCase,
    SupportLevel,
    PatientState,
)
from app import config


# =============================================================================
# INTERNAL TRIAGE RECORD
# =============================================================================

@dataclass
class TriageRecord:
    """
    Full internal record of one ethical triage event.
    Stores both the correct answer and the agent's decision.
    """
    case_id: str
    hour: int

    # The two competing patients
    patient_a_id: str
    patient_b_id: str
    patient_a_sofa: int
    patient_b_sofa: int
    patient_a_wait_hours: int
    patient_b_wait_hours: int

    # Ground truth — computed deterministically from SOFA + wait time
    correct_patient_id: str
    correct_reason: str   # e.g. "lower_sofa" or "tie_broken_by_wait_time"

    # Agent's response
    agent_chosen_id: Optional[str] = None
    agent_responded: bool = False
    is_correct: Optional[bool] = None


# =============================================================================
# ETHICAL TRIAGE ENGINE
# =============================================================================

class EthicalTriageEngine:
    """
    Manages ethical triage events across an episode.

    A triage event is triggered when:
      - Two or more patients are in a state requiring a ventilator
        (INTUBATED_UNSTABLE deteriorating to needing escalation, or
         an incoming patient) AND
      - Only one ventilator is available

    In Task 3 this is injected at hour 10 by the crisis schedule.
    It can also arise organically in Tasks 2 and 3 from fleet pressure.

    The engine generates the EthicalTriageCase for the agent's Observation,
    records the agent's decision, and scores it deterministically.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.current_hour = 0

        # All triage records this episode
        self._records: Dict[str, TriageRecord] = {}

        # Case ID counter
        self._case_counter = 0

        # Per-patient wait time tracking (hours spent needing a vent without one)
        self._waiting_since: Dict[str, int] = {}

        # Currently active triage case (only one can be active at a time)
        self._active_case_id: Optional[str] = None

        # Pending reward events for ward.py
        self._pending_rewards: List[Tuple[float, str]] = []

    # =========================================================================
    # WAIT TIME TRACKING
    # =========================================================================

    def record_patient_waiting(self, patient_id: str) -> None:
        """
        Mark a patient as waiting for a ventilator this hour.
        Called by ward.py when a patient needs escalation but no vent is free.
        """
        if patient_id not in self._waiting_since:
            self._waiting_since[patient_id] = self.current_hour

    def get_wait_hours(self, patient_id: str) -> int:
        """How many hours has this patient been waiting for a vent."""
        if patient_id not in self._waiting_since:
            return 0
        return self.current_hour - self._waiting_since[patient_id]

    # =========================================================================
    # TRIAGE CASE GENERATION
    # =========================================================================

    def _next_case_id(self) -> str:
        self._case_counter += 1
        return f"TRIAGE-{self._case_counter:03d}"

    def _determine_correct_patient(
        self,
        patient_a_id: str,
        patient_b_id: str,
        sofa_a: int,
        sofa_b: int,
        wait_a: int,
        wait_b: int,
    ) -> Tuple[str, str]:
        """
        Apply the SOFA-based triage framework deterministically.

        Rules (in order):
          1. Lower SOFA score wins (better prognosis)
          2. Tie on SOFA: longer wait time wins (first-come priority)
          3. Tie on both: lower patient ID wins (stable, arbitrary tie-break)

        Returns (correct_patient_id, reason_string).
        """
        if sofa_a < sofa_b:
            return patient_a_id, "lower_sofa"

        elif sofa_b < sofa_a:
            return patient_b_id, "lower_sofa"

        else:
            # SOFA tied — use wait time
            if wait_a > wait_b:
                return patient_a_id, "tie_broken_by_wait_time"
            elif wait_b > wait_a:
                return patient_b_id, "tie_broken_by_wait_time"
            else:
                # Both tied — deterministic tie-break by patient ID
                winner = min(patient_a_id, patient_b_id)
                return winner, "tie_broken_by_patient_id"

    def check_triage_needed(
        self,
        patients: list,
        available_ventilators: int,
    ) -> Optional[EthicalTriageCase]:
        """
        Scan the ward for a triage scenario.

        A triage case exists when:
          - Exactly 1 ventilator is available
          - 2 or more patients urgently need that ventilator
            (state INTUBATED_UNSTABLE with hemodynamic instability,
             or REINTUBATED within this step)

        Returns an EthicalTriageCase for the agent's Observation if active,
        None otherwise. Only one case can be active per episode step.
        """
        # If a case is already active and unresolved, return it again
        if self._active_case_id is not None:
            record = self._records.get(self._active_case_id)
            if record and not record.agent_responded:
                return self._build_case_from_record(record)

        if available_ventilators != 1:
            return None

        # Find patients urgently needing a ventilator
        urgent_patients = [
            p for p in patients
            if (
                p.state == PatientState.INTUBATED_UNSTABLE
                and p.support_level != SupportLevel.FULL_VENTILATOR
                and not p.is_discharged
            )
        ]

        if len(urgent_patients) < 2:
            return None

        # Take the two most urgent (highest SOFA = worst prognosis pair)
        # We pick the pair to maximise drama and clinical realism
        urgent_patients.sort(key=lambda p: p.sofa_score, reverse=True)
        patient_a = urgent_patients[0]
        patient_b = urgent_patients[1]

        wait_a = self.get_wait_hours(patient_a.patient_id)
        wait_b = self.get_wait_hours(patient_b.patient_id)

        correct_id, reason = self._determine_correct_patient(
            patient_a.patient_id, patient_b.patient_id,
            patient_a.sofa_score, patient_b.sofa_score,
            wait_a, wait_b,
        )

        case_id = self._next_case_id()
        record = TriageRecord(
            case_id=case_id,
            hour=self.current_hour,
            patient_a_id=patient_a.patient_id,
            patient_b_id=patient_b.patient_id,
            patient_a_sofa=patient_a.sofa_score,
            patient_b_sofa=patient_b.sofa_score,
            patient_a_wait_hours=wait_a,
            patient_b_wait_hours=wait_b,
            correct_patient_id=correct_id,
            correct_reason=reason,
        )
        self._records[case_id] = record
        self._active_case_id = case_id

        return self._build_case_from_record(record)

    def _build_case_from_record(self, record: TriageRecord) -> EthicalTriageCase:
        """Build the agent-visible EthicalTriageCase from an internal record."""
        return EthicalTriageCase(
            case_id=record.case_id,
            patient_a_id=record.patient_a_id,
            patient_b_id=record.patient_b_id,
            patient_a_sofa=record.patient_a_sofa,
            patient_b_sofa=record.patient_b_sofa,
            patient_a_wait_hours=record.patient_a_wait_hours,
            patient_b_wait_hours=record.patient_b_wait_hours,
            resource_type=SupportLevel.FULL_VENTILATOR,
        )

    # =========================================================================
    # INJECT FORCED TRIAGE (Task 3 crisis at hour 10)
    # =========================================================================

    def inject_forced_triage(
        self,
        patient_a_id: str,
        patient_b_id: str,
        patient_a_sofa: int,
        patient_b_sofa: int,
    ) -> EthicalTriageCase:
        """
        Force a triage event at a specific hour regardless of organic conditions.
        Used by task3_crisis.py to inject the scripted crisis at hour 10.
        """
        wait_a = self.get_wait_hours(patient_a_id)
        wait_b = self.get_wait_hours(patient_b_id)

        correct_id, reason = self._determine_correct_patient(
            patient_a_id, patient_b_id,
            patient_a_sofa, patient_b_sofa,
            wait_a, wait_b,
        )

        case_id = self._next_case_id()
        record = TriageRecord(
            case_id=case_id,
            hour=self.current_hour,
            patient_a_id=patient_a_id,
            patient_b_id=patient_b_id,
            patient_a_sofa=patient_a_sofa,
            patient_b_sofa=patient_b_sofa,
            patient_a_wait_hours=wait_a,
            patient_b_wait_hours=wait_b,
            correct_patient_id=correct_id,
            correct_reason=reason,
        )
        self._records[case_id] = record
        self._active_case_id = case_id

        return self._build_case_from_record(record)

    # =========================================================================
    # RECORD AND SCORE AGENT DECISION
    # =========================================================================

    def record_agent_decision(
        self,
        case_id: str,
        chosen_patient_id: str,
    ) -> Tuple[float, str]:
        """
        Record the agent's triage decision and score it immediately.

        Returns (reward_delta, event_string).
        Clears the active case so no duplicate presentation occurs.
        """
        record = self._records.get(case_id)
        if record is None:
            return 0.0, "triage_case_not_found"

        if record.agent_responded:
            return 0.0, "triage_already_responded"

        record.agent_chosen_id = chosen_patient_id
        record.agent_responded = True
        record.is_correct = (chosen_patient_id == record.correct_patient_id)

        # Clear active case
        if self._active_case_id == case_id:
            self._active_case_id = None

        if record.is_correct:
            reward = config.REWARD_ETHICAL_TRIAGE_CORRECT
            event = (
                f"triage_correct:{case_id}:"
                f"chose_{chosen_patient_id}:"
                f"reason_{record.correct_reason}"
            )
        else:
            reward = config.PENALTY_ETHICAL_TRIAGE_WRONG
            event = (
                f"triage_wrong:{case_id}:"
                f"chose_{chosen_patient_id}:"
                f"correct_was_{record.correct_patient_id}:"
                f"reason_{record.correct_reason}"
            )

        return reward, event

    def resolve_unanswered_cases(self) -> List[Tuple[float, str]]:
        """
        At episode end, penalise any triage cases the agent never answered.
        Called by ward.py when done=True.
        """
        results = []
        for case_id, record in self._records.items():
            if not record.agent_responded:
                record.agent_responded = True
                record.is_correct = False
                record.agent_chosen_id = None
                results.append((
                    config.PENALTY_ETHICAL_TRIAGE_WRONG,
                    f"triage_unanswered:{case_id}"
                ))
        return results

    # =========================================================================
    # ADVANCE HOUR
    # =========================================================================

    def advance_hour(self) -> None:
        """Tick the hour counter. Called by ward.py each step."""
        self.current_hour += 1

    # =========================================================================
    # GRADER STATISTICS
    # =========================================================================

    def get_triage_stats(self) -> dict:
        """
        Compute triage decision statistics across the episode.
        Used by crisis_grader to score ethical triage performance.
        """
        total = len(self._records)
        correct = sum(1 for r in self._records.values() if r.is_correct is True)
        wrong = sum(1 for r in self._records.values() if r.is_correct is False)
        unanswered = sum(
            1 for r in self._records.values() if not r.agent_responded
        )

        accuracy = correct / total if total > 0 else 1.0

        cases_detail = []
        for record in self._records.values():
            cases_detail.append({
                "case_id": record.case_id,
                "hour": record.hour,
                "patient_a": record.patient_a_id,
                "patient_b": record.patient_b_id,
                "sofa_a": record.patient_a_sofa,
                "sofa_b": record.patient_b_sofa,
                "correct_answer": record.correct_patient_id,
                "correct_reason": record.correct_reason,
                "agent_chose": record.agent_chosen_id,
                "is_correct": record.is_correct,
            })

        return {
            "total_cases": total,
            "correct_decisions": correct,
            "wrong_decisions": wrong,
            "unanswered": unanswered,
            "accuracy": round(accuracy, 4),
            "cases": cases_detail,
        }