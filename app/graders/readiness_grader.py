# app/graders/readiness_grader.py
# =============================================================================
# RespiraCare-ICU — Task 1 Grader: Readiness Triage
#
# Scores the agent's readiness classifications against the ACCP ground truth.
#
# Grading logic:
#   For each of the 5 patients:
#     - Compute the correct classification using evaluate_patient_readiness()
#     - Check what action the agent took
#     - Award 1.0 for correct, 0.5 for adjacent (e.g. attempt_sbt when
#       READY_TO_EXTUBATE), 0.0 for wrong
#   Score = sum of per-patient scores / 5
#
# The grader is fully deterministic — same episode state → same score.
# It does not require the episode to be done (grader can be called mid-episode
# for Task 1 since it's a single-step task).
# =============================================================================

from __future__ import annotations

from typing import List, Dict

from app.environment.episode import Episode
from app.models import ActionType
from app.tasks.task1_readiness import evaluate_patient_readiness, get_correct_action
from app.reward.reward_function import clamp_score


# =============================================================================
# PARTIAL CREDIT TABLE
# Adjacent classifications get 0.5 — the clinical reasoning was close.
# Opposite classifications (e.g. extubate a NOT_READY patient) get 0.0.
# =============================================================================

PARTIAL_CREDIT = {
    # (correct_action, agent_action) → score
    ("extubate",          "attempt_sbt"):      0.5,   # Close — almost ready
    ("attempt_sbt",       "extubate"):          0.3,   # Slightly aggressive
    ("attempt_sbt",       "hold_and_monitor"):  0.3,   # Too conservative
    ("hold_and_monitor",  "attempt_sbt"):       0.2,   # Too aggressive
    ("extubate",          "hold_and_monitor"):  0.0,   # Clearly wrong — missed ready patient
    ("hold_and_monitor",  "extubate"):          0.0,   # Dangerous — extubating not-ready patient
}


def grade(episode: Episode) -> Dict:
    """
    Grade a completed Task 1 episode.

    Args:
        episode: A completed Episode object (step_count >= 1)

    Returns:
        A dict with:
          score         — final grade in [0.0, 1.0]
          per_patient   — per-patient breakdown
          correct_count — number of fully correct classifications
          details       — human-readable summary
    """
    if not episode.history:
        return {
            "score": 0.0001,
            "per_patient": [],
            "correct_count": 0,
            "details": "No steps recorded — agent did not act.",
        }

    # Task 1 is a single step — use the first (and only) step record
    step_record = episode.history[0]

    # Use the observation stored at reset time — patient states have
    # already changed by the time grade() is called after step().
    initial_obs = episode.initial_observation
    if initial_obs is None:
        return {
            "score": 0.0001,
            "per_patient": [],
            "correct_count": 0,
            "details": "No initial observation stored — was reset() called?",
        }

    # Build action lookup: patient_id → action_type string
    action_map: Dict[str, str] = {}
    for action in step_record.actions:
        action_map[action.patient_id] = action.action_type.value

    per_patient = []
    total_score = 0.0
    correct_count = 0

    for patient_obs in initial_obs.patients:
        pid = patient_obs.patient_id

        # Ground truth
        correct_readiness = evaluate_patient_readiness(patient_obs)
        correct_action = get_correct_action(correct_readiness)

        # Agent's action (default to hold_and_monitor if not submitted)
        agent_action = action_map.get(pid, "hold_and_monitor")

        # Score this patient
        if agent_action == correct_action:
            patient_score = 1.0
            correct_count += 1
            verdict = "correct"
        else:
            # Check for partial credit
            partial_key = (correct_action, agent_action)
            patient_score = PARTIAL_CREDIT.get(partial_key, 0.0)
            verdict = "partial" if patient_score > 0 else "wrong"

        total_score += patient_score

        per_patient.append({
            "patient_id": pid,
            "correct_readiness": correct_readiness,
            "correct_action": correct_action,
            "agent_action": agent_action,
            "score": patient_score,
            "verdict": verdict,
            "rsbi": patient_obs.rsbi,
            "pf_ratio": patient_obs.pf_ratio,
            "rass": patient_obs.rass,
            "hemodynamically_stable": patient_obs.hemodynamically_stable,
            "sbt_passed_within_hours": patient_obs.sbt_passed_within_hours,
        })

    # Final score = average across 5 patients
    n_patients = len(initial_obs.patients)
    final_score = clamp_score(total_score / n_patients if n_patients > 0 else 0.0)

    return {
        "score": final_score,
        "per_patient": per_patient,
        "correct_count": correct_count,
        "total_patients": n_patients,
        "details": (
            f"Task 1 score: {final_score:.3f} "
            f"({correct_count}/{n_patients} fully correct)"
        ),
    }


def grade_from_actions(
    observation_patients: list,
    actions: List[Dict],
) -> Dict:
    """
    Lightweight grader that works directly from patient observations and
    a list of action dicts. Used by inference.py for quick scoring without
    a full Episode object.

    Args:
        observation_patients: list of PatientObservation objects
        actions: list of dicts with patient_id and action_type keys

    Returns same structure as grade().
    """
    action_map = {a["patient_id"]: a["action_type"] for a in actions}

    per_patient = []
    total_score = 0.0
    correct_count = 0

    for patient_obs in observation_patients:
        pid = patient_obs.patient_id
        correct_readiness = evaluate_patient_readiness(patient_obs)
        correct_action = get_correct_action(correct_readiness)
        agent_action = action_map.get(pid, "hold_and_monitor")

        if agent_action == correct_action:
            patient_score = 1.0
            correct_count += 1
            verdict = "correct"
        else:
            partial_key = (correct_action, agent_action)
            patient_score = PARTIAL_CREDIT.get(partial_key, 0.0)
            verdict = "partial" if patient_score > 0 else "wrong"

        total_score += patient_score
        per_patient.append({
            "patient_id": pid,
            "correct_readiness": correct_readiness,
            "correct_action": correct_action,
            "agent_action": agent_action,
            "score": patient_score,
            "verdict": verdict,
        })

    n = len(observation_patients)
    final_score = clamp_score(total_score / n if n > 0 else 0.0)

    return {
        "score": final_score,
        "per_patient": per_patient,
        "correct_count": correct_count,
        "total_patients": n,
        "details": f"Task 1 score: {final_score:.3f} ({correct_count}/{n} correct)",
    }