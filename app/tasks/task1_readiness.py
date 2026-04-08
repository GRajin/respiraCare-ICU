# app/tasks/task1_readiness.py
# =============================================================================
# RespiraCare-ICU — Task 1: Readiness Triage (Easy)
#
# The simplest task. 5 patients, 1 step, deterministic grader.
# The agent must classify each patient into exactly one of three categories
# using published ACCP weaning readiness criteria.
#
# This module provides:
#   get_task_config()     → dict of task parameters used by Episode/ward
#   describe_patient()    → human-readable patient summary for LLM prompt
#   format_prompt()       → full prompt string for inference.py
#
# The actual patient generation is handled by patient_generator.py
# (TASK_DISTRIBUTIONS[1]). This module focuses on prompt formatting
# and task metadata.
# =============================================================================

from __future__ import annotations

from typing import List

from app.models import Observation, PatientObservation
from app import config


# =============================================================================
# TASK METADATA
# =============================================================================

TASK_ID = 1
TASK_NAME = "Readiness Triage"
TASK_DIFFICULTY = "easy"
TASK_MAX_STEPS = 1
EXPECTED_SCORE_MIN = 0.65
EXPECTED_SCORE_MAX = 0.90


def get_task_config() -> dict:
    """Returns the task configuration dict for openenv.yaml and grader use."""
    return {
        "task_id": TASK_ID,
        "name": TASK_NAME,
        "difficulty": TASK_DIFFICULTY,
        "max_steps": TASK_MAX_STEPS,
        "num_patients": config.TASK1_NUM_PATIENTS,
        "expected_score_min": EXPECTED_SCORE_MIN,
        "expected_score_max": EXPECTED_SCORE_MAX,
        "grader": "readiness_grader",
        "description": (
            "Classify 5 ventilated ICU patients into readiness categories "
            "using ACCP weaning criteria. Single-step episode."
        ),
    }


# =============================================================================
# ACCP CRITERIA EVALUATION
# Used by both the grader and the prompt formatter.
# =============================================================================

def evaluate_patient_readiness(patient: PatientObservation) -> str:
    """
    Apply ACCP weaning readiness criteria to determine the correct
    classification for a patient.

    Returns one of:
      'READY_TO_EXTUBATE'  — SBT passed + all criteria met
      'READY_FOR_SBT'      — all criteria met, no SBT yet
      'NOT_READY'          — one or more criteria failed

    This is the ground-truth function used by the grader.
    The agent must reach the same conclusion from the observation.
    """
    # --- Prerequisite checks ---
    if patient.rsbi is None or patient.pf_ratio is None:
        return "NOT_READY"

    if patient.rass is None:
        return "NOT_READY"

    # --- ACCP criteria ---
    rsbi_ok    = patient.rsbi < config.RSBI_THRESHOLD
    pf_ok      = patient.pf_ratio > config.PF_RATIO_THRESHOLD
    rass_ok    = config.RASS_MIN <= patient.rass <= config.RASS_MAX
    hemo_ok    = patient.hemodynamically_stable
    fio2_ok    = patient.fio2 <= config.FIO2_MAX_FOR_SBT
    peep_ok    = patient.peep <= config.PEEP_MAX_FOR_SBT
    vent_ok    = patient.hours_on_vent >= config.HOURS_ON_VENT_BEFORE_SBT

    all_criteria_met = rsbi_ok and pf_ok and rass_ok and hemo_ok and fio2_ok and peep_ok and vent_ok

    if not all_criteria_met:
        return "NOT_READY"

    # --- Extubation check: SBT passed within valid window ---
    sbt_recent = (
        patient.sbt_passed_within_hours is not None
        and patient.sbt_passed_within_hours <= config.SBT_DURATION_HOURS
    )

    if sbt_recent:
        return "READY_TO_EXTUBATE"

    return "READY_FOR_SBT"


def get_correct_action(readiness: str) -> str:
    """
    Map a readiness classification to the correct ActionType string.
    Used by grader to check agent's action against ground truth.
    """
    mapping = {
        "READY_TO_EXTUBATE": "extubate",
        "READY_FOR_SBT":     "attempt_sbt",
        "NOT_READY":         "hold_and_monitor",
    }
    return mapping.get(readiness, "hold_and_monitor")


# =============================================================================
# PROMPT FORMATTING
# Used by inference.py to build the LLM prompt for Task 1.
# =============================================================================

def describe_patient(patient: PatientObservation, index: int) -> str:
    """
    Format a single patient's data as a readable table row for the LLM prompt.
    """
    rsbi_str    = f"{patient.rsbi:.1f}" if patient.rsbi is not None else "N/A"
    pf_str      = f"{patient.pf_ratio:.1f}" if patient.pf_ratio is not None else "N/A"
    rass_str    = f"{patient.rass:.1f}" if patient.rass is not None else "N/A"
    sbt_str     = (
        f"{patient.sbt_passed_within_hours}h ago"
        if patient.sbt_passed_within_hours is not None
        else "None"
    )

    return (
        f"Patient {patient.patient_id}:\n"
        f"  State              : {patient.state.value}\n"
        f"  RSBI               : {rsbi_str}  (threshold: <{config.RSBI_THRESHOLD})\n"
        f"  P/F ratio          : {pf_str}  (threshold: >{config.PF_RATIO_THRESHOLD})\n"
        f"  RASS               : {rass_str}  (acceptable: {config.RASS_MIN} to {config.RASS_MAX})\n"
        f"  FiO2               : {patient.fio2:.2f}  (max for SBT: {config.FIO2_MAX_FOR_SBT})\n"
        f"  PEEP               : {patient.peep:.1f}  (max for SBT: {config.PEEP_MAX_FOR_SBT})\n"
        f"  Hemodynamically stable: {patient.hemodynamically_stable}\n"
        f"  Hours on ventilator: {patient.hours_on_vent}  (min: {config.HOURS_ON_VENT_BEFORE_SBT})\n"
        f"  SBT passed         : {sbt_str}\n"
    )


def format_prompt(observation: Observation) -> str:
    """
    Build the complete LLM prompt for Task 1.
    Used by inference.py when running Task 1.
    """
    patient_sections = "\n".join(
        describe_patient(p, i)
        for i, p in enumerate(observation.patients)
    )

    return f"""You are a Charge Respiratory Therapist in an ICU.
Your task is to classify each of the following {len(observation.patients)} ventilated patients
into exactly one readiness category using ACCP weaning criteria.

ACCP WEANING READINESS CRITERIA:
  READY_FOR_SBT requires ALL of:
    - RSBI < {config.RSBI_THRESHOLD} breaths/min/L
    - P/F ratio > {config.PF_RATIO_THRESHOLD} mmHg
    - RASS between {config.RASS_MIN} and {config.RASS_MAX}
    - Hemodynamically stable (no vasopressors)
    - FiO2 <= {config.FIO2_MAX_FOR_SBT}
    - PEEP <= {config.PEEP_MAX_FOR_SBT} cmH2O
    - On ventilator >= {config.HOURS_ON_VENT_BEFORE_SBT} hours

  READY_TO_EXTUBATE requires ALL of the above PLUS:
    - SBT passed within the last {config.SBT_DURATION_HOURS} hours

  NOT_READY: any criterion above is failed or data is missing (N/A)

PATIENT DATA:
{patient_sections}

INSTRUCTIONS:
For each patient, submit exactly one action:
  - If READY_TO_EXTUBATE  → action_type: "extubate"
  - If READY_FOR_SBT      → action_type: "attempt_sbt"
  - If NOT_READY          → action_type: "hold_and_monitor"

Respond with a JSON array of action objects. Example format:
[
  {{"patient_id": "P01", "action_type": "hold_and_monitor", "priority": 2}},
  {{"patient_id": "P02", "action_type": "attempt_sbt", "priority": 1}},
  {{"patient_id": "P03", "action_type": "extubate", "priority": 1}}
]

Return ONLY the JSON array. No explanation, no markdown, no preamble.
"""