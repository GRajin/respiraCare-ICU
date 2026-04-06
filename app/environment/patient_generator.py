# app/environment/patient_generator.py
# =============================================================================
# RespiraCare-ICU — Synthetic Patient Generator
#
# Generates reproducible synthetic patients for each task using seeded numpy
# random. The same seed always produces the same ward — critical for the
# Phase 2 judge requirement that scores are reproducible.
#
# Public interface:
#   generate_patient(patient_id, severity, seed)  → PatientStateMachine
#   generate_ward(task_id, seed)                  → List[PatientStateMachine]
#
# Severity bands:
#   LOW    — stable, near extubation-ready, low VAP risk
#   MEDIUM — improving, may be a weaning candidate, moderate risk
#   HIGH   — unstable, high SOFA, high VAP risk, resource-intensive
#
# Task distributions:
#   Task 1 — 5 patients covering all 3 readiness categories deterministically
#   Task 2 — 10 patients, mixed severity, 2 borderline extubation candidates
#   Task 3 — 12 patients, full severity spread, 3 high-acuity, VAP risk seeded
# =============================================================================

from __future__ import annotations

import random
from typing import List

import numpy as np

from app.environment.patient import PatientStateMachine
from app.models import PatientState, SupportLevel, Severity
from app import config


# =============================================================================
# SEVERITY BAND DEFINITIONS
# Each band defines a range for every vital sign.
# generate_patient() samples uniformly within these ranges.
# =============================================================================

SEVERITY_BANDS = {
    Severity.LOW: {
        # Stable patient clearly ready or near-ready for weaning
        "rsbi":                  (42.0,  90.0),
        "pf_ratio":              (240.0, 390.0),
        "rass":                  (-2.0,   0.0),
        "fio2":                  (0.25,   0.42),
        "peep":                  (4.0,    7.0),
        "spo2":                  (95.0,  100.0),
        "heart_rate":            (58.0,   90.0),
        "resp_rate":             (10.0,   18.0),
        "sofa_score":            (1,       6),
        "hours_on_vent":         (26,     72),
        "vap_risk":              (0.0,    0.15),
        "hemodynamically_stable": 0.97,   # probability of being stable
    },
    Severity.MEDIUM: {
        # Improving patient, not yet ready — borderline on several criteria
        "rsbi":                  (80.0,  130.0),
        "pf_ratio":              (160.0, 260.0),
        "rass":                  (-3.0,   0.0),
        "fio2":                  (0.38,   0.58),
        "peep":                  (6.0,   10.0),
        "spo2":                  (91.0,   97.0),
        "heart_rate":            (75.0,  110.0),
        "resp_rate":             (16.0,   26.0),
        "sofa_score":            (5,      12),
        "hours_on_vent":         (18,     52),
        "vap_risk":              (0.05,   0.40),
        "hemodynamically_stable": 0.78,
    },
    Severity.HIGH: {
        # Unstable patient — high SOFA, poor vitals, significant VAP risk
        "rsbi":                  (110.0, 175.0),
        "pf_ratio":              (80.0,  185.0),
        "rass":                  (-5.0,  -2.0),
        "fio2":                  (0.55,   0.95),
        "peep":                  (8.0,   18.0),
        "spo2":                  (83.0,   93.0),
        "heart_rate":            (95.0,  148.0),
        "resp_rate":             (22.0,   38.0),
        "sofa_score":            (10,     22),
        "hours_on_vent":         (8,      36),
        "vap_risk":              (0.25,   0.68),
        "hemodynamically_stable": 0.35,
    },
}


# =============================================================================
# TASK SEVERITY DISTRIBUTION
# How many patients of each severity band each task uses.
# Tuned so Task 1 is classifiable, Task 2 needs planning, Task 3 is brutal.
# =============================================================================

TASK_DISTRIBUTIONS = {
    1: [
        # 5 patients covering all 3 readiness categories
        # 2 clearly NOT_READY, 2 READY_FOR_SBT, 1 READY_TO_EXTUBATE
        {"severity": Severity.HIGH,   "force_state": PatientState.INTUBATED_UNSTABLE},
        {"severity": Severity.MEDIUM, "force_state": PatientState.INTUBATED_STABLE,
         "force_vitals": {"rsbi": 122.0, "pf_ratio": 175.0}},   # NOT_READY
        {"severity": Severity.LOW,    "force_state": PatientState.INTUBATED_STABLE,
         "force_vitals": {"rsbi": 78.0,  "pf_ratio": 255.0, "rass": -1.0}},   # READY_FOR_SBT
        {"severity": Severity.LOW,    "force_state": PatientState.INTUBATED_STABLE,
         "force_vitals": {"rsbi": 62.0,  "pf_ratio": 300.0, "rass": -1.0}},   # READY_FOR_SBT
        {"severity": Severity.LOW,    "force_state": PatientState.READY_TO_EXTUBATE,
         "force_vitals": {"rsbi": 55.0,  "pf_ratio": 330.0, "rass":  0.0},
         "sbt_passed": True},  # READY_TO_EXTUBATE
    ],
    2: [
        # 10 patients — mixed, with trauma incoming at hour 3 needing 3 vents
        {"severity": Severity.LOW,    "force_state": PatientState.INTUBATED_STABLE},
        {"severity": Severity.LOW,    "force_state": PatientState.READY_TO_EXTUBATE, "sbt_passed": True},
        {"severity": Severity.MEDIUM, "force_state": PatientState.INTUBATED_STABLE},
        {"severity": Severity.MEDIUM, "force_state": PatientState.INTUBATED_STABLE},
        {"severity": Severity.MEDIUM, "force_state": PatientState.INTUBATED_STABLE},
        {"severity": Severity.LOW,    "force_state": PatientState.INTUBATED_STABLE,
         "force_vitals": {"rsbi": 88.0, "pf_ratio": 235.0}},    # Borderline — SBT candidate
        {"severity": Severity.HIGH,   "force_state": PatientState.INTUBATED_UNSTABLE},
        {"severity": Severity.HIGH,   "force_state": PatientState.INTUBATED_UNSTABLE},
        {"severity": Severity.MEDIUM, "force_state": PatientState.INTUBATED_WITH_VAP},
        {"severity": Severity.LOW,    "force_state": PatientState.EXTUBATED,
         "support": SupportLevel.BIPAP},
    ],
    3: [
        # 12 patients — full ward, all 5 problems active
        # Structured so crises at hours 3,5,6,7,9,10,11 are meaningful
        {"severity": Severity.LOW,    "force_state": PatientState.INTUBATED_STABLE},
        {"severity": Severity.LOW,    "force_state": PatientState.READY_TO_EXTUBATE, "sbt_passed": True},
        {"severity": Severity.MEDIUM, "force_state": PatientState.INTUBATED_STABLE},
        {"severity": Severity.MEDIUM, "force_state": PatientState.INTUBATED_STABLE},
        {"severity": Severity.LOW,    "force_state": PatientState.INTUBATED_STABLE,
         "force_vitals": {"rsbi": 84.0, "pf_ratio": 248.0, "rass": -1.0}},
        {"severity": Severity.HIGH,   "force_state": PatientState.INTUBATED_UNSTABLE},
        {"severity": Severity.HIGH,   "force_state": PatientState.INTUBATED_UNSTABLE},
        {"severity": Severity.HIGH,   "force_state": PatientState.INTUBATED_WITH_VAP,
         "force_vitals": {"vap_risk": 0.60}},
        {"severity": Severity.MEDIUM, "force_state": PatientState.INTUBATED_STABLE},
        {"severity": Severity.MEDIUM, "force_state": PatientState.EXTUBATED,
         "support": SupportLevel.BIPAP},
        {"severity": Severity.LOW,    "force_state": PatientState.EXTUBATED,
         "support": SupportLevel.HFNC},
        {"severity": Severity.HIGH,   "force_state": PatientState.INTUBATED_UNSTABLE,
         "force_vitals": {"sofa_score": 16, "vap_risk": 0.45}},
    ],
}


# =============================================================================
# CORE GENERATOR FUNCTIONS
# =============================================================================

def generate_patient(
    patient_id: str,
    severity: Severity,
    seed: int,
    force_state: PatientState = PatientState.INTUBATED_STABLE,
    force_vitals: dict = None,
    sbt_passed: bool = False,
    support: SupportLevel = SupportLevel.FULL_VENTILATOR,
) -> PatientStateMachine:
    """
    Generate a single synthetic patient with reproducible randomness.

    Args:
        patient_id:   Unique ID string, e.g. 'P01'
        severity:     Severity band — LOW / MEDIUM / HIGH
        seed:         Integer seed for this patient's RNG
        force_state:  Override the patient's initial clinical state
        force_vitals: Dict of vital overrides applied after sampling
        sbt_passed:   If True, marks patient as having passed an SBT at hour 0
        support:      Override the support level (e.g. BIPAP for stepped-down patients)

    Returns:
        A fully initialised PatientStateMachine ready to be placed in the ward.
    """
    rng = random.Random(seed)
    band = SEVERITY_BANDS[severity]

    def sample(key: str) -> float:
        lo, hi = band[key]
        return rng.uniform(lo, hi)

    def sample_int(key: str) -> int:
        lo, hi = band[key]
        return rng.randint(lo, hi)

    # --- Sample vitals from severity band ---
    rsbi        = round(sample("rsbi"), 1)
    pf_ratio    = round(sample("pf_ratio"), 1)
    rass        = round(sample("rass"), 1)
    fio2        = round(sample("fio2"), 2)
    peep        = round(sample("peep"), 1)
    spo2        = round(sample("spo2"), 1)
    heart_rate  = round(sample("heart_rate"), 1)
    resp_rate   = round(sample("resp_rate"), 1)
    sofa_score  = sample_int("sofa_score")
    hours_on_vent = sample_int("hours_on_vent")
    vap_risk    = round(sample("vap_risk"), 3)
    hemo_stable = rng.random() < band["hemodynamically_stable"]

    # --- Apply forced vital overrides ---
    if force_vitals:
        if "rsbi"        in force_vitals: rsbi        = force_vitals["rsbi"]
        if "pf_ratio"    in force_vitals: pf_ratio    = force_vitals["pf_ratio"]
        if "rass"        in force_vitals: rass        = force_vitals["rass"]
        if "fio2"        in force_vitals: fio2        = force_vitals["fio2"]
        if "peep"        in force_vitals: peep        = force_vitals["peep"]
        if "spo2"        in force_vitals: spo2        = force_vitals["spo2"]
        if "heart_rate"  in force_vitals: heart_rate  = force_vitals["heart_rate"]
        if "resp_rate"   in force_vitals: resp_rate   = force_vitals["resp_rate"]
        if "sofa_score"  in force_vitals: sofa_score  = int(force_vitals["sofa_score"])
        if "hours_on_vent" in force_vitals: hours_on_vent = int(force_vitals["hours_on_vent"])
        if "vap_risk"    in force_vitals: vap_risk    = force_vitals["vap_risk"]

    # --- Patients not on full vent don't have RSBI (requires ET tube) ---
    if support != SupportLevel.FULL_VENTILATOR:
        rsbi = None

    # --- Patients in clearly non-ready states lose RSBI/PF visibility ---
    if force_state == PatientState.INTUBATED_UNSTABLE:
        # Unstable patients often can't cooperate with RSBI measurement
        if rng.random() < 0.40:
            rsbi = None

    # --- Build the state machine ---
    patient = PatientStateMachine(
        patient_id=patient_id,
        state=force_state,
        support_level=support,
        severity=severity,
        rsbi=rsbi,
        pf_ratio=pf_ratio,
        rass=rass,
        fio2=fio2,
        peep=peep,
        spo2=spo2,
        heart_rate=heart_rate,
        resp_rate=resp_rate,
        hemodynamically_stable=hemo_stable,
        hours_on_vent=hours_on_vent,
        sofa_score=sofa_score,
        vap_risk=vap_risk,
        rng=rng,
    )

    # --- Mark SBT as passed if required ---
    if sbt_passed:
        patient.sbt_passed_at_hour = 0   # Passed before episode started

    return patient


def generate_ward(task_id: int, seed: int) -> List[PatientStateMachine]:
    """
    Generate the full list of patients for a given task and seed.

    The same task_id + seed always produces the exact same ward state.
    Different seeds produce different (but valid) ward configurations.

    Args:
        task_id:  1, 2, or 3
        seed:     Master seed for the episode

    Returns:
        List of PatientStateMachine instances in bed order (P01, P02, ...)
    """
    if task_id not in TASK_DISTRIBUTIONS:
        raise ValueError(f"Unknown task_id: {task_id}. Must be 1, 2, or 3.")

    distribution = TASK_DISTRIBUTIONS[task_id]
    patients: List[PatientStateMachine] = []

    # Use numpy to derive per-patient seeds from the master seed
    # This ensures seeds are well-spread and don't collide
    np_rng = np.random.default_rng(seed)
    per_patient_seeds = np_rng.integers(
        low=1000, high=999999, size=len(distribution)
    ).tolist()

    for i, spec in enumerate(distribution):
        patient_id = f"P{i+1:02d}"
        patient_seed = per_patient_seeds[i]

        patient = generate_patient(
            patient_id=patient_id,
            severity=spec["severity"],
            seed=patient_seed,
            force_state=spec.get("force_state", PatientState.INTUBATED_STABLE),
            force_vitals=spec.get("force_vitals", None),
            sbt_passed=spec.get("sbt_passed", False),
            support=spec.get("support", SupportLevel.FULL_VENTILATOR),
        )

        patients.append(patient)

    return patients


def get_task_description(task_id: int) -> str:
    """
    Returns the plain-English task description sent to the agent in ResetResponse.
    """
    descriptions = {
        1: (
            "TASK 1 — Readiness Triage (Easy)\n"
            "You are a Charge Respiratory Therapist reviewing 5 ventilated patients.\n"
            "For each patient, classify their readiness using ACCP weaning criteria:\n"
            "  - READY_FOR_SBT: RSBI < 105, PF ratio > 200, RASS -2 to 0, hemodynamically stable,\n"
            "                    FiO2 ≤ 0.50, PEEP ≤ 8, on vent ≥ 24h\n"
            "  - READY_TO_EXTUBATE: All above criteria met AND SBT passed within last 2 hours\n"
            "  - NOT_READY: Any criterion failed\n"
            "Submit one action per patient: attempt_sbt (if READY_FOR_SBT), "
            "extubate (if READY_TO_EXTUBATE), or hold_and_monitor (if NOT_READY).\n"
            "Score = correct classifications / 5."
        ),
        2: (
            "TASK 2 — Shift Resource Optimization (Medium)\n"
            "You are managing a 10-patient ICU ward over 6 hours.\n"
            "CRITICAL: 3 trauma patients are arriving in 3 hours and each needs a ventilator.\n"
            "You currently have limited free ventilators — you must free equipment by then.\n"
            "Each hour, submit one action per patient. Priorities:\n"
            "  1. Identify and extubate ready patients to free ventilators\n"
            "  2. Filter alarms — respond to real ones, suppress false positives\n"
            "  3. Maintain VAP bundle compliance for all ventilated patients\n"
            "  4. Assign RT attention to the highest-acuity patients\n"
            "Graded on: equipment availability at hour 3 (35%), reintubation rate (25%), "
            "VAP compliance (20%), alarm accuracy (20%)."
        ),
        3: (
            "TASK 3 — Full Shift Crisis Management (Hard)\n"
            "You are managing a full 12-patient ICU ward across a complete 12-hour shift.\n"
            "All five problem dimensions are active simultaneously.\n"
            "Crisis events will be injected at specific hours — you must anticipate and respond.\n"
            "Known crisis schedule:\n"
            "  Hour 3:  Mass casualty event — 4 patients arriving in 2 hours, need ventilators\n"
            "  Hour 5:  One ventilator malfunctions — reduced equipment capacity\n"
            "  Hour 6:  Shift handover — some patient observations will be degraded\n"
            "  Hour 7:  BiPAP patient crashes — emergency re-intubation required\n"
            "  Hour 9:  Two patients show VAP signs\n"
            "  Hour 10: Ethical triage — two patients need the one available ventilator\n"
            "  Hour 11: One RT goes offline — only 2 staff for the final hours\n"
            "Graded on: mortality proxy (30%), equipment utilization (20%), "
            "VAP incidence (20%), reintubation quality (15%), crisis response speed (15%)."
        ),
    }
    return descriptions.get(task_id, f"Task {task_id} description not available.")