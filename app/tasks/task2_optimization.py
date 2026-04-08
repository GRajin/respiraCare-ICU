# app/tasks/task2_optimization.py
# =============================================================================
# RespiraCare-ICU — Task 2: Shift Resource Optimization (Medium)
#
# 10 patients, 6-hour episode, incoming trauma at hour 3.
# The agent must balance extubation decisions, alarm filtering,
# VAP bundle compliance, and equipment availability simultaneously.
#
# This module provides:
#   get_task_config()   → task metadata dict
#   format_prompt()     → full LLM prompt for inference.py
# =============================================================================

from __future__ import annotations

from app.models import Observation
from app import config


TASK_ID = 2
TASK_NAME = "Shift Resource Optimization"
TASK_DIFFICULTY = "medium"
TASK_MAX_STEPS = 6
EXPECTED_SCORE_MIN = 0.35
EXPECTED_SCORE_MAX = 0.65


def get_task_config() -> dict:
    return {
        "task_id": TASK_ID,
        "name": TASK_NAME,
        "difficulty": TASK_DIFFICULTY,
        "max_steps": TASK_MAX_STEPS,
        "num_patients": config.TASK2_NUM_PATIENTS,
        "expected_score_min": EXPECTED_SCORE_MIN,
        "expected_score_max": EXPECTED_SCORE_MAX,
        "grader": "shift_grader",
        "description": (
            "Manage a 10-patient ICU ward over 6 hours. "
            "3 trauma patients arrive at hour 3 needing ventilators. "
            "Balance extubations, alarms, VAP compliance, and RT allocation."
        ),
    }


def format_prompt(observation: Observation) -> str:
    """
    Build the complete LLM prompt for one step of Task 2.
    Called each step by inference.py.
    """
    # --- Ward resources ---
    resource_section = (
        f"WARD RESOURCES:\n"
        f"  Ventilators available : {observation.available_ventilators} / {config.WARD_VENTILATORS}\n"
        f"  BiPAP units available : {observation.available_bipap} / {config.WARD_BIPAP_UNITS}\n"
        f"  HFNC units available  : {observation.available_hfnc} / {config.WARD_HFNC_UNITS}\n"
        f"  Respiratory therapists: {observation.available_rts} / {config.WARD_RESPIRATORY_THERAPISTS}\n"
    )

    # --- Incoming patients ---
    if observation.incoming_patients:
        incoming_lines = []
        for alert in observation.incoming_patients:
            incoming_lines.append(
                f"  {alert.alert_id}: arrives in {alert.eta_hours}h — "
                f"needs {alert.support_needed.value} — "
                f"severity {alert.severity_estimate.value}"
            )
        incoming_section = "INCOMING PATIENTS (need equipment ready):\n" + "\n".join(incoming_lines)
    else:
        incoming_section = "INCOMING PATIENTS: None announced."

    # --- Patient table ---
    patient_lines = []
    for p in observation.patients:
        rsbi_str  = f"{p.rsbi:.1f}" if p.rsbi is not None else "N/A"
        pf_str    = f"{p.pf_ratio:.1f}" if p.pf_ratio is not None else "N/A"
        rass_str  = f"{p.rass:.1f}" if p.rass is not None else "N/A"
        sbt_str   = f"{p.sbt_passed_within_hours}h ago" if p.sbt_passed_within_hours is not None else "None"
        vap_str   = f"{p.vap_risk:.2f}"
        degrad    = " [DEGRADED]" if p.observation_degraded else ""

        patient_lines.append(
            f"  {p.patient_id}{degrad}: {p.state.value} | {p.support_level.value}\n"
            f"    RSBI={rsbi_str} PF={pf_str} RASS={rass_str} "
            f"FiO2={p.fio2:.2f} PEEP={p.peep:.1f}\n"
            f"    Hemo-stable={p.hemodynamically_stable} "
            f"Hours-on-vent={p.hours_on_vent} SBT={sbt_str}\n"
            f"    VAP-risk={vap_str} Bundle-hours={p.vap_bundle_compliance_hours} "
            f"SOFA={p.sofa_score} Severity={p.severity.value}"
        )
    patient_section = "PATIENTS:\n" + "\n".join(patient_lines)

    # --- Active alarms ---
    if observation.active_alarms:
        alarm_lines = []
        for alarm in observation.active_alarms:
            alarm_lines.append(
                f"  {alarm.alarm_id}: {alarm.patient_id} — "
                f"{alarm.alarm_type.value} — "
                f"consecutive_same_type={alarm.consecutive_same_type}"
            )
        alarm_section = (
            "ACTIVE ALARMS (you cannot see which are real — use cry-wolf pattern):\n"
            + "\n".join(alarm_lines)
        )
    else:
        alarm_section = "ACTIVE ALARMS: None."

    # --- Triage ---
    triage_section = ""
    if observation.triage_decision_required:
        t = observation.triage_decision_required
        triage_section = (
            f"\nETHICAL TRIAGE REQUIRED (case {t.case_id}):\n"
            f"  {t.patient_a_id}: SOFA={t.patient_a_sofa} wait={t.patient_a_wait_hours}h\n"
            f"  {t.patient_b_id}: SOFA={t.patient_b_sofa} wait={t.patient_b_wait_hours}h\n"
            f"  Only ONE ventilator available. Lower SOFA = better prognosis = priority.\n"
        )

    return f"""You are a Charge Respiratory Therapist managing a {config.TASK2_NUM_PATIENTS}-patient ICU ward.
Current shift hour: {observation.shift_hour} / {config.TASK2_MAX_HOURS}

CRITICAL OBJECTIVE: 3 trauma patients are arriving at hour 3 needing ventilators.
You must free enough ventilators before they arrive.

{resource_section}
{incoming_section}

{patient_section}

{alarm_section}
{triage_section}

ACCP WEANING CRITERIA (for extubation decisions):
  READY_FOR_SBT    : RSBI<{config.RSBI_THRESHOLD}, PF>{config.PF_RATIO_THRESHOLD}, RASS {config.RASS_MIN} to {config.RASS_MAX}, hemo-stable, FiO2<={config.FIO2_MAX_FOR_SBT}, PEEP<={config.PEEP_MAX_FOR_SBT}, vent>={config.HOURS_ON_VENT_BEFORE_SBT}h
  READY_TO_EXTUBATE: All above + SBT passed within {config.SBT_DURATION_HOURS}h

AVAILABLE ACTIONS PER PATIENT:
  attempt_sbt           — try spontaneous breathing trial (needs READY_FOR_SBT criteria)
  extubate              — remove tube, free ventilator (needs READY_TO_EXTUBATE)
  step_down_to_bipap    — move from vent to BiPAP (frees vent, needs BiPAP available)
  step_down_to_hfnc     — move from BiPAP/vent to HFNC (needs HFNC available)
  escalate_to_full_vent — re-intubate deteriorating patient (needs free vent)
  assign_rt_attention   — dedicate an RT to this patient (use for highest acuity)
  enforce_vap_bundle    — perform VAP prevention steps (use for vented patients >48h)
  respond_to_alarm      — treat alarm as real and intervene
  suppress_alarm        — treat alarm as false positive
  hold_and_monitor      — no change, continue monitoring
  ethical_triage_select — select patient for scarce resource (only when triage required)

STRATEGY HINTS:
  - Extubate ready patients NOW to free vents before hour 3
  - High consecutive_same_type alarms are likely false positives (cry-wolf pattern)
  - Enforce VAP bundle for patients on vent >48h with rising vap_risk
  - Assign RT to INTUBATED_UNSTABLE patients first

Respond with a JSON array — one action per patient. Include ALL {len(observation.patients)} patients.
Return ONLY valid JSON. No explanation, no markdown.

Example format:
[
  {{"patient_id": "P01", "action_type": "extubate", "priority": 1}},
  {{"patient_id": "P02", "action_type": "enforce_vap_bundle", "priority": 2}},
  {{"patient_id": "P03", "action_type": "hold_and_monitor", "priority": 3}}
]
"""