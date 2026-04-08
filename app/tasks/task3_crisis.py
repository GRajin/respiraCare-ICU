# app/tasks/task3_crisis.py
# =============================================================================
# RespiraCare-ICU — Task 3: Full Shift Crisis Management (Hard)
#
# 12 patients, 12-hour episode, 7 scripted crisis events injected at
# predetermined hours. All 5 problem dimensions active simultaneously.
#
# Crisis schedule (from config.TASK3_CRISIS_SCHEDULE):
#   Hour 3:  mass_casualty_alert    — 4 patients arriving in 2 hours
#   Hour 5:  ventilator_malfunction — 1 vent offline (7/8 available)
#   Hour 6:  shift_handover         — observation partially degraded
#   Hour 7:  bipap_crash            — BiPAP patient needs emergency re-intubation
#   Hour 9:  vap_outbreak           — 2 patients show VAP signs
#   Hour 10: ethical_triage         — 2 patients need 1 available vent
#   Hour 11: rt_unavailable         — 1 RT offline for final 2 hours
#
# This module provides:
#   get_task_config()        → task metadata
#   apply_crisis(ward, hour) → inject scripted crisis events into ward
#   format_prompt()          → full LLM prompt for inference.py
# =============================================================================

from __future__ import annotations

from typing import Optional

from app.models import Observation, SupportLevel, PatientState
from app import config


TASK_ID = 3
TASK_NAME = "Full Shift Crisis Management"
TASK_DIFFICULTY = "hard"
TASK_MAX_STEPS = 12
EXPECTED_SCORE_MIN = 0.10
EXPECTED_SCORE_MAX = 0.35


def get_task_config() -> dict:
    return {
        "task_id": TASK_ID,
        "name": TASK_NAME,
        "difficulty": TASK_DIFFICULTY,
        "max_steps": TASK_MAX_STEPS,
        "num_patients": config.TASK3_NUM_PATIENTS,
        "expected_score_min": EXPECTED_SCORE_MIN,
        "expected_score_max": EXPECTED_SCORE_MAX,
        "grader": "crisis_grader",
        "description": (
            "Manage a 12-patient ICU ward across a complete 12-hour shift "
            "with 7 scripted crisis events. All 5 problem dimensions active."
        ),
    }


# =============================================================================
# CRISIS INJECTION
# Called by ward.py/episode.py at the start of each hour tick.
# =============================================================================

def apply_crisis(ward, hour: int) -> list:
    """
    Inject scripted crisis events into the ward at the appropriate hour.
    Called by episode.py after advance_hour() each step.

    Args:
        ward: WardStateManager instance
        hour: current hour AFTER advancing (post-tick hour)

    Returns list of crisis event strings for the episode log.
    """
    crisis_type = config.TASK3_CRISIS_SCHEDULE.get(hour)
    if crisis_type is None:
        return []

    events = []

    # -----------------------------------------------------------------
    # Hour 3: Mass casualty alert — 4 patients arriving in 2 hours
    # The FleetManager already schedules these in its incoming_records.
    # We just log the event here for the crisis grader.
    # -----------------------------------------------------------------
    if crisis_type == "mass_casualty_alert":
        events.append("crisis:mass_casualty_alert:4_patients_in_2h")

    # -----------------------------------------------------------------
    # Hour 5: Ventilator malfunction — reduce total vents by 1
    # -----------------------------------------------------------------
    elif crisis_type == "ventilator_malfunction":
        if ward.fleet.equipment.total_ventilators > 7:
            ward.fleet.equipment.total_ventilators -= 1
            events.append(
                f"crisis:ventilator_malfunction:"
                f"total_now={ward.fleet.equipment.total_ventilators}"
            )

    # -----------------------------------------------------------------
    # Hour 6: Shift handover — handled by handover.py automatically.
    # We just log the trigger here.
    # -----------------------------------------------------------------
    elif crisis_type == "shift_handover":
        events.append("crisis:shift_handover:observation_degrading")

    # -----------------------------------------------------------------
    # Hour 7: BiPAP crash — force the BiPAP patient to crash
    # Find the first patient on BiPAP and force reintubation need.
    # -----------------------------------------------------------------
    elif crisis_type == "bipap_crash":
        bipap_patients = [
            p for p in ward.patients
            if p.support_level == SupportLevel.BIPAP
            and not p.is_discharged
        ]
        if bipap_patients:
            patient = bipap_patients[0]
            # Force deterioration — set vitals to critical levels
            patient.spo2 = 84.0
            patient.heart_rate = 138.0
            patient.resp_rate = 36.0
            patient.hemodynamically_stable = False
            patient.state = PatientState.INTUBATED_UNSTABLE
            events.append(
                f"crisis:bipap_crash:{patient.patient_id}:needs_reintubation"
            )

    # -----------------------------------------------------------------
    # Hour 9: VAP outbreak — two patients spike VAP signs
    # Find 2 high-risk ventilated patients and push their VAP risk high.
    # -----------------------------------------------------------------
    elif crisis_type == "vap_outbreak":
        at_risk = [
            p for p in ward.patients
            if p.support_level == SupportLevel.FULL_VENTILATOR
            and not p.is_discharged
            and not p.has_vap
            and p.hours_on_vent >= config.VAP_ELIGIBLE_AFTER_HOURS
        ]
        # Sort by existing VAP risk — highest risk patients show signs first
        at_risk.sort(key=lambda p: p.vap_risk, reverse=True)
        outbreak_count = 0
        for patient in at_risk[:2]:
            patient.vap_risk = min(1.0, patient.vap_risk + 0.25)
            events.append(
                f"crisis:vap_outbreak:{patient.patient_id}:"
                f"vap_risk_now={patient.vap_risk:.2f}"
            )
            outbreak_count += 1
        if outbreak_count == 0:
            events.append("crisis:vap_outbreak:no_eligible_patients")

    # -----------------------------------------------------------------
    # Hour 10: Ethical triage — force a triage decision
    # Find two ventilator-needing patients with different SOFA scores.
    # -----------------------------------------------------------------
    elif crisis_type == "ethical_triage":
        candidates = [
            p for p in ward.patients
            if p.state == PatientState.INTUBATED_UNSTABLE
            and not p.is_discharged
        ]
        if len(candidates) >= 2:
            candidates.sort(key=lambda p: p.sofa_score, reverse=True)
            pa = candidates[0]
            pb = candidates[1]
            # Ensure different SOFA scores for clear decision
            if pa.sofa_score == pb.sofa_score:
                pb.sofa_score = max(0, pb.sofa_score - 2)
            ward.triage.inject_forced_triage(
                pa.patient_id, pb.patient_id,
                pa.sofa_score, pb.sofa_score,
            )
            events.append(
                f"crisis:ethical_triage:{pa.patient_id}(SOFA={pa.sofa_score})"
                f"_vs_{pb.patient_id}(SOFA={pb.sofa_score})"
            )
        else:
            events.append("crisis:ethical_triage:insufficient_candidates")

    # -----------------------------------------------------------------
    # Hour 11: RT unavailable — handled by ward._get_available_rts()
    # We just log the trigger here.
    # -----------------------------------------------------------------
    elif crisis_type == "rt_unavailable":
        events.append(
            f"crisis:rt_unavailable:rts_now={ward._get_available_rts()}"
        )

    return events


# =============================================================================
# PROMPT FORMATTING
# =============================================================================

def format_prompt(observation: Observation, crisis_log: list = None) -> str:
    """
    Build the complete LLM prompt for one step of Task 3.
    Includes the full crisis schedule so the agent can plan ahead.
    """
    crisis_log = crisis_log or []

    resource_section = (
        f"WARD RESOURCES:\n"
        f"  Ventilators : {observation.available_ventilators} free\n"
        f"  BiPAP       : {observation.available_bipap} free\n"
        f"  HFNC        : {observation.available_hfnc} free\n"
        f"  RTs on duty : {observation.available_rts}\n"
    )

    if observation.incoming_patients:
        incoming_lines = [
            f"  {a.alert_id}: arrives in {a.eta_hours}h — "
            f"{a.support_needed.value} — severity {a.severity_estimate.value}"
            for a in observation.incoming_patients
        ]
        incoming_section = (
            "INCOMING PATIENTS:\n" + "\n".join(incoming_lines)
        )
    else:
        incoming_section = "INCOMING PATIENTS: None currently announced."

    patient_lines = []
    for p in observation.patients:
        rsbi_str = f"{p.rsbi:.1f}" if p.rsbi is not None else "N/A"
        pf_str   = f"{p.pf_ratio:.1f}" if p.pf_ratio is not None else "N/A"
        rass_str = f"{p.rass:.1f}" if p.rass is not None else "N/A"
        degrad   = " [HANDOVER-DEGRADED]" if p.observation_degraded else ""
        patient_lines.append(
            f"  {p.patient_id}{degrad}: {p.state.value} | "
            f"{p.support_level.value} | SOFA={p.sofa_score}\n"
            f"    RSBI={rsbi_str} PF={pf_str} RASS={rass_str} "
            f"FiO2={p.fio2:.2f} PEEP={p.peep:.1f} "
            f"Hemo={p.hemodynamically_stable}\n"
            f"    Hours-on-vent={p.hours_on_vent} "
            f"VAP-risk={p.vap_risk:.2f} "
            f"Reint-risk={p.reintubation_risk:.2f} "
            f"Severity={p.severity.value}"
        )
    patient_section = "PATIENTS:\n" + "\n".join(patient_lines)

    if observation.active_alarms:
        alarm_lines = [
            f"  {a.alarm_id}: {a.patient_id} {a.alarm_type.value} "
            f"consecutive={a.consecutive_same_type}"
            for a in observation.active_alarms
        ]
        alarm_section = (
            "ACTIVE ALARMS:\n" + "\n".join(alarm_lines)
        )
    else:
        alarm_section = "ACTIVE ALARMS: None."

    triage_section = ""
    if observation.triage_decision_required:
        t = observation.triage_decision_required
        triage_section = (
            f"\nETHICAL TRIAGE REQUIRED — case {t.case_id}:\n"
            f"  {t.patient_a_id}: SOFA={t.patient_a_sofa} "
            f"wait={t.patient_a_wait_hours}h\n"
            f"  {t.patient_b_id}: SOFA={t.patient_b_sofa} "
            f"wait={t.patient_b_wait_hours}h\n"
            f"  ONE vent available. Lower SOFA = better prognosis = priority.\n"
            f"  Use action_type='ethical_triage_select' with "
            f"ethical_triage_patient_id set to chosen patient.\n"
        )

    handover_note = ""
    if observation.handover_degraded:
        handover_note = (
            "\nWARNING: Shift handover in progress. "
            "Some patient observations are approximate or missing. "
            "Use clinical judgment — do not act aggressively on degraded data.\n"
        )

    crisis_history = ""
    if crisis_log:
        crisis_history = (
            "\nCRISIS EVENTS THIS EPISODE:\n"
            + "\n".join(f"  {e}" for e in crisis_log[-10:])
        )

    return f"""You are a Charge Respiratory Therapist managing a {config.TASK3_NUM_PATIENTS}-patient ICU ward.
Current shift hour: {observation.shift_hour} / {config.TASK3_MAX_HOURS}

KNOWN CRISIS SCHEDULE (plan ahead):
  Hour 3:  Mass casualty — 4 patients arriving in 2h needing ventilators
  Hour 5:  Ventilator malfunction — capacity drops by 1
  Hour 6:  Shift handover — some observations will be degraded
  Hour 7:  BiPAP patient may crash — watch for deterioration
  Hour 9:  VAP outbreak expected — bundle compliance critical now
  Hour 10: Ethical triage — prepare SOFA scores for triage decision
  Hour 11: 1 RT unavailable — prioritise staffing carefully

{resource_section}
{incoming_section}

{patient_section}

{alarm_section}
{triage_section}{handover_note}{crisis_history}

AVAILABLE ACTIONS:
  attempt_sbt, extubate, step_down_to_bipap, step_down_to_hfnc,
  escalate_to_full_vent, assign_rt_attention, enforce_vap_bundle,
  respond_to_alarm, suppress_alarm, hold_and_monitor, ethical_triage_select

ACCP CRITERIA: RSBI<{config.RSBI_THRESHOLD}, PF>{config.PF_RATIO_THRESHOLD}, \
RASS {config.RASS_MIN}-{config.RASS_MAX}, hemo-stable, \
FiO2<={config.FIO2_MAX_FOR_SBT}, PEEP<={config.PEEP_MAX_FOR_SBT}, \
vent>={config.HOURS_ON_VENT_BEFORE_SBT}h

Respond with JSON array — one action per patient. ALL {len(observation.patients)} patients required.
Return ONLY valid JSON. No explanation, no markdown.
"""