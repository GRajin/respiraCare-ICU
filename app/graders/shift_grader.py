# app/graders/shift_grader.py
# =============================================================================
# RespiraCare-ICU — Task 2 Grader: Shift Resource Optimization
#
# Composite grader with 4 weighted metrics.
# Each metric is designed to produce different scores across different
# ward configurations — even for a null agent.
#
# Metric design principle: scores must vary based on INITIAL WARD STATE
# not just agent behaviour. This ensures Phase 1 gate compliance
# (graders must not be constant across seeds).
#
#   1. missed_opportunities  (35%) — READY_TO_EXTUBATE patients not extubated
#                                    before trauma arrival + vent shortfalls
#   2. reintubation_rate     (25%) — penalised by predictability
#   3. vap_risk_growth       (20%) — how much VAP risk grew during episode
#   4. alarm_response        (20%) — real alarm volume missed × FN penalty
# =============================================================================

from __future__ import annotations

from typing import Dict, List

from app.environment.episode import Episode
from app.reward.reward_function import clamp_score
from app import config


def grade(episode: Episode) -> Dict:
    """
    Grade a completed Task 2 episode.

    Returns a dict with:
      score              — final weighted score in [0.0, 1.0]
      component_scores   — per-metric breakdown
      details            — human-readable summary
    """
    if not episode.history:
        return {
            "score": 0.0,
            "component_scores": {},
            "details": "No steps recorded.",
        }

    # =========================================================================
    # METRIC 1 — Missed extubation opportunities + equipment shortfall (0.35)
    #
    # Measures: did the agent free equipment proactively?
    # Varies by config because each config has different numbers of
    # READY_TO_EXTUBATE patients at the start.
    # =========================================================================
    opportunity_score = _score_missed_opportunities(episode)

    # =========================================================================
    # METRIC 2 — Reintubation rate (0.25)
    #
    # Counts reintubation events. Varies if agent attempts extubations.
    # Null agent gets 1.0 (no extubations = no reintubations).
    # Smart agent may score lower if it extubates prematurely.
    # =========================================================================
    reintubation_score = _score_reintubation_rate(episode)

    # =========================================================================
    # METRIC 3 — VAP risk growth during episode (0.20)
    #
    # Measures total VAP risk growth across all patients.
    # Varies by config because configs have different starting VAP risks
    # and different numbers of unstable/high-risk patients.
    # More growth = worse score. Naturally varies without agent intervention.
    # =========================================================================
    vap_growth_score = _score_vap_risk_growth(episode)

    # =========================================================================
    # METRIC 4 — Real alarm volume response (0.20)
    #
    # Measures missed real alarms as a fraction of total real alarms.
    # Varies by config because configs with more INTUBATED_UNSTABLE patients
    # generate more real alarms. Missing more real alarms = lower score.
    # =========================================================================
    alarm_score = _score_alarm_response(episode)

    # =========================================================================
    # WEIGHTED COMPOSITE
    # =========================================================================
    w1 = config.GRADER_T2_EQUIPMENT_AVAILABILITY   # 0.35
    w2 = config.GRADER_T2_REINTUBATION_RATE        # 0.25
    w3 = config.GRADER_T2_VAP_COMPLIANCE           # 0.20
    w4 = config.GRADER_T2_ALARM_ACCURACY           # 0.20

    composite = (
        w1 * opportunity_score +
        w2 * reintubation_score +
        w3 * vap_growth_score +
        w4 * alarm_score
    )
    final_score = clamp_score(composite)

    component_scores = {
        "missed_opportunities": round(opportunity_score, 4),
        "reintubation_rate":    round(reintubation_score, 4),
        "vap_risk_growth":      round(vap_growth_score, 4),
        "alarm_response":       round(alarm_score, 4),
    }

    details = (
        f"Task 2 score: {final_score:.3f} | "
        f"Opportunities={opportunity_score:.3f}(x{w1}) "
        f"Reintubation={reintubation_score:.3f}(x{w2}) "
        f"VAP_growth={vap_growth_score:.3f}(x{w3}) "
        f"Alarm={alarm_score:.3f}(x{w4})"
    )

    return {
        "score": final_score,
        "component_scores": component_scores,
        "details": details,
    }


# =============================================================================
# METRIC HELPERS
# =============================================================================

def _score_missed_opportunities(episode: Episode) -> float:
    """
    Score based on missed extubation opportunities before trauma arrival
    combined with equipment shortfalls when trauma arrived.

    Two components:
      A. Equipment shortfall (hard penalty): no vent for incoming patient
      B. Missed opportunities: READY_TO_EXTUBATE patients at start that
         were not extubated before hour 3

    Varies by config because each config has 0-3 ready patients.
    Config with 3 ready = max opportunity = max penalty if not acted on.
    Config with 0 ready = no opportunity = no opportunity penalty.
    """
    # Component A: hard penalty for equipment shortfall
    no_vent_count = episode.ward.fleet.equipment.no_vent_penalty_count
    fleet_summary = episode.ward.fleet.get_summary()
    total_incoming = fleet_summary["incoming_total"]

    if total_incoming == 0:
        shortfall_score = 0.85   # No incoming — neutral
    elif no_vent_count == 0:
        shortfall_score = 1.0
    elif no_vent_count == 1:
        shortfall_score = 0.40
    elif no_vent_count == 2:
        shortfall_score = 0.15
    else:
        shortfall_score = 0.0

    # Component B: missed extubation opportunities
    # Count READY_TO_EXTUBATE patients in the initial observation
    initial_obs = episode.initial_observation
    if initial_obs is None:
        return shortfall_score

    n_ready_at_start = sum(
        1 for p in initial_obs.patients
        if p.state.value == "READY_TO_EXTUBATE"
    )

    # Count how many extubations actually happened in episode
    n_extubated = sum(
        1 for record in episode.history
        for event in record.events
        if "extubation_successful" in event or "extubate_success" in event
    )

    # Missed = ready patients that weren't extubated
    n_missed = max(0, n_ready_at_start - n_extubated)

    # Each missed opportunity reduces the opportunity score
    # 1 missed = -0.15, 2 missed = -0.30, 3 missed = -0.45
    opportunity_penalty = n_missed * 0.15
    opportunity_score = clamp_score(shortfall_score - opportunity_penalty)

    return opportunity_score


def _score_reintubation_rate(episode: Episode) -> float:
    """
    Score based on reintubation events.
    Null agent always gets 1.0 (no extubations = no reintubations).
    Smart agents that extubate prematurely get penalised.
    """
    reintubation_count = sum(
        1 for record in episode.history
        for event in record.events
        if "reintubated" in event or "extubation_failed" in event
    )

    n_patients = config.TASK2_NUM_PATIENTS
    reintubation_rate = reintubation_count / n_patients
    score = clamp_score(1.0 - (reintubation_rate / 0.30))
    return score


def _score_vap_risk_growth(episode: Episode) -> float:
    """
    Score based on how much VAP risk grew during the episode.

    Compares initial VAP risks (from initial_observation) against
    final VAP risks (from current patient states).

    Varies by config: configs with higher starting VAP risks and more
    high-risk patients grow more → lower score for null agent on those configs.

    Also penalises any VAP incidents that occurred.
    """
    initial_obs = episode.initial_observation
    if initial_obs is None:
        vap_stats = episode.ward.vap.get_compliance_stats()
        return clamp_score(vap_stats["compliance_score"])

    # Build initial VAP risk map
    initial_vap = {p.patient_id: p.vap_risk for p in initial_obs.patients}

    # Compute total risk growth
    total_growth = 0.0
    n_ventilated = 0
    for patient in episode.ward.patients:
        if patient.support_level.value == "FULL_VENTILATOR":
            n_ventilated += 1
            initial_risk = initial_vap.get(patient.patient_id, 0.0)
            growth = max(0.0, patient.vap_risk - initial_risk)
            total_growth += growth

    # Normalise growth: max possible growth = n_ventilated * 0.03 * max_hours
    max_possible_growth = n_ventilated * config.VAP_RISK_PER_MISSED_HOUR * config.TASK2_MAX_HOURS
    if max_possible_growth <= 0:
        normalised_growth = 0.0
    else:
        normalised_growth = min(1.0, total_growth / max_possible_growth)

    # Also penalise VAP incidents
    vap_stats = episode.ward.vap.get_compliance_stats()
    vap_incident_penalty = vap_stats["total_vap_incidents"] * 0.20

    score = clamp_score(1.0 - normalised_growth - vap_incident_penalty)
    return score


def _score_alarm_response(episode: Episode) -> float:
    """
    Score based on how many real alarms the agent responded to,
    weighted by the volume of real alarms present.

    Varies by config: configs with more INTUBATED_UNSTABLE patients
    generate more real alarms. Missing more real alarms = lower score.

    For null agent:
      - Configs with 3 unstable patients → many real alarms missed → low score
      - Configs with 1 unstable patient  → fewer real alarms → higher score
    """
    alarm_stats = episode.ward.alarms.get_alarm_accuracy_stats()

    total_real = alarm_stats["total_real"]
    true_positives = alarm_stats["true_positives"]
    false_negatives = alarm_stats["false_negatives"]   # missed real alarms
    false_positives = alarm_stats["false_positives"]

    if total_real == 0:
        return 1.0   # No real alarms — nothing to miss

    # Sensitivity: fraction of real alarms caught
    sensitivity = true_positives / total_real

    # Volume-weighted FN penalty: missing 10% of real alarms is worse
    # than missing 10% when there are twice as many real alarms
    fn_rate = false_negatives / max(1, total_real + alarm_stats["true_negatives"])
    fp_rate = false_positives / max(1, total_real + alarm_stats["true_negatives"])

    # Score: sensitivity weighted 0.60, FN penalty 0.30, FP cost 0.10
    score = (
        sensitivity * 0.60
        - fn_rate * 0.30
        - fp_rate * 0.10
    )
    return clamp_score(score)


# =============================================================================
# DISTRIBUTION CHECK — used by verify script
# =============================================================================

def check_score_distribution(n_seeds: int = 10) -> Dict:
    """Run the null agent across n_seeds and return score statistics."""
    from app.environment.episode import Episode
    from app.models import Action, ActionType

    scores = []
    for seed in range(n_seeds):
        ep = Episode()
        ep.reset(task_id=2, seed=seed)
        for _ in range(config.TASK2_MAX_HOURS):
            if ep.is_done:
                break
            acts = [
                Action(
                    patient_id=p.patient_id,
                    action_type=ActionType.HOLD_AND_MONITOR,
                )
                for p in ep.ward.get_observation(ep.ward.current_hour).patients
            ]
            ep.step(acts)
        result = grade(ep)
        scores.append(result["score"])

    return {
        "scores": scores,
        "min": min(scores),
        "max": max(scores),
        "mean": round(sum(scores) / len(scores), 4),
        "is_constant": len(set(scores)) == 1,
    }