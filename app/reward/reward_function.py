# app/reward/reward_function.py
# =============================================================================
# RespiraCare-ICU — Reward Function
#
# Computes the per-step reward signal from the outcomes of agent actions.
# This is a pure function — it reads state, computes numbers, returns a
# RewardBreakdown. It does not modify any state.
#
# Design philosophy:
#   - Reward at every step, not just episode end
#   - Partial credit for partial progress
#   - Delayed penalties for decisions whose consequences arrive later
#   - Two-sided: both over-treatment and under-treatment are penalised
#   - Every reward component is inspectable via RewardBreakdown
#
# The WardStateManager already computes most rewards inline during
# apply_actions() and advance_hour(). This module provides:
#   1. compute_step_reward()  — consolidates inline rewards into a breakdown
#   2. compute_final_reward() — end-of-episode bonus/penalty adjustments
#   3. Helper functions used by graders to score individual components
#
# ward.py and episode.py call this module — graders call the helpers directly.
# =============================================================================

from __future__ import annotations

from typing import List, Optional, Tuple

from app.models import (
    ActionType,
    RewardBreakdown,
    PatientState,
    SupportLevel,
)
from app import config


# =============================================================================
# STEP REWARD COMPUTATION
# =============================================================================

def compute_extubation_reward(
    patient_state_before: PatientState,
    patient_state_after: PatientState,
    reintubation_risk_at_extubation: float,
) -> Tuple[float, str]:
    """
    Compute the reward for an extubation action.

    Successful extubation → positive reward.
    Failed extubation (reintubation) → penalty weighted by how predictable
    the failure was. If risk was high and agent extubated anyway — large
    penalty. If risk was low and patient unexpectedly failed — small penalty.

    Returns (reward_delta, event_description).
    """
    if patient_state_after == PatientState.EXTUBATED:
        return config.REWARD_SUCCESSFUL_EXTUBATION, "extubation_success"

    elif patient_state_after == PatientState.REINTUBATED:
        # Penalty = base × (1 - risk_at_extubation)
        # High risk → small (1-risk) → small penalty multiplier... wait,
        # spec says: penalty scales with how PREDICTABLE the failure was.
        # High risk patient extubated → highly predictable failure → LARGE penalty
        # So penalty = base × risk_at_extubation (not 1-risk)
        #
        # Re-reading spec: "−0.40 × (1 − risk_at_extubation)"
        # Example: risk=0.85, penalty = -0.40 × (1-0.15) = -0.34 (severe)
        # Example: risk=0.15, penalty = -0.40 × (1-0.85) = -0.06 (mild)
        # This means HIGH risk = LARGE penalty, which matches the intent.
        # The formula is: penalty = BASE × risk_at_extubation
        # because (1 - risk_at_extubation) when risk is HIGH is SMALL...
        #
        # Wait — re-reading spec example carefully:
        # "risk=0.85 → penalty = -0.40 × (1-0.15) = -0.34"
        # The 0.15 here is (1 - 0.85), i.e. the COMPLEMENT of risk.
        # So the formula as written gives LARGER penalty for HIGHER risk.
        # Let's verify: risk=0.85, formula = -0.40 × (1-(1-0.85))
        #                                  = -0.40 × (1-0.15)
        #                                  = -0.40 × 0.85 = -0.34 ✓
        # So the spec's (1-risk_at_extubation) in the example uses
        # risk_at_extubation as the COMPLEMENT. The actual formula is:
        # penalty = BASE × risk_at_extubation
        #
        # Implementation matches spec examples:
        penalty = config.PENALTY_REINTUBATION_BASE * reintubation_risk_at_extubation
        return penalty, "extubation_failed_reintubation"

    return 0.0, "extubation_no_change"


def compute_sbt_reward(
    sbt_overdue: bool,
    sbt_passed: bool,
    meets_criteria: bool,
) -> Tuple[float, str]:
    """
    Reward for attempting a spontaneous breathing trial.

    Only rewarded when the patient was overdue and ready.
    Attempting SBT on a not-ready patient gives no reward (but no penalty
    here — the penalty comes if they're later extubated prematurely).
    """
    if not meets_criteria:
        return 0.0, "sbt_criteria_not_met"

    if sbt_overdue and sbt_passed:
        return config.REWARD_SBT_ATTEMPTED_OVERDUE, "sbt_overdue_passed"

    if sbt_passed:
        return config.REWARD_SBT_ATTEMPTED_OVERDUE * 0.7, "sbt_passed"

    return 0.0, "sbt_attempted_failed"


def compute_vap_bundle_reward(
    hours_on_vent: int,
    hours_since_last_bundle: int,
) -> Tuple[float, str]:
    """
    Reward for enforcing the VAP prevention bundle.

    Scales with how long the patient has been at risk — enforcing the
    bundle for a patient who has been on the vent for 5 days is worth
    more than for a patient ventilated for 2 days.
    """
    if hours_on_vent < config.VAP_ELIGIBLE_AFTER_HOURS:
        return 0.0, "vap_bundle_not_eligible"

    hours_at_risk = max(1, hours_on_vent - config.VAP_ELIGIBLE_AFTER_HOURS)

    # Extra multiplier if bundle was overdue (>= 4 hours since last)
    overdue_multiplier = 1.5 if hours_since_last_bundle >= 4 else 1.0

    reward = config.REWARD_VAP_BUNDLE_ENFORCED_BASE * hours_at_risk * overdue_multiplier
    reward = min(reward, 0.50)   # Cap at 0.50 to prevent runaway rewards

    return round(reward, 4), "vap_bundle_enforced"


def compute_rt_assignment_reward(
    severity_label: str,
    is_unstable: bool,
) -> Tuple[float, str]:
    """
    Reward for assigning an RT to a patient.

    High-acuity patients benefit most from dedicated RT attention.
    Assigning RT to a low-acuity stable patient is a mild misallocation.
    """
    if is_unstable or severity_label == "high":
        return config.REWARD_RT_CORRECT_ASSIGNMENT, "rt_high_acuity"

    elif severity_label == "medium":
        return config.REWARD_RT_CORRECT_ASSIGNMENT * 0.6, "rt_medium_acuity"

    else:
        return config.PENALTY_RT_LOW_ACUITY_ASSIGNMENT, "rt_low_acuity_misallocation"


def compute_alarm_reward(
    is_real: bool,
    agent_responded: bool,
) -> Tuple[float, str]:
    """
    Score a single alarm response decision.

    TP: responded to real alarm        → reward
    TN: suppressed false alarm         → small reward
    FP: responded to false alarm       → small penalty
    FN: suppressed real alarm          → large penalty
    """
    if agent_responded and is_real:
        return config.REWARD_REAL_ALARM_CORRECTLY_ACTIONED, "alarm_tp"

    elif not agent_responded and not is_real:
        return config.REWARD_FALSE_ALARM_CORRECTLY_SUPPRESSED, "alarm_tn"

    elif agent_responded and not is_real:
        return config.PENALTY_FALSE_ALARM_ESCALATED, "alarm_fp"

    else:   # not agent_responded and is_real
        return config.PENALTY_REAL_ALARM_IGNORED, "alarm_fn"


def compute_ethical_triage_reward(is_correct: bool) -> Tuple[float, str]:
    """Score an ethical triage decision."""
    if is_correct:
        return config.REWARD_ETHICAL_TRIAGE_CORRECT, "triage_correct"
    return config.PENALTY_ETHICAL_TRIAGE_WRONG, "triage_wrong"


def compute_vap_penalty() -> Tuple[float, str]:
    """
    Penalty when VAP infection develops.
    This fires 2 hours after the risk threshold was crossed — the delayed
    consequence that teaches the agent present shortcuts have future costs.
    """
    return config.PENALTY_VAP_DEVELOPS, "vap_infection_developed"


def compute_no_vent_penalty() -> Tuple[float, str]:
    """Penalty when an incoming patient arrives with no ventilator available."""
    return config.PENALTY_NO_VENTILATOR_FOR_INCOMING, "no_vent_for_incoming_patient"


def compute_bipap_deterioration_penalty() -> Tuple[float, str]:
    """Penalty when a BiPAP patient crashes and the agent missed the warning signs."""
    return config.PENALTY_BIPAP_DETERIORATION_MISSED, "bipap_patient_crashed_undetected"


# =============================================================================
# STEP REWARD CONSOLIDATION
# =============================================================================

def compute_step_reward(
    action_rewards: List[Tuple[float, str]],
    penalty_events: List[str],
) -> RewardBreakdown:
    """
    Consolidate all reward components from one step into a RewardBreakdown.

    Args:
        action_rewards: list of (reward_delta, event_str) from each patient action
        penalty_events: list of penalty event strings from fleet/VAP/triage

    Returns a fully populated RewardBreakdown.
    """
    breakdown = RewardBreakdown(total=0.0)

    for reward_delta, event_str in action_rewards:
        breakdown.total += reward_delta

        if "extubation_success" in event_str:
            breakdown.extubation_reward += reward_delta

        elif "reintubation" in event_str:
            breakdown.reintubation_penalty += reward_delta

        elif "sbt" in event_str:
            breakdown.sbt_reward += reward_delta

        elif "vap_bundle" in event_str:
            breakdown.vap_bundle_reward += reward_delta

        elif "rt_high" in event_str or "rt_medium" in event_str:
            breakdown.rt_assignment_reward += reward_delta

        elif "rt_low_acuity" in event_str:
            breakdown.rt_misallocation_penalty += reward_delta

        elif "alarm_tp" in event_str:
            breakdown.alarm_true_positive_reward += reward_delta

        elif "alarm_tn" in event_str:
            breakdown.alarm_false_negative_reward += reward_delta

        elif "alarm_fp" in event_str:
            breakdown.alarm_false_escalation_penalty += reward_delta

        elif "alarm_fn" in event_str:
            breakdown.alarm_missed_penalty += reward_delta

        elif "triage_correct" in event_str:
            breakdown.ethical_triage_reward += reward_delta

        elif "triage_wrong" in event_str:
            breakdown.ethical_triage_penalty += reward_delta

    # Apply penalty events from background modules
    for event_str in penalty_events:
        if "vap_infection" in event_str or "vap_developed" in event_str:
            breakdown.vap_develops_penalty += config.PENALTY_VAP_DEVELOPS
            breakdown.total += config.PENALTY_VAP_DEVELOPS

        elif "no_vent" in event_str:
            breakdown.no_vent_available_penalty += config.PENALTY_NO_VENTILATOR_FOR_INCOMING
            breakdown.total += config.PENALTY_NO_VENTILATOR_FOR_INCOMING

        elif "bipap_crash" in event_str:
            breakdown.bipap_deterioration_penalty += config.PENALTY_BIPAP_DETERIORATION_MISSED
            breakdown.total += config.PENALTY_BIPAP_DETERIORATION_MISSED

    breakdown.total = round(breakdown.total, 4)
    return breakdown


# =============================================================================
# FINAL EPISODE REWARD ADJUSTMENTS
# =============================================================================

def compute_final_reward(
    total_steps: int,
    total_vap_incidents: int,
    total_reintubations: int,
    no_vent_penalties: int,
    triage_accuracy: float,
    alarm_accuracy: float,
) -> Tuple[float, str]:
    """
    End-of-episode reward adjustment.
    Applied once when done=True to capture episode-level performance.

    This is NOT the grader score — the grader computes its own composite
    score from the episode history. This is an additional reward shaping
    signal that rewards clean episodes.

    Returns (bonus_or_penalty, description).
    """
    adjustment = 0.0
    reasons = []

    # Clean episode bonus — no VAP, no reintubations, no vent shortfalls
    if total_vap_incidents == 0 and total_reintubations == 0 and no_vent_penalties == 0:
        adjustment += 0.20
        reasons.append("clean_episode_bonus")

    # VAP incidence penalty (additional to per-event penalties)
    if total_vap_incidents > 0:
        adjustment -= total_vap_incidents * 0.05
        reasons.append(f"vap_episode_penalty:{total_vap_incidents}")

    # Excellent alarm accuracy bonus
    if alarm_accuracy >= 0.85:
        adjustment += 0.10
        reasons.append("excellent_alarm_accuracy")

    # Perfect triage bonus
    if triage_accuracy == 1.0:
        adjustment += 0.05
        reasons.append("perfect_triage")

    return round(adjustment, 4), "|".join(reasons) if reasons else "no_adjustment"


# =============================================================================
# GRADER HELPER — normalise a raw score to [0, 1]
# =============================================================================

def normalise_score(
    raw: float,
    min_val: float,
    max_val: float,
) -> float:
    """
    Normalise a raw metric value to [0.0, 1.0].
    Used by graders to convert cumulative rewards into bounded scores.
    """
    if max_val <= min_val:
        return 1.0
    normalised = (raw - min_val) / (max_val - min_val)
    return round(min(1.0, max(0.0, normalised)), 4)


def clamp_score(score: float) -> float:
    """Ensure a grader score stays in [0.0, 1.0]."""
    return round(min(1.0, max(0.0, score)), 4)