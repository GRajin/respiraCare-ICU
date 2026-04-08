# app/graders/crisis_grader.py
# =============================================================================
# RespiraCare-ICU — Task 3 Grader: Full Shift Crisis Management
#
# Composite grader with 5 weighted signals:
#
#   1. mortality_proxy       (30%) — patient outcome scores based on
#                                    final states and SOFA trajectories
#   2. equipment_utilization (20%) — was equipment allocated to highest-need?
#   3. vap_incidence         (20%) — bundle compliance + VAP incidents
#   4. reintubation_quality  (15%) — quality of extubation decisions
#   5. crisis_response_speed (15%) — time to respond to injected crises
#
# Score = weighted sum clamped to [0.0, 1.0]
# Expected range for null agent: 0.10 – 0.30
# =============================================================================

from __future__ import annotations

from typing import Dict, List

from app.environment.episode import Episode
from app.reward.reward_function import clamp_score
from app import config


def grade(episode: Episode) -> Dict:
    """
    Grade a completed Task 3 episode.

    Returns a dict with:
      score            — final weighted score in [0.0, 1.0]
      component_scores — per-signal breakdown
      details          — human-readable summary
    """
    if not episode.history:
        return {
            "score": 0.0,
            "component_scores": {},
            "details": "No steps recorded.",
        }

    # =========================================================================
    # SIGNAL 1 — Mortality proxy (0.30)
    # Based on final patient states and SOFA score trajectories.
    # =========================================================================
    mortality_score = _score_mortality_proxy(episode)

    # =========================================================================
    # SIGNAL 2 — Equipment utilisation (0.20)
    # Were vents allocated to highest-SOFA patients?
    # Were vents freed before surge arrivals?
    # =========================================================================
    equipment_score = _score_equipment_utilization(episode)

    # =========================================================================
    # SIGNAL 3 — VAP incidence (0.20)
    # Bundle compliance rate penalised by VAP incidents.
    # =========================================================================
    vap_score = _score_vap_incidence(episode)

    # =========================================================================
    # SIGNAL 4 — Reintubation quality (0.15)
    # Penalises predictable reintubations more than unpredictable ones.
    # =========================================================================
    reintubation_score = _score_reintubation_quality(episode)

    # =========================================================================
    # SIGNAL 5 — Crisis response speed (0.15)
    # Did the agent respond to injected crises promptly?
    # =========================================================================
    crisis_score = _score_crisis_response(episode)

    # =========================================================================
    # WEIGHTED COMPOSITE
    # =========================================================================
    w1 = config.GRADER_T3_MORTALITY_PROXY        # 0.30
    w2 = config.GRADER_T3_EQUIPMENT_UTILIZATION  # 0.20
    w3 = config.GRADER_T3_VAP_INCIDENCE          # 0.20
    w4 = config.GRADER_T3_REINTUBATION_QUALITY   # 0.15
    w5 = config.GRADER_T3_CRISIS_RESPONSE_SPEED  # 0.15

    composite = (
        w1 * mortality_score +
        w2 * equipment_score +
        w3 * vap_score +
        w4 * reintubation_score +
        w5 * crisis_score
    )
    final_score = clamp_score(composite)

    component_scores = {
        "mortality_proxy":       round(mortality_score, 4),
        "equipment_utilization": round(equipment_score, 4),
        "vap_incidence":         round(vap_score, 4),
        "reintubation_quality":  round(reintubation_score, 4),
        "crisis_response_speed": round(crisis_score, 4),
    }

    details = (
        f"Task 3 score: {final_score:.3f} | "
        f"Mortality={mortality_score:.3f}(x{w1}) "
        f"Equipment={equipment_score:.3f}(x{w2}) "
        f"VAP={vap_score:.3f}(x{w3}) "
        f"Reint={reintubation_score:.3f}(x{w4}) "
        f"Crisis={crisis_score:.3f}(x{w5})"
    )

    return {
        "score": final_score,
        "component_scores": component_scores,
        "details": details,
    }


# =============================================================================
# SIGNAL HELPERS
# =============================================================================

def _score_mortality_proxy(episode: Episode) -> float:
    """
    Score based on patient outcome states at episode end.

    Outcome scoring per patient:
      DISCHARGED              → 1.0  (best outcome)
      EXTUBATED (room air)    → 0.9
      EXTUBATED (HFNC)        → 0.75
      EXTUBATED (BiPAP)       → 0.65
      INTUBATED_STABLE        → 0.50
      INTUBATED_UNSTABLE      → 0.25
      INTUBATED_WITH_VAP      → 0.15
      REINTUBATED             → 0.10

    Also penalises high final SOFA scores (SOFA > threshold = bad prognosis).
    Varies by config because configs start with different severity distributions.
    """
    outcome_scores = {
        "DISCHARGED":         1.0,
        "EXTUBATED":          0.80,
        "INTUBATED_STABLE":   0.50,
        "INTUBATED_UNSTABLE": 0.25,
        "INTUBATED_WITH_VAP": 0.15,
        "REINTUBATED":        0.10,
        "SBT_IN_PROGRESS":    0.55,
        "READY_TO_EXTUBATE":  0.60,
    }

    patients = episode.ward.patients
    if not patients:
        return 0.0

    total = 0.0
    for patient in patients:
        base = outcome_scores.get(patient.state.value, 0.30)

        # SOFA penalty — high SOFA at end means patient deteriorated
        sofa_penalty = 0.0
        if patient.sofa_score > config.SOFA_HIGH_THRESHOLD:
            sofa_penalty = 0.15
        elif patient.sofa_score > config.SOFA_LOW_THRESHOLD:
            sofa_penalty = 0.07

        # Support level adjustment for EXTUBATED patients
        if patient.state.value == "EXTUBATED":
            if patient.support_level.value == "ROOM_AIR":
                base = 0.90
            elif patient.support_level.value == "HFNC":
                base = 0.75
            elif patient.support_level.value == "BIPAP":
                base = 0.65

        total += max(0.0, base - sofa_penalty)

    raw_score = total / len(patients)

    # Compare against initial state — improvement is rewarded
    initial_obs = episode.initial_observation
    if initial_obs:
        initial_avg_sofa = sum(
            p.sofa_score for p in initial_obs.patients
        ) / len(initial_obs.patients)
        final_avg_sofa = sum(
            p.sofa_score for p in patients
        ) / len(patients)

        # Bonus if average SOFA improved
        if final_avg_sofa < initial_avg_sofa:
            raw_score = min(1.0, raw_score + 0.05)

    return clamp_score(raw_score)


def _score_equipment_utilization(episode: Episode) -> float:
    """
    Score based on equipment allocation quality and surge preparation.

    Three components:
      A. Shortfall penalty — incoming patients with no vent
      B. Utilisation quality — vents held by appropriate patients
      C. Starting headroom penalty — ward that starts at 0 free vents
         and never frees any equipment scores lower
    """
    fleet_summary = episode.ward.fleet.get_summary()
    no_vent_count = episode.ward.fleet.equipment.no_vent_penalty_count
    total_incoming = fleet_summary["incoming_total"]

    # Component A: shortfall penalty
    if total_incoming > 0:
        shortfall_rate = no_vent_count / max(1, total_incoming)
        shortfall_score = clamp_score(1.0 - shortfall_rate * 1.5)
    else:
        shortfall_score = 0.70   # No incoming — neutral, not perfect

    # Component B: utilisation quality
    final_patients = episode.ward.patients
    total_vented = sum(
        1 for p in final_patients
        if p.support_level.value == "FULL_VENTILATOR" and not p.is_discharged
    )
    high_acuity_vented = sum(
        1 for p in final_patients
        if p.support_level.value == "FULL_VENTILATOR"
        and not p.is_discharged
        and p.sofa_score >= config.SOFA_LOW_THRESHOLD
    )
    utilisation_ratio = (
        high_acuity_vented / total_vented if total_vented > 0 else 1.0
    )

    # Component C: starting headroom — penalise if ward started full
    # and no equipment was freed during the episode
    initial_obs = episode.initial_observation
    initial_vents_free = 0
    if initial_obs:
        initial_vents_free = initial_obs.available_ventilators

    n_extubated = sum(
        1 for record in episode.history
        for event in record.events
        if "extubation_successful" in event
    )
    n_stepdowns = sum(
        1 for record in episode.history
        for event in record.events
        if "stepped_down" in event
    )
    equipment_freed = n_extubated + n_stepdowns

    # If ward started with 0 free vents AND agent freed nothing → heavy penalty
    if initial_vents_free == 0 and equipment_freed == 0:
        headroom_penalty = 0.35
    elif initial_vents_free == 0 and equipment_freed < 3:
        headroom_penalty = 0.15
    else:
        headroom_penalty = 0.0

    combined = (
        shortfall_score * 0.50
        + utilisation_ratio * 0.30
        - headroom_penalty
    )
    return clamp_score(combined)


def _score_vap_incidence(episode: Episode) -> float:
    """
    Score based on VAP prevention quality.

    Combines compliance rate with incident count.
    Heavily penalises each VAP incident (preventable harm).
    Varies by config due to different initial VAP risks.
    """
    vap_stats = episode.ward.vap.get_compliance_stats()
    compliance_rate = vap_stats["overall_compliance_rate"]
    n_incidents = vap_stats["total_vap_incidents"]

    # VAP incidents are heavily penalised in Task 3
    incident_penalty = n_incidents * 0.20

    # Also penalise VAP risk growth (from crisis_grader perspective)
    initial_obs = episode.initial_observation
    if initial_obs:
        initial_vap_avg = sum(
            p.vap_risk for p in initial_obs.patients
        ) / len(initial_obs.patients)

        final_vap_avg = sum(
            p.vap_risk for p in episode.ward.patients
        ) / len(episode.ward.patients)

        growth_penalty = max(0.0, (final_vap_avg - initial_vap_avg) * 2.0)
    else:
        growth_penalty = 0.0

    score = clamp_score(compliance_rate - incident_penalty - growth_penalty)
    return score


def _score_reintubation_quality(episode: Episode) -> float:
    """
    Score based on the quality of extubation decisions.

    Two components:
      A. Reintubation penalty — predictable failures penalised more
      B. Missed opportunity penalty — READY_TO_EXTUBATE patients
         that were never extubated during the episode

    This ensures the null agent (never extubates) is penalised for
    ignoring clearly ready patients, not rewarded with a perfect score.
    """
    # Component A: reintubation events
    reintubation_count = 0
    for record in episode.history:
        for event in record.events:
            if "reintubated" in event or "extubation_failed" in event:
                reintubation_count += 1

    n_patients = config.TASK3_NUM_PATIENTS
    reintubation_rate = reintubation_count / n_patients

    # Component B: missed extubation opportunities
    # Count patients who were READY_TO_EXTUBATE at episode start
    # but were never extubated during the episode
    initial_obs = episode.initial_observation
    n_ready_at_start = 0
    if initial_obs:
        n_ready_at_start = sum(
            1 for p in initial_obs.patients
            if p.state.value == "READY_TO_EXTUBATE"
        )

    # Count successful extubations during episode
    n_extubated = sum(
        1 for record in episode.history
        for event in record.events
        if "extubation_successful" in event
    )

    # Also count patients who ended up extubated or discharged
    n_good_outcomes = sum(
        1 for p in episode.ward.patients
        if p.state.value in ("EXTUBATED", "DISCHARGED")
    )

    # Missed = ready patients never acted on
    n_missed = max(0, n_ready_at_start - n_extubated)
    missed_rate = n_missed / max(1, n_patients)

    # Opportunity score: starts at 1.0, reduced by missed opportunities
    # Each missed ready patient = -0.12 penalty
    opportunity_penalty = missed_rate * 1.2

    # Good outcome bonus: patients who ended extubated/discharged
    outcome_bonus = min(0.20, (n_good_outcomes / n_patients) * 0.30)

    score = clamp_score(
        1.0
        - (reintubation_rate / 0.25)   # Reintubation penalty
        - opportunity_penalty           # Missed opportunities
        + outcome_bonus                 # Good outcome bonus
    )
    return score


def _score_crisis_response(episode: Episode) -> float:
    """
    Score based on how quickly and correctly the agent responded to
    injected crisis events.

    Checks the event log for crisis events and measures response lag:
      - BiPAP crash: how many hours before escalate_to_full_vent?
      - Ethical triage: did the agent respond within 1 hour?
      - VAP outbreak: did the agent enforce bundles in the next 2 hours?
      - Mass casualty: were vents freed before arrival?

    Varies by config because different configs start with different
    equipment headroom, making surge preparation easier or harder.
    """
    total_crises = len(config.TASK3_CRISIS_SCHEDULE)
    responded_count = 0

    event_log = episode.ward.event_log

    # Check 1: BiPAP crash response (crisis at hour 7)
    bipap_crisis_hour = 7
    bipap_responded = False
    for record in episode.history:
        if record.hour_after >= bipap_crisis_hour:
            for event in record.events:
                if "escalated_to_full_vent" in event or "escalate" in event:
                    bipap_responded = True
                    break
    if bipap_responded:
        responded_count += 1

    # Check 2: Ethical triage response (crisis at hour 10)
    triage_stats = episode.ward.triage.get_triage_stats()
    if triage_stats["total_cases"] > 0:
        if triage_stats["correct_decisions"] > 0:
            responded_count += 2   # Correct triage = double credit
        elif triage_stats["total_cases"] - triage_stats["unanswered"] > 0:
            responded_count += 1   # Responded but wrong

    # Check 3: VAP bundle response after outbreak (crisis at hour 9)
    vap_stats = episode.ward.vap.get_compliance_stats()
    post_outbreak_compliance = 0
    eligible_post = 0
    for record in episode.history:
        if record.hour_after >= 9:
            for event in record.events:
                if "vap_bundle_enforced" in event:
                    post_outbreak_compliance += 1
            eligible_post += 1

    if eligible_post > 0 and post_outbreak_compliance > 0:
        responded_count += 1

    # Check 4: Mass casualty preparation (crisis at hour 3 — vents freed by hour 5)
    fleet_summary = episode.ward.fleet.get_summary()
    if fleet_summary["no_vent_penalty_count"] == 0:
        responded_count += 1

    # Normalise: max possible responses = total_crises + 1 (triage bonus)
    max_responses = total_crises + 1
    crisis_score = clamp_score(responded_count / max_responses)
    return crisis_score


# =============================================================================
# DISTRIBUTION CHECK
# =============================================================================

def check_score_distribution(n_seeds: int = 10) -> Dict:
    """Run the null agent across n_seeds and return score statistics."""
    from app.environment.episode import Episode
    from app.models import Action, ActionType
    from app.tasks.task3_crisis import apply_crisis

    scores = []
    for seed in range(n_seeds):
        ep = Episode()
        ep.reset(task_id=3, seed=seed)
        for _ in range(config.TASK3_MAX_HOURS):
            if ep.is_done:
                break
            obs = ep.ward.get_observation(ep.ward.current_hour)
            acts = [
                Action(
                    patient_id=p.patient_id,
                    action_type=ActionType.HOLD_AND_MONITOR,
                )
                for p in obs.patients
            ]
            ep.step(acts)
            # Apply crisis events
            crisis_events = apply_crisis(ep.ward, ep.ward.current_hour)
            ep.ward.event_log.extend(crisis_events)

        result = grade(ep)
        scores.append(result["score"])

    return {
        "scores": scores,
        "min": min(scores),
        "max": max(scores),
        "mean": round(sum(scores) / len(scores), 4),
        "is_constant": len(set(scores)) == 1,
    }