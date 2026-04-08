# tests/test_graders.py
import pytest
from app.environment.episode import Episode
from app.models import Action, ActionType
from app.tasks.task1_readiness import evaluate_patient_readiness, get_correct_action
from app.graders.readiness_grader import grade as grade1
from app.graders.shift_grader import grade as grade2
from app.graders.crisis_grader import grade as grade3
from app.tasks.task3_crisis import apply_crisis
from app import config


# =============================================================================
# TASK 1 GRADER
# =============================================================================

def test_task1_perfect_agent_scores_1():
    ep = Episode()
    reset = ep.reset(task_id=1, seed=42)
    actions = [
        Action(
            patient_id=p.patient_id,
            action_type=ActionType(
                get_correct_action(evaluate_patient_readiness(p))
            )
        )
        for p in reset.observation.patients
    ]
    ep.step(actions)
    result = grade1(ep)
    assert result["score"] == 1.0


def test_task1_score_in_range():
    ep = Episode()
    reset = ep.reset(task_id=1, seed=42)
    ep.step([
        Action(patient_id=p.patient_id, action_type=ActionType.HOLD_AND_MONITOR)
        for p in reset.observation.patients
    ])
    result = grade1(ep)
    assert 0.0 <= result["score"] <= 1.0


def test_task1_scores_vary_across_seeds():
    scores = []
    for seed in range(6):
        ep = Episode()
        r = ep.reset(task_id=1, seed=seed)
        ep.step([
            Action(patient_id=p.patient_id, action_type=ActionType.HOLD_AND_MONITOR)
            for p in r.observation.patients
        ])
        scores.append(grade1(ep)["score"])
    assert len(set(scores)) > 1, "Task 1 grader must not be constant"


def test_task1_no_episode_returns_zero():
    ep = Episode()
    ep.reset(task_id=1, seed=42)
    # No step taken
    result = grade1(ep)
    assert result["score"] == 0.0


# =============================================================================
# TASK 2 GRADER
# =============================================================================

def test_task2_score_in_range():
    ep = Episode()
    ep.reset(task_id=2, seed=42)
    for _ in range(config.TASK2_MAX_HOURS):
        if ep.is_done:
            break
        ep.step([
            Action(patient_id=p.patient_id, action_type=ActionType.HOLD_AND_MONITOR)
            for p in ep.ward.get_observation(ep.ward.current_hour).patients
        ])
    result = grade2(ep)
    assert 0.0 <= result["score"] <= 1.0


def test_task2_has_all_components():
    ep = Episode()
    ep.reset(task_id=2, seed=42)
    for _ in range(config.TASK2_MAX_HOURS):
        if ep.is_done:
            break
        ep.step([
            Action(patient_id=p.patient_id, action_type=ActionType.HOLD_AND_MONITOR)
            for p in ep.ward.get_observation(ep.ward.current_hour).patients
        ])
    result = grade2(ep)
    components = result["component_scores"]
    assert "missed_opportunities" in components
    assert "reintubation_rate" in components
    assert "vap_risk_growth" in components
    assert "alarm_response" in components


def test_task2_scores_vary_across_seeds():
    scores = []
    for seed in range(6):
        ep = Episode()
        ep.reset(task_id=2, seed=seed)
        for _ in range(config.TASK2_MAX_HOURS):
            if ep.is_done:
                break
            ep.step([
                Action(patient_id=p.patient_id, action_type=ActionType.HOLD_AND_MONITOR)
                for p in ep.ward.get_observation(ep.ward.current_hour).patients
            ])
        scores.append(grade2(ep)["score"])
    assert len(set(scores)) > 1, "Task 2 grader must not be constant"


# =============================================================================
# TASK 3 GRADER
# =============================================================================

def test_task3_score_in_range():
    ep = Episode()
    ep.reset(task_id=3, seed=42)
    for _ in range(config.TASK3_MAX_HOURS):
        if ep.is_done:
            break
        obs = ep.ward.get_observation(ep.ward.current_hour)
        ep.step([
            Action(patient_id=p.patient_id, action_type=ActionType.HOLD_AND_MONITOR)
            for p in obs.patients
        ])
        crisis_events = apply_crisis(ep.ward, ep.ward.current_hour)
        ep.ward.event_log.extend(crisis_events)
    result = grade3(ep)
    assert 0.0 <= result["score"] <= 1.0


def test_task3_has_all_components():
    ep = Episode()
    ep.reset(task_id=3, seed=42)
    for _ in range(config.TASK3_MAX_HOURS):
        if ep.is_done:
            break
        obs = ep.ward.get_observation(ep.ward.current_hour)
        ep.step([
            Action(patient_id=p.patient_id, action_type=ActionType.HOLD_AND_MONITOR)
            for p in obs.patients
        ])
        ep.ward.event_log.extend(apply_crisis(ep.ward, ep.ward.current_hour))
    result = grade3(ep)
    components = result["component_scores"]
    assert "mortality_proxy" in components
    assert "equipment_utilization" in components
    assert "vap_incidence" in components
    assert "reintubation_quality" in components
    assert "crisis_response_speed" in components


def test_task3_null_agent_below_threshold():
    ep = Episode()
    ep.reset(task_id=3, seed=42)
    for _ in range(config.TASK3_MAX_HOURS):
        if ep.is_done:
            break
        obs = ep.ward.get_observation(ep.ward.current_hour)
        ep.step([
            Action(patient_id=p.patient_id, action_type=ActionType.HOLD_AND_MONITOR)
            for p in obs.patients
        ])
        ep.ward.event_log.extend(apply_crisis(ep.ward, ep.ward.current_hour))
    result = grade3(ep)
    assert result["score"] <= 0.50, \
        f"Task 3 null agent scored too high: {result['score']}"