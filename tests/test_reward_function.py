# tests/test_reward_function.py
import pytest
from app.models import PatientState
from app.reward.reward_function import (
    compute_extubation_reward,
    compute_vap_bundle_reward,
    compute_rt_assignment_reward,
    compute_alarm_reward,
    compute_ethical_triage_reward,
    compute_step_reward,
    compute_final_reward,
    normalise_score,
    clamp_score,
)


def test_successful_extubation_positive():
    r, _ = compute_extubation_reward(
        PatientState.READY_TO_EXTUBATE,
        PatientState.EXTUBATED,
        0.10,
    )
    assert r > 0


def test_reintubation_penalty_negative():
    r, _ = compute_extubation_reward(
        PatientState.READY_TO_EXTUBATE,
        PatientState.REINTUBATED,
        0.50,
    )
    assert r < 0


def test_high_risk_reintubation_worse_than_low():
    r_high, _ = compute_extubation_reward(
        PatientState.READY_TO_EXTUBATE,
        PatientState.REINTUBATED,
        0.90,
    )
    r_low, _ = compute_extubation_reward(
        PatientState.READY_TO_EXTUBATE,
        PatientState.REINTUBATED,
        0.10,
    )
    assert abs(r_high) > abs(r_low)


def test_vap_bundle_ineligible_patient_zero():
    r, _ = compute_vap_bundle_reward(hours_on_vent=10, hours_since_last_bundle=2)
    assert r == 0.0


def test_vap_bundle_eligible_patient_positive():
    r, _ = compute_vap_bundle_reward(hours_on_vent=72, hours_since_last_bundle=5)
    assert r > 0


def test_alarm_tp_positive():
    r, e = compute_alarm_reward(is_real=True, agent_responded=True)
    assert r > 0
    assert e == "alarm_tp"


def test_alarm_fn_negative():
    r, e = compute_alarm_reward(is_real=True, agent_responded=False)
    assert r < 0
    assert e == "alarm_fn"


def test_alarm_fn_worse_than_fp():
    r_fn, _ = compute_alarm_reward(is_real=True, agent_responded=False)
    r_fp, _ = compute_alarm_reward(is_real=False, agent_responded=True)
    assert abs(r_fn) > abs(r_fp)


def test_ethical_triage_correct_positive():
    r, _ = compute_ethical_triage_reward(is_correct=True)
    assert r > 0


def test_ethical_triage_wrong_negative():
    r, _ = compute_ethical_triage_reward(is_correct=False)
    assert r < 0


def test_clamp_score_bounds():
    assert clamp_score(1.5) == 1.0
    assert clamp_score(-0.5) == 0.0
    assert clamp_score(0.5) == 0.5


def test_normalise_score():
    assert normalise_score(5.0, 0.0, 10.0) == 0.5
    assert normalise_score(0.0, 0.0, 10.0) == 0.0
    assert normalise_score(10.0, 0.0, 10.0) == 1.0


def test_step_reward_consolidation():
    action_rewards = [
        (0.35, "extubation_success"),
        (-0.30, "alarm_fn"),
    ]
    breakdown = compute_step_reward(action_rewards, [])
    assert breakdown.total == pytest.approx(0.05, abs=0.001)
    assert breakdown.extubation_reward == 0.35
    assert breakdown.alarm_missed_penalty == -0.30


def test_final_reward_clean_episode():
    bonus, desc = compute_final_reward(
        total_steps=6,
        total_vap_incidents=0,
        total_reintubations=0,
        no_vent_penalties=0,
        triage_accuracy=1.0,
        alarm_accuracy=0.90,
    )
    assert bonus > 0
    assert "clean_episode_bonus" in desc