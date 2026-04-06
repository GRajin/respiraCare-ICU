# app/environment/episode.py
# =============================================================================
# RespiraCare-ICU — Episode Lifecycle Manager
#
# The Episode class is the single object that inference.py and the FastAPI
# server interact with. It wraps WardStateManager and owns the full
# request/response cycle for one complete episode.
#
# Public interface:
#   reset(task_id, seed)       → ResetResponse
#   step(actions)              → EpisodeResponse
#   get_state()                → dict  (for /state endpoint)
#   is_done                    → bool
#
# Step execution order (each call to step()):
#   1. Validate episode is not already done
#   2. Apply agent actions via ward.apply_actions()
#   3. Advance simulation one hour via ward.advance_hour()
#   4. Build new observation via ward.get_observation()
#   5. Check done condition
#   6. If done: apply final reward adjustments + resolve unanswered triage
#   7. Store step record in history
#   8. Return EpisodeResponse
#
# The full step history is accessible to graders via get_history().
# =============================================================================

from __future__ import annotations

import time
from typing import List, Dict, Any, Optional

from app.environment.ward import WardStateManager
from app.reward.reward_function import compute_final_reward
from app.models import (
    Action,
    ActionType,
    EpisodeResponse,
    ResetResponse,
    RewardBreakdown,
    Observation,
)
from app import config


# =============================================================================
# STEP RECORD — stored in history for grader access
# =============================================================================

class StepRecord:
    """
    Complete record of one episode step.
    Stored in Episode.history — graders read this at episode end.
    """

    def __init__(
        self,
        step_number: int,
        hour_before: int,
        hour_after: int,
        actions: List[Action],
        reward: float,
        breakdown: RewardBreakdown,
        events: List[str],
        tick_events: List[str],
        observation_after: Observation,
        done: bool,
        timestamp: float,
    ):
        self.step_number = step_number
        self.hour_before = hour_before
        self.hour_after = hour_after
        self.actions = actions
        self.reward = reward
        self.breakdown = breakdown
        self.events = events
        self.tick_events = tick_events
        self.observation_after = observation_after
        self.done = done
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "hour_before": self.hour_before,
            "hour_after": self.hour_after,
            "actions": [
                {
                    "patient_id": a.patient_id,
                    "action_type": a.action_type.value,
                    "priority": a.priority,
                }
                for a in self.actions
            ],
            "reward": self.reward,
            "reward_breakdown": {
                "total": self.breakdown.total,
                "extubation": self.breakdown.extubation_reward,
                "reintubation": self.breakdown.reintubation_penalty,
                "sbt": self.breakdown.sbt_reward,
                "vap_bundle": self.breakdown.vap_bundle_reward,
                "rt_assignment": self.breakdown.rt_assignment_reward,
                "rt_misallocation": self.breakdown.rt_misallocation_penalty,
                "alarm_tp": self.breakdown.alarm_true_positive_reward,
                "alarm_tn": self.breakdown.alarm_false_negative_reward,
                "alarm_fp": self.breakdown.alarm_false_escalation_penalty,
                "alarm_fn": self.breakdown.alarm_missed_penalty,
                "vap_penalty": self.breakdown.vap_develops_penalty,
                "no_vent_penalty": self.breakdown.no_vent_available_penalty,
                "bipap_penalty": self.breakdown.bipap_deterioration_penalty,
                "triage_reward": self.breakdown.ethical_triage_reward,
                "triage_penalty": self.breakdown.ethical_triage_penalty,
            },
            "events": self.events,
            "tick_events": self.tick_events,
            "done": self.done,
        }


# =============================================================================
# EPISODE
# =============================================================================

class Episode:
    """
    Manages the complete lifecycle of one RespiraCare-ICU episode.

    One Episode instance maps to one session in the FastAPI server.
    The session store (main.py) holds a dict of session_id → Episode.

    Usage:
        episode = Episode()
        reset_resp = episode.reset(task_id=2, seed=42)
        while not episode.is_done:
            actions = agent.decide(reset_resp.observation)
            response = episode.step(actions)
    """

    def __init__(self):
        self.ward = WardStateManager()
        self.task_id: int = 0
        self.seed: int = 0
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.history: List[StepRecord] = []
        self._started: bool = False
        self._done: bool = False
        self._start_time: float = 0.0
        self._final_adjustment: float = 0.0
        self._final_adjustment_desc: str = ""

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self, task_id: int, seed: int) -> ResetResponse:
        """
        Initialise a fresh episode. Can be called multiple times to restart.
        Returns a ResetResponse with session_id, initial observation, and
        task description.
        """
        self.task_id = task_id
        self.seed = seed
        self.step_count = 0
        self.total_reward = 0.0
        self.history = []
        self._started = True
        self._done = False
        self._start_time = time.time()
        self._final_adjustment = 0.0
        self._final_adjustment_desc = ""

        # Delegate to ward — returns initial Observation
        initial_obs = self.ward.reset(task_id=task_id, seed=seed)

        return ResetResponse(
            session_id=self.ward.episode_id,
            observation=initial_obs,
            task_description=self.ward.task_description,
        )

    # =========================================================================
    # STEP
    # =========================================================================

    def step(self, actions: List[Action]) -> EpisodeResponse:
        """
        Execute one step of the episode.

        Args:
            actions: List of Action objects from the agent.
                     Patients not mentioned receive HOLD_AND_MONITOR.

        Returns:
            EpisodeResponse with new observation, reward, done flag, and info.

        Raises:
            RuntimeError if called before reset() or after episode is done.
        """
        if not self._started:
            raise RuntimeError(
                "Episode.step() called before reset(). Call reset() first."
            )
        if self._done:
            raise RuntimeError(
                "Episode.step() called on a completed episode. "
                "Call reset() to start a new episode."
            )

        hour_before = self.ward.current_hour
        self.step_count += 1

        # -----------------------------------------------------------------
        # 1. Apply agent actions
        # -----------------------------------------------------------------
        action_reward, action_events, action_breakdown = self.ward.apply_actions(actions)

        # -----------------------------------------------------------------
        # 2. Advance simulation one hour
        # -----------------------------------------------------------------
        tick_events, tick_breakdown = self.ward.advance_hour()

        # Merge tick breakdown into action breakdown
        action_breakdown.total += tick_breakdown.total
        action_breakdown.vap_develops_penalty += tick_breakdown.vap_develops_penalty
        action_breakdown.no_vent_available_penalty += tick_breakdown.no_vent_available_penalty
        action_breakdown.bipap_deterioration_penalty += tick_breakdown.bipap_deterioration_penalty
        action_breakdown.total = round(action_breakdown.total, 4)

        # -----------------------------------------------------------------
        # 3. Build new observation
        # -----------------------------------------------------------------
        hour_after = self.ward.current_hour
        new_obs = self.ward.get_observation(hour_after)

        # -----------------------------------------------------------------
        # 4. Check done condition
        # -----------------------------------------------------------------
        done = self.ward.is_done
        final_adjustment = 0.0
        final_desc = ""

        if done:
            # Resolve any unanswered ethical triage cases
            unanswered = self.ward.triage.resolve_unanswered_cases()
            for penalty, event in unanswered:
                action_breakdown.ethical_triage_penalty += penalty
                action_breakdown.total += penalty
                action_events.append(f"triage_auto_resolved:{event}")

            # Compute end-of-episode bonus/penalty
            vap_stats = self.ward.vap.get_compliance_stats()
            alarm_stats = self.ward.alarms.get_alarm_accuracy_stats()
            triage_stats = self.ward.triage.get_triage_stats()

            # Count reintubations from history
            total_reintubations = sum(
                1 for record in self.history
                for event in record.events
                if "reintubated" in event
            )

            final_adjustment, final_desc = compute_final_reward(
                total_steps=self.step_count,
                total_vap_incidents=vap_stats["total_vap_incidents"],
                total_reintubations=total_reintubations,
                no_vent_penalties=self.ward.fleet.equipment.no_vent_penalty_count,
                triage_accuracy=triage_stats["accuracy"],
                alarm_accuracy=alarm_stats["accuracy_score"],
            )
            action_breakdown.total += final_adjustment
            action_breakdown.total = round(action_breakdown.total, 4)
            self._final_adjustment = final_adjustment
            self._final_adjustment_desc = final_desc
            self._done = True

        # -----------------------------------------------------------------
        # 5. Accumulate total reward
        # -----------------------------------------------------------------
        step_reward = action_breakdown.total
        self.total_reward += step_reward

        # -----------------------------------------------------------------
        # 6. Store step record
        # -----------------------------------------------------------------
        record = StepRecord(
            step_number=self.step_count,
            hour_before=hour_before,
            hour_after=hour_after,
            actions=actions,
            reward=step_reward,
            breakdown=action_breakdown,
            events=action_events,
            tick_events=tick_events,
            observation_after=new_obs,
            done=done,
            timestamp=time.time(),
        )
        self.history.append(record)

        # -----------------------------------------------------------------
        # 7. Build info dict
        # -----------------------------------------------------------------
        info: dict = {
            "step": self.step_count,
            "hour": hour_after,
            "total_reward": round(self.total_reward, 4),
            "reward_breakdown": record.to_dict()["reward_breakdown"],
            "events": action_events[:10],   # First 10 events to keep payload small
            "tick_events": tick_events[:10],
        }

        if done:
            info["final_adjustment"] = final_adjustment
            info["final_adjustment_desc"] = final_desc
            info["episode_summary"] = self._build_episode_summary()

        return EpisodeResponse(
            observation=new_obs,
            reward=step_reward,
            done=done,
            info=info,
        )

    # =========================================================================
    # EPISODE SUMMARY — built at done=True
    # =========================================================================

    def _build_episode_summary(self) -> Dict[str, Any]:
        """
        Build the full episode summary included in the final EpisodeResponse.
        This is what graders use as their primary data source.
        """
        vap_stats = self.ward.vap.get_compliance_stats()
        alarm_stats = self.ward.alarms.get_alarm_accuracy_stats()
        triage_stats = self.ward.triage.get_triage_stats()
        handover_stats = self.ward.handover.get_handover_stats()
        fleet_summary = self.ward.fleet.get_summary()

        # Per-step reward history
        reward_history = [
            {"step": r.step_number, "hour": r.hour_after, "reward": r.reward}
            for r in self.history
        ]

        # Patient outcome summary
        patient_outcomes = []
        for p in self.ward.patients:
            patient_outcomes.append({
                "patient_id": p.patient_id,
                "final_state": p.state.value,
                "final_support": p.support_level.value,
                "sofa_score": p.sofa_score,
                "has_vap": p.has_vap,
                "hours_on_vent": p.hours_on_vent,
                "reintubation_risk": round(p.reintubation_risk, 3),
            })

        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "total_steps": self.step_count,
            "total_reward": round(self.total_reward, 4),
            "final_adjustment": self._final_adjustment,
            "final_adjustment_desc": self._final_adjustment_desc,
            "reward_history": reward_history,
            "patient_outcomes": patient_outcomes,
            "vap_stats": vap_stats,
            "alarm_stats": alarm_stats,
            "triage_stats": triage_stats,
            "handover_stats": handover_stats,
            "fleet_summary": fleet_summary,
            "elapsed_seconds": round(time.time() - self._start_time, 2),
        }

    # =========================================================================
    # ACCESSORS
    # =========================================================================

    @property
    def is_done(self) -> bool:
        return self._done

    def get_history(self) -> List[StepRecord]:
        """Full step history — used by graders."""
        return self.history

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the full internal state dict for the GET /state endpoint.
        Includes ground-truth data not visible to the agent.
        """
        state = self.ward.get_state_summary()
        state["step_count"] = self.step_count
        state["total_reward"] = round(self.total_reward, 4)
        state["is_done"] = self._done
        state["elapsed_seconds"] = round(time.time() - self._start_time, 2)
        return state