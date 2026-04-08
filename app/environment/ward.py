# app/environment/ward.py
# =============================================================================
# RespiraCare-ICU — ICU Ward State Manager
#
# The central coordinator of the entire simulation. Every agent action flows
# through here. Every observation is assembled here. Every reward signal
# originates here.
#
# Owns:
#   - The list of PatientStateMachine objects (one per bed)
#   - All 5 problem simulators (fleet, alarm, VAP, triage, handover)
#   - The episode step counter and task configuration
#
# Public interface (called by episode.py):
#   reset(task_id, seed)          → Observation
#   apply_actions(actions)        → (total_reward, events, info)
#   advance_hour()                → (events, penalties)
#   get_observation(hour)         → Observation
#   get_state_summary()           → dict  (for /state endpoint)
#
# Step execution order (called by episode.py each turn):
#   1. apply_actions(actions)     — process all agent actions
#   2. advance_hour()             — tick all patients + all problem modules
#   3. get_observation(new_hour)  — assemble next observation
# =============================================================================

from __future__ import annotations

import uuid
from typing import List, Tuple, Dict, Any, Optional

from app.environment.patient import PatientStateMachine
from app.environment.patient_generator import generate_ward, get_task_description
from app.problems.fleet_management import FleetManager
from app.problems.alarm_fatigue import AlarmFatigueSimulator
from app.problems.vap_prevention import VAPPreventionCoordinator
from app.problems.ethical_triage import EthicalTriageEngine
from app.problems.handover import HandoverCoordinator
from app.models import (
    Action,
    ActionType,
    Observation,
    PatientObservation,
    RewardBreakdown,
    SupportLevel,
    PatientState,
)
from app import config


class WardStateManager:
    """
    Manages the complete state of one ICU episode.

    One WardStateManager instance is created per episode (per session).
    It holds the ward's full ground-truth state and exposes only the
    agent-visible Observation to the outside world.
    """

    def __init__(self):
        # Set during reset()
        self.task_id: int = 0
        self.seed: int = 0
        self.episode_id: str = ""
        self.current_hour: int = 0
        self.max_hours: int = 0
        self.task_description: str = ""

        # Patients — indexed by patient_id for O(1) lookup
        self.patients: List[PatientStateMachine] = []
        self._patient_map: Dict[str, PatientStateMachine] = {}

        # Problem simulators — initialised in reset()
        self.fleet: Optional[FleetManager] = None
        self.alarms: Optional[AlarmFatigueSimulator] = None
        self.vap: Optional[VAPPreventionCoordinator] = None
        self.triage: Optional[EthicalTriageEngine] = None
        self.handover: Optional[HandoverCoordinator] = None

        # Episode-level accumulators
        self.total_reward: float = 0.0
        self.step_history: List[dict] = []
        self.event_log: List[str] = []

        # RT usage this step
        self._rts_used_this_step: int = 0

    # =========================================================================
    # RESET — initialise a fresh episode
    # =========================================================================

    def reset(self, task_id: int, seed: int) -> Observation:
        """
        Initialise a completely fresh episode for the given task and seed.
        Returns the initial Observation.
        """
        self.task_id = task_id
        self.seed = seed
        self.episode_id = str(uuid.uuid4())[:8]
        self.current_hour = 0
        self.total_reward = 0.0
        self.step_history = []
        self.event_log = []
        self._rts_used_this_step = 0

        # Set episode length from task
        hour_map = {
            1: config.TASK1_MAX_HOURS,
            2: config.TASK2_MAX_HOURS,
            3: config.TASK3_MAX_HOURS,
        }
        self.max_hours = hour_map.get(task_id, config.TASK2_MAX_HOURS)
        self.task_description = get_task_description(task_id)

        # Generate patients
        self.patients = generate_ward(task_id, seed)
        self._patient_map = {p.patient_id: p for p in self.patients}

        # Initialise all 5 problem simulators
        self.fleet = FleetManager(task_id=task_id, seed=seed)
        self.fleet.initialise_from_patients(self.patients)

        self.alarms = AlarmFatigueSimulator(seed=seed)
        for p in self.patients:
            self.alarms.initialise_patient(p.patient_id)

        self.vap = VAPPreventionCoordinator(seed=seed)
        self.vap.initialise_from_patients(self.patients)

        self.triage = EthicalTriageEngine(seed=seed)

        self.handover = HandoverCoordinator(seed=seed)

        # Generate initial alarm feed for hour 0
        self.alarms.generate_alarms(self.patients, hour=0)

        return self.get_observation(hour=0)

    # =========================================================================
    # APPLY ACTIONS — process all agent actions for one step
    # =========================================================================

    def apply_actions(
        self, actions: List[Action]
    ) -> Tuple[float, List[str], RewardBreakdown]:
        """
        Process all agent actions for this step.

        Execution order per action:
          1. Validate equipment availability (fleet)
          2. Apply action to patient state machine
          3. Record equipment changes (fleet)
          4. Record alarm responses (alarm fatigue)
          5. Record VAP bundle actions (vap)
          6. Record ethical triage decisions (triage)

        Returns (total_reward, events, reward_breakdown).
        """
        breakdown = RewardBreakdown(total=0.0)
        events: List[str] = []
        self._rts_used_this_step = 0

        # Build a lookup: patient_id → action
        # Patients not mentioned get HOLD_AND_MONITOR implicitly
        action_map: Dict[str, Action] = {a.patient_id: a for a in actions}

        for patient in self.patients:
            if patient.is_discharged:
                continue

            action = action_map.get(
                patient.patient_id,
                Action(
                    patient_id=patient.patient_id,
                    action_type=ActionType.HOLD_AND_MONITOR,
                )
            )

            action_type = action.action_type

            # -----------------------------------------------------------------
            # 1. VALIDATE equipment availability
            # -----------------------------------------------------------------
            is_valid, reason = self.fleet.validate_action(
                patient.patient_id,
                action_type,
                patient.support_level,
            )
            if not is_valid:
                events.append(
                    f"action_blocked:{patient.patient_id}:{action_type.value}:{reason}"
                )
                continue

            # -----------------------------------------------------------------
            # 2. RT RESOURCE CHECK
            # ASSIGN_RT_ATTENTION consumes one RT slot per step
            # -----------------------------------------------------------------
            if action_type == ActionType.ASSIGN_RT_ATTENTION:
                if self._rts_used_this_step >= self.fleet.available_ventilators:
                    # Use available_rts from ward resources
                    pass   # RT tracking handled below

                rt_crisis_hour = 11
                available_rts = (
                    config.WARD_RESPIRATORY_THERAPISTS
                    if self.current_hour < rt_crisis_hour
                    else config.WARD_RESPIRATORY_THERAPISTS - 1
                )
                if self._rts_used_this_step >= available_rts:
                    events.append(
                        f"rt_unavailable:{patient.patient_id}"
                    )
                    # Apply as hold instead
                    action_type = ActionType.HOLD_AND_MONITOR

            # -----------------------------------------------------------------
            # 3. APPLY ACTION to patient state machine
            # -----------------------------------------------------------------
            reward_delta, event_str = patient.apply_action(action_type)
            events.append(f"{patient.patient_id}:{event_str}")

            # -----------------------------------------------------------------
            # 4. POST-ACTION ACCOUNTING
            # -----------------------------------------------------------------

            # Fleet: record new support level after action
            self.fleet.record_action(
                patient.patient_id,
                action_type,
                patient.support_level,
            )

            # Alarm fatigue: record agent's alarm response
            if action_type in (
                ActionType.RESPOND_TO_ALARM,
                ActionType.SUPPRESS_ALARM,
            ):
                alarm_reward, alarm_event = self.alarms.record_agent_response(
                    patient.patient_id,
                    action_type,
                    self.current_hour,
                )
                reward_delta += alarm_reward
                events.append(alarm_event)

                # Update breakdown
                if action_type == ActionType.RESPOND_TO_ALARM:
                    if alarm_reward > 0:
                        breakdown.alarm_true_positive_reward += alarm_reward
                    else:
                        breakdown.alarm_false_escalation_penalty += alarm_reward
                else:
                    if alarm_reward > 0:
                        breakdown.alarm_false_negative_reward += alarm_reward
                    else:
                        breakdown.alarm_missed_penalty += alarm_reward

            # VAP prevention: record bundle action
            if action_type == ActionType.ENFORCE_VAP_BUNDLE:
                self.vap.record_bundle_action(patient.patient_id)
                breakdown.vap_bundle_reward += reward_delta

            # RT assignment tracking
            if action_type == ActionType.ASSIGN_RT_ATTENTION:
                self._rts_used_this_step += 1
                if reward_delta > 0:
                    breakdown.rt_assignment_reward += reward_delta
                else:
                    breakdown.rt_misallocation_penalty += reward_delta

            # Extubation accounting
            if action_type == ActionType.EXTUBATE:
                if "successful" in event_str:
                    breakdown.extubation_reward += reward_delta
                elif "reintubated" in event_str:
                    breakdown.reintubation_penalty += reward_delta

            # SBT accounting
            if action_type == ActionType.ATTEMPT_SBT:
                breakdown.sbt_reward += reward_delta

            # Ethical triage decision
            if action_type == ActionType.ETHICAL_TRIAGE_SELECT:
                triage_case_id = self._get_active_triage_case_id()
                if triage_case_id and action.ethical_triage_patient_id:
                    triage_reward, triage_event = self.triage.record_agent_decision(
                        triage_case_id,
                        action.ethical_triage_patient_id,
                    )
                    reward_delta += triage_reward
                    events.append(triage_event)
                    if triage_reward > 0:
                        breakdown.ethical_triage_reward += triage_reward
                    else:
                        breakdown.ethical_triage_penalty += triage_reward

            # Accumulate total reward
            breakdown.total += reward_delta

        # -----------------------------------------------------------------
        # Fleet penalties: no vent available for incoming patient
        # -----------------------------------------------------------------
        fleet_penalties = self.fleet.pop_pending_penalties()
        for penalty_str in fleet_penalties:
            if "no_vent" in penalty_str:
                breakdown.no_vent_available_penalty += config.PENALTY_NO_VENTILATOR_FOR_INCOMING
                breakdown.total += config.PENALTY_NO_VENTILATOR_FOR_INCOMING
                events.append(f"penalty:{penalty_str}")

        self.total_reward += breakdown.total
        return breakdown.total, events, breakdown

    # =========================================================================
    # ADVANCE HOUR — tick all modules forward
    # =========================================================================

    def advance_hour(self) -> Tuple[List[str], RewardBreakdown]:
        """
        Advance the entire ward one simulated hour.

        Execution order:
          1. Tick each patient (physiology, VAP onset check)
          2. Tick fleet (incoming patient arrivals)
          3. Tick VAP coordinator (compliance logging, infection events)
          4. Tick ethical triage (hour counter)
          5. Tick handover (degradation trigger/clear)
          6. Generate next hour's alarm feed
          7. Increment ward hour counter

        Returns (events, reward_breakdown_for_this_tick).
        """
        events: List[str] = []
        tick_breakdown = RewardBreakdown(total=0.0)

        # 1. Tick all patients
        for patient in self.patients:
            if patient.is_discharged:
                continue
            patient_events = patient.advance_hour()
            events.extend(patient_events)

        # 2. Tick fleet
        fleet_events = self.fleet.advance_hour()
        events.extend(fleet_events)

        # Fleet penalties from this tick
        fleet_penalties = self.fleet.pop_pending_penalties()
        for penalty_str in fleet_penalties:
            if "no_vent" in penalty_str:
                tick_breakdown.no_vent_available_penalty += (
                    config.PENALTY_NO_VENTILATOR_FOR_INCOMING
                )
                tick_breakdown.total += config.PENALTY_NO_VENTILATOR_FOR_INCOMING
                events.append(f"penalty:{penalty_str}")

        # 3. Tick VAP coordinator
        vap_events = self.vap.advance_hour(self.patients)
        events.extend(vap_events)

        # VAP penalties
        vap_penalties = self.vap.pop_pending_penalties()
        for _ in vap_penalties:
            tick_breakdown.vap_develops_penalty += config.PENALTY_VAP_DEVELOPS
            tick_breakdown.total += config.PENALTY_VAP_DEVELOPS
            events.append("penalty:vap_developed")

        # 4. Tick ethical triage
        self.triage.advance_hour()

        # 5. Tick handover
        handover_events = self.handover.advance_hour(self.patients)
        events.extend(handover_events)

        # 6. Generate alarm feed for the next hour
        self.current_hour += 1
        self.alarms.generate_alarms(self.patients, hour=self.current_hour)

        # 7. Log to event log
        self.event_log.extend(events)
        self.total_reward += tick_breakdown.total

        return events, tick_breakdown

    # =========================================================================
    # BUILD OBSERVATION
    # =========================================================================

    def get_observation(self, hour: int) -> Observation:
        """
        Assemble the full Observation for the agent.
        Applies handover degradation to appropriate patients.
        """
        # Determine which patients have degraded observations
        degraded_ids = self.handover.get_degraded_patient_ids()
        is_handover = self.handover.is_handover_active(hour)

        # If handover just became active, trigger degradation selection
        if is_handover and not degraded_ids:
            self.handover.apply_degradation(self.patients, hour)
            degraded_ids = self.handover.get_degraded_patient_ids()

        # Build per-patient observations
        patient_obs: List[PatientObservation] = []
        for patient in self.patients:
            if patient.is_discharged:
                continue
            is_degraded = patient.patient_id in degraded_ids
            obs = patient.build_observation(
                degraded=is_degraded,
                current_hour=hour,
            )
            patient_obs.append(obs)

        # Check for active ethical triage case
        triage_case = self.triage.check_triage_needed(
            self.patients,
            self.fleet.available_ventilators,
        )

        return Observation(
            episode_id=self.episode_id,
            task_id=self.task_id,
            shift_hour=hour,
            available_ventilators=self.fleet.available_ventilators,
            available_bipap=self.fleet.available_bipap,
            available_hfnc=self.fleet.available_hfnc,
            available_rts=self._get_available_rts(),
            patients=patient_obs,
            active_alarms=self.alarms.get_active_alarms(),
            incoming_patients=self.fleet.get_active_alerts(),
            handover_degraded=is_handover,
            triage_decision_required=triage_case,
        )

    def _get_available_rts(self) -> int:
        """
        Returns current available RT count.
        Reduced by 1 after RT_UNAVAILABLE crisis in Task 3 hour 11.
        """
        base = config.WARD_RESPIRATORY_THERAPISTS
        rt_crisis_hour = 11   # From TASK3_CRISIS_SCHEDULE key
        if self.task_id == 3 and self.current_hour >= rt_crisis_hour:
            return max(0, base - 1)
        return base

    def _get_active_triage_case_id(self) -> Optional[str]:
        """Returns the active triage case ID if one exists."""
        if self.triage._active_case_id:
            return self.triage._active_case_id
        return None

    # =========================================================================
    # EPISODE STATE
    # =========================================================================

    @property
    def is_done(self) -> bool:
        """True when the episode has reached its maximum hours."""
        return self.current_hour >= self.max_hours

    # =========================================================================
    # STATE SUMMARY — for /state endpoint
    # =========================================================================

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Returns the full internal state of the ward.
        Used by the GET /state endpoint — not sent to agent during normal play.
        Contains ground-truth data including is_real alarm flags.
        """
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "current_hour": self.current_hour,
            "max_hours": self.max_hours,
            "is_done": self.is_done,
            "total_reward": round(self.total_reward, 4),
            "fleet": self.fleet.get_summary(),
            "vap": self.vap.get_compliance_stats(),
            "triage": self.triage.get_triage_stats(),
            "handover": self.handover.get_handover_stats(),
            "alarm_accuracy": self.alarms.get_alarm_accuracy_stats(),
            "patients": [
                {
                    "patient_id": p.patient_id,
                    "state": p.state.value,
                    "support_level": p.support_level.value,
                    "sofa_score": p.sofa_score,
                    "vap_risk": round(p.vap_risk, 3),
                    "has_vap": p.has_vap,
                    "reintubation_risk": round(p.reintubation_risk, 3),
                    "hours_on_vent": p.hours_on_vent,
                }
                for p in self.patients
            ],
            "event_log": self.event_log[-50:],   # Last 50 events
        }