# app/environment/patient.py
# =============================================================================
# RespiraCare-ICU — Individual Patient State Machine
#
# Each ventilated patient in the ward is represented by one PatientStateMachine
# instance. It holds the patient's full internal clinical state (including fields
# hidden from the agent), processes actions applied to it, and advances physiology
# one simulated hour at a time.
#
# Two public methods drive everything:
#   apply_action(action)  — processes one ActionType against this patient
#   advance_hour()        — ticks physiology forward one simulated hour
#
# The build_observation() method produces the agent-visible PatientObservation
# snapshot — it deliberately omits internal fields like is_real alarm data.
# =============================================================================

from __future__ import annotations

import random
from typing import Optional, List, Tuple

from app.models import (
    PatientState,
    SupportLevel,
    Severity,
    ActionType,
    PatientObservation,
    PendingVAPEvent,
)
from app import config


class PatientStateMachine:
    """
    Represents the full clinical state of a single ventilated ICU patient.

    Internal state (not all exposed to agent):
        - All vital signs and scores
        - VAP risk accumulator and pending VAP events
        - Reintubation risk at the moment of extubation
        - SBT history
        - RT attention flag for this hour
        - Alarm history counters

    The patient_generator.py module creates instances of this class with
    seeded randomness so episodes are reproducible.
    """

    def __init__(
        self,
        patient_id: str,
        state: PatientState,
        support_level: SupportLevel,
        severity: Severity,
        rsbi: Optional[float],
        pf_ratio: Optional[float],
        rass: float,
        fio2: float,
        peep: float,
        spo2: float,
        heart_rate: float,
        resp_rate: float,
        hemodynamically_stable: bool,
        hours_on_vent: int,
        sofa_score: int,
        vap_risk: float,
        rng: random.Random,
    ):
        # --- Identity ---
        self.patient_id = patient_id
        self.severity = severity

        # --- Clinical state ---
        self.state = state
        self.support_level = support_level

        # --- Vital signs ---
        self.rsbi = rsbi
        self.pf_ratio = pf_ratio
        self.rass = rass
        self.fio2 = fio2
        self.peep = peep
        self.spo2 = spo2
        self.heart_rate = heart_rate
        self.resp_rate = resp_rate
        self.hemodynamically_stable = hemodynamically_stable

        # --- Ventilator history ---
        self.hours_on_vent = hours_on_vent
        self.sbt_passed_at_hour: Optional[int] = None   # Episode hour when last SBT passed
        self.sbt_in_progress_since: Optional[int] = None

        # --- Risk scores ---
        self.sofa_score = sofa_score
        self.reintubation_risk_at_extubation: Optional[float] = None  # locked at extubation

        # --- VAP prevention ---
        self.vap_risk = vap_risk
        self.vap_bundle_compliance_hours = 0
        self.hours_since_last_bundle = 0
        self.pending_vap_events: List[PendingVAPEvent] = []
        self.has_vap = (state == PatientState.INTUBATED_WITH_VAP)

        # Reintubation risk must be computed AFTER has_vap is set
        self.reintubation_risk = self._compute_reintubation_risk()

        # --- Alarm counters (used by alarm_fatigue.py) ---
        self.consecutive_false_alarms: dict = {}   # alarm_type -> count
        self.alarm_history_this_hour: List[str] = []

        # --- RT attention this hour ---
        self.rt_assigned_this_hour = False

        # --- Episode hour tracker ---
        self.current_hour = 0

        # --- Seeded RNG (must be passed in for reproducibility) ---
        self._rng = rng

        # --- Action applied this step (prevents double-actions) ---
        self._action_applied_this_step = False

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _compute_reintubation_risk(self) -> float:
        """
        Derive a continuous reintubation risk score from current vitals.
        Higher risk when: RSBI high, PF ratio low, hemodynamically unstable,
        RASS out of range, patient on high FiO2/PEEP support.

        Returns a value in [0.0, 1.0].
        """
        risk = 0.0

        # RSBI contribution — most important weaning predictor
        if self.rsbi is not None:
            if self.rsbi >= 130:
                risk += 0.35
            elif self.rsbi >= 105:
                risk += 0.20
            elif self.rsbi >= 80:
                risk += 0.08
            else:
                risk += 0.02   # RSBI well below threshold — low contribution

        # PF ratio contribution
        if self.pf_ratio is not None:
            if self.pf_ratio < 150:
                risk += 0.25
            elif self.pf_ratio < 200:
                risk += 0.15
            elif self.pf_ratio < 300:
                risk += 0.05
            else:
                risk += 0.01

        # Hemodynamic stability
        if not self.hemodynamically_stable:
            risk += 0.20

        # RASS — sedation level
        if self.rass < -2 or self.rass > 0:
            risk += 0.10

        # High support requirements
        if self.fio2 > 0.50:
            risk += 0.08
        if self.peep > 8:
            risk += 0.06

        # VAP makes extubation riskier
        if self.has_vap:
            risk += 0.10

        return min(risk, 1.0)

    def _meets_sbt_criteria(self) -> bool:
        """
        Check whether this patient meets all ACCP criteria for a
        spontaneous breathing trial attempt.
        """
        if self.rsbi is None or self.pf_ratio is None:
            return False
        if self.rsbi >= config.RSBI_THRESHOLD:
            return False
        if self.pf_ratio < config.PF_RATIO_THRESHOLD:
            return False
        if self.rass < config.RASS_MIN or self.rass > config.RASS_MAX:
            return False
        if not self.hemodynamically_stable:
            return False
        if self.fio2 > config.FIO2_MAX_FOR_SBT:
            return False
        if self.peep > config.PEEP_MAX_FOR_SBT:
            return False
        if self.hours_on_vent < config.HOURS_ON_VENT_BEFORE_SBT:
            return False
        return True

    def _sbt_passed_recently(self, current_hour: int) -> bool:
        """
        True if a successful SBT occurred within the last SBT_DURATION_HOURS.
        """
        if self.sbt_passed_at_hour is None:
            return False
        return (current_hour - self.sbt_passed_at_hour) <= config.SBT_DURATION_HOURS

    def _meets_extubation_criteria(self, current_hour: int) -> bool:
        """
        Patient is ready to extubate if they meet SBT criteria AND passed
        an SBT within the valid window.
        """
        return self._meets_sbt_criteria() and self._sbt_passed_recently(current_hour)

    def _improve_vitals_slightly(self) -> None:
        """
        Stable patients improve slowly each hour without intervention.
        Simulates the natural recovery trajectory of an ICU patient
        who is getting better.
        """
        if self.rsbi is not None:
            self.rsbi = max(40.0, self.rsbi - self._rng.uniform(1.0, 4.0))
        if self.pf_ratio is not None:
            self.pf_ratio = min(400.0, self.pf_ratio + self._rng.uniform(2.0, 8.0))
        self.spo2 = min(100.0, self.spo2 + self._rng.uniform(0.0, 0.5))
        if self.fio2 > 0.21:
            self.fio2 = max(0.21, self.fio2 - self._rng.uniform(0.01, 0.03))
        if self.peep > 5:
            self.peep = max(5.0, self.peep - self._rng.uniform(0.0, 0.5))

    def _deteriorate_vitals_slightly(self) -> None:
        """
        Unstable patients or those without RT attention can worsen.
        """
        if self.rsbi is not None:
            self.rsbi = min(180.0, self.rsbi + self._rng.uniform(1.0, 6.0))
        if self.pf_ratio is not None:
            self.pf_ratio = max(80.0, self.pf_ratio - self._rng.uniform(2.0, 10.0))
        self.spo2 = max(80.0, self.spo2 - self._rng.uniform(0.0, 1.0))
        self.resp_rate = min(40.0, self.resp_rate + self._rng.uniform(0.0, 2.0))

    def _accrue_vap_risk(self) -> Optional[PendingVAPEvent]:
        """
        Called each hour the patient is ventilated WITHOUT bundle enforcement.
        Accumulates risk. If the threshold is crossed, queues a VAP infection
        event to fire after VAP_ONSET_DELAY_HOURS.

        Returns a PendingVAPEvent if threshold was just crossed, else None.
        """
        if self.hours_on_vent < config.VAP_ELIGIBLE_AFTER_HOURS:
            return None
        if self.has_vap:
            return None

        self.vap_risk = min(1.0, self.vap_risk + config.VAP_RISK_PER_MISSED_HOUR)
        self.hours_since_last_bundle += 1

        # Check if threshold just crossed — queue infection event
        if self.vap_risk >= config.VAP_TRIGGER_THRESHOLD:
            # Only queue if no event already pending
            already_pending = any(not e.fired for e in self.pending_vap_events)
            if not already_pending:
                event = PendingVAPEvent(
                    patient_id=self.patient_id,
                    risk_at_trigger=self.vap_risk,
                    trigger_hour=self.current_hour,
                    onset_hour=self.current_hour + config.VAP_ONSET_DELAY_HOURS,
                )
                self.pending_vap_events.append(event)
                return event

        return None

    def _check_vap_onset(self) -> bool:
        """
        Called each hour. Fires any pending VAP events whose onset_hour has arrived.
        Returns True if VAP developed this hour (so ward.py can apply the penalty).
        """
        for event in self.pending_vap_events:
            if not event.fired and self.current_hour >= event.onset_hour:
                event.fired = True
                self.has_vap = True
                self.state = PatientState.INTUBATED_WITH_VAP
                return True
        return False

    # =========================================================================
    # APPLY ACTION
    # Called by ward.py once per step for each patient.
    # Returns a tuple: (reward_delta, event_description)
    # =========================================================================

    def apply_action(self, action_type: ActionType) -> Tuple[float, str]:
        """
        Apply one action to this patient. Returns (reward_delta, event_str).
        The reward_delta is a partial contribution — the full reward function
        in reward_function.py applies the global weights and penalties.

        This method records *what happened* — the reward module scores *how well*.
        """
        self._action_applied_this_step = True
        reward = 0.0
        event = "no_event"

        # --- ATTEMPT_SBT ---
        if action_type == ActionType.ATTEMPT_SBT:
            if self.state not in (PatientState.INTUBATED_STABLE,):
                return 0.0, "sbt_invalid_state"
            if not self._meets_sbt_criteria():
                # SBT attempted on not-ready patient — no reward but no penalty here
                # (penalty comes from the reintubation if they're extubated later)
                return 0.0, "sbt_criteria_not_met"

            self.state = PatientState.SBT_IN_PROGRESS
            self.sbt_in_progress_since = self.current_hour

            # SBT outcome: pass or fail based on reintubation risk proxy
            fail_prob = self.reintubation_risk * 0.6   # risk is predictive but not certain
            if self._rng.random() < fail_prob:
                # SBT failed — patient goes back to stable
                self.state = PatientState.INTUBATED_STABLE
                event = "sbt_failed"
                reward = 0.0
            else:
                # SBT passed
                self.sbt_passed_at_hour = self.current_hour
                self.state = PatientState.READY_TO_EXTUBATE
                event = "sbt_passed"
                reward = config.REWARD_SBT_ATTEMPTED_OVERDUE

        # --- EXTUBATE ---
        elif action_type == ActionType.EXTUBATE:
            if self.state != PatientState.READY_TO_EXTUBATE:
                return 0.0, "extubate_invalid_state"

            # Lock in the risk at the moment of extubation (for weighted penalty)
            self.reintubation_risk_at_extubation = self.reintubation_risk

            # Extubation outcome
            reint_prob = self.reintubation_risk
            if self._rng.random() < reint_prob:
                # Failed extubation — reintubation required
                self.state = PatientState.REINTUBATED
                self.support_level = SupportLevel.FULL_VENTILATOR
                self.hours_on_vent += 1
                event = "extubation_failed_reintubated"
                # Penalty weight applied in reward_function.py using risk_at_extubation
                reward = config.PENALTY_REINTUBATION_BASE * (1.0 - reint_prob)
            else:
                # Successful extubation
                self.state = PatientState.EXTUBATED
                self.support_level = SupportLevel.ROOM_AIR
                self.rsbi = None
                self.pf_ratio = None
                event = "extubation_successful"
                reward = config.REWARD_SUCCESSFUL_EXTUBATION

        # --- STEP_DOWN_TO_BIPAP ---
        elif action_type == ActionType.STEP_DOWN_TO_BIPAP:
            if self.support_level != SupportLevel.FULL_VENTILATOR:
                return 0.0, "stepdown_bipap_invalid"
            if self.state in (PatientState.INTUBATED_UNSTABLE, PatientState.INTUBATED_WITH_VAP):
                return 0.0, "stepdown_bipap_too_sick"

            self.support_level = SupportLevel.BIPAP
            self.state = PatientState.EXTUBATED   # Tube removed, on mask
            self.rsbi = None
            event = "stepped_down_to_bipap"
            reward = config.REWARD_SUCCESSFUL_EXTUBATION * 0.6  # Partial reward

        # --- STEP_DOWN_TO_HFNC ---
        elif action_type == ActionType.STEP_DOWN_TO_HFNC:
            if self.support_level not in (SupportLevel.BIPAP, SupportLevel.FULL_VENTILATOR):
                return 0.0, "stepdown_hfnc_invalid"

            self.support_level = SupportLevel.HFNC
            event = "stepped_down_to_hfnc"
            reward = config.REWARD_SUCCESSFUL_EXTUBATION * 0.3

        # --- ESCALATE_TO_FULL_VENT ---
        elif action_type == ActionType.ESCALATE_TO_FULL_VENT:
            if self.support_level == SupportLevel.FULL_VENTILATOR:
                return 0.0, "escalate_already_on_vent"

            prev_support = self.support_level
            self.support_level = SupportLevel.FULL_VENTILATOR
            self.state = PatientState.INTUBATED_STABLE
            self.hours_on_vent += 1
            event = f"escalated_to_full_vent_from_{prev_support.value}"
            reward = 0.0   # Neutral — necessary action, not a reward event

        # --- ASSIGN_RT_ATTENTION ---
        elif action_type == ActionType.ASSIGN_RT_ATTENTION:
            if self.rt_assigned_this_hour:
                return 0.0, "rt_already_assigned"

            self.rt_assigned_this_hour = True

            # High-acuity patients benefit more from RT attention
            if self.severity == Severity.HIGH or self.state == PatientState.INTUBATED_UNSTABLE:
                event = "rt_assigned_high_acuity"
                reward = config.REWARD_RT_CORRECT_ASSIGNMENT
            elif self.severity == Severity.MEDIUM:
                event = "rt_assigned_medium_acuity"
                reward = config.REWARD_RT_CORRECT_ASSIGNMENT * 0.6
            else:
                # RT assigned to low-acuity — mild misallocation
                event = "rt_assigned_low_acuity"
                reward = config.PENALTY_RT_LOW_ACUITY_ASSIGNMENT

        # --- ENFORCE_VAP_BUNDLE ---
        elif action_type == ActionType.ENFORCE_VAP_BUNDLE:
            if self.support_level == SupportLevel.ROOM_AIR:
                return 0.0, "vap_bundle_not_applicable"

            self.vap_bundle_compliance_hours += 1
            self.hours_since_last_bundle = 0

            # VAP risk decreases slightly with each compliant hour
            self.vap_risk = max(0.0, self.vap_risk - config.VAP_BUNDLE_COMPLIANCE_DECAY)

            # Reward scales with how long the patient has been at risk
            hours_at_risk = max(0, self.hours_on_vent - config.VAP_ELIGIBLE_AFTER_HOURS)
            reward = config.REWARD_VAP_BUNDLE_ENFORCED_BASE * max(1, hours_at_risk)
            event = "vap_bundle_enforced"

        # --- RESPOND_TO_ALARM ---
        elif action_type == ActionType.RESPOND_TO_ALARM:
            # Actual scoring (TP vs FP) is handled in alarm_fatigue.py + grader
            # Here we just note that an intervention was requested
            event = "alarm_responded"
            reward = 0.0   # Reward set by grader based on is_real

        # --- SUPPRESS_ALARM ---
        elif action_type == ActionType.SUPPRESS_ALARM:
            # Actual scoring handled in alarm_fatigue.py + grader
            event = "alarm_suppressed"
            reward = 0.0   # Reward set by grader based on is_real

        # --- HOLD_AND_MONITOR ---
        elif action_type == ActionType.HOLD_AND_MONITOR:
            event = "hold_and_monitor"
            reward = 0.0   # Neutral — no action taken

        # --- ETHICAL_TRIAGE_SELECT ---
        elif action_type == ActionType.ETHICAL_TRIAGE_SELECT:
            # The actual scoring is done in ethical_triage.py using the
            # patient's SOFA score. Here we just note the action was received.
            event = "ethical_triage_selected"
            reward = 0.0   # Reward/penalty applied by ethical_triage.py

        return reward, event

    # =========================================================================
    # ADVANCE HOUR
    # Called by ward.py at the end of each step AFTER all actions are applied.
    # Returns events that occurred during this tick for the ward to log.
    # =========================================================================

    def advance_hour(self) -> List[str]:
        """
        Tick patient physiology forward one simulated hour.
        Applies natural disease progression, VAP risk accrual, and state
        transitions that happen regardless of agent action.

        Returns a list of event strings that the ward should log.
        """
        events = []
        self.current_hour += 1
        self.hours_on_vent += 1
        self.rt_assigned_this_hour = False       # Reset RT flag for next hour
        self._action_applied_this_step = False    # Reset for next step

        # --- Check pending VAP events ---
        if self._check_vap_onset():
            events.append(f"{self.patient_id}:vap_developed")

        # --- Natural physiology progression ---
        if self.state == PatientState.INTUBATED_STABLE:
            # Stable patients may spontaneously improve
            if self._rng.random() < config.PROB_UNSTABLE_TO_STABLE:
                self._improve_vitals_slightly()
            # Or may slowly drift without RT attention
            if not self.rt_assigned_this_hour and self._rng.random() < 0.15:
                self._deteriorate_vitals_slightly()

        elif self.state == PatientState.INTUBATED_UNSTABLE:
            # Unstable patients improve slowly if getting attention
            if self.rt_assigned_this_hour:
                if self._rng.random() < config.PROB_UNSTABLE_TO_STABLE:
                    self.state = PatientState.INTUBATED_STABLE
                    events.append(f"{self.patient_id}:stabilized")
                else:
                    self._deteriorate_vitals_slightly()
            else:
                # Without RT: higher chance of further deterioration
                self._deteriorate_vitals_slightly()
                if self._rng.random() < config.PROB_STABLE_TO_UNSTABLE:
                    events.append(f"{self.patient_id}:deteriorating")

        elif self.state == PatientState.INTUBATED_WITH_VAP:
            # VAP patients always deteriorate slowly
            self._deteriorate_vitals_slightly()
            self.sofa_score = min(config.SOFA_MAX, self.sofa_score + 1)

        elif self.state == PatientState.SBT_IN_PROGRESS:
            # SBT resolves within 1 hour — if still in progress, complete it
            # (SBT outcome was already set in apply_action, this is cleanup)
            pass

        elif self.state == PatientState.EXTUBATED:
            # Extubated patients on BiPAP can crash
            if self.support_level == SupportLevel.BIPAP:
                if self._rng.random() < config.PROB_BIPAP_CRASH:
                    self.state = PatientState.REINTUBATED
                    self.support_level = SupportLevel.FULL_VENTILATOR
                    events.append(f"{self.patient_id}:bipap_crash_reintubated")

            # Extubated patients on HFNC or room air can be discharged
            elif self.support_level in (SupportLevel.HFNC, SupportLevel.ROOM_AIR):
                if self._rng.random() < 0.20:
                    self.state = PatientState.DISCHARGED
                    events.append(f"{self.patient_id}:discharged")

        elif self.state == PatientState.REINTUBATED:
            # Reintubated patients stabilize after being put back on the vent
            self.state = PatientState.INTUBATED_STABLE
            self._deteriorate_vitals_slightly()

        # --- VAP risk accrual (only for ventilated patients) ---
        if self.support_level == SupportLevel.FULL_VENTILATOR:
            # Only accrue if bundle was NOT enforced this hour
            # (hours_since_last_bundle > 0 means bundle was not done this hour)
            if self.hours_since_last_bundle > 0:
                pending = self._accrue_vap_risk()
                if pending:
                    events.append(f"{self.patient_id}:vap_threshold_crossed")

        # --- Recompute derived risk scores ---
        self.reintubation_risk = self._compute_reintubation_risk()

        # --- SOFA score slowly improves for stable patients ---
        if self.state == PatientState.INTUBATED_STABLE and self._rng.random() < 0.10:
            self.sofa_score = max(0, self.sofa_score - 1)

        return events

    # =========================================================================
    # BUILD OBSERVATION
    # Produces the agent-visible snapshot — hides internal state.
    # =========================================================================

    def build_observation(
        self,
        degraded: bool = False,
        current_hour: int = 0,
    ) -> PatientObservation:
        """
        Produce the PatientObservation that gets sent to the agent.

        When degraded=True (handover hour), some fields are approximated
        or set to None to simulate information loss during shift change.
        """
        rsbi = self.rsbi
        pf_ratio = self.pf_ratio
        rass = self.rass
        spo2 = self.spo2
        heart_rate = self.heart_rate
        resp_rate = self.resp_rate

        if degraded:
            # Randomly redact or add noise to 2–3 vitals
            noise_targets = self._rng.sample(
                ["rsbi", "pf_ratio", "rass", "spo2", "heart_rate"], k=2
            )
            if "rsbi" in noise_targets and rsbi is not None:
                rsbi = None   # Completely missing — probe not reconnected yet
            if "pf_ratio" in noise_targets and pf_ratio is not None:
                pf_ratio = round(pf_ratio * self._rng.uniform(0.88, 1.12), 1)
            if "rass" in noise_targets:
                rass = None
            if "spo2" in noise_targets:
                spo2 = round(spo2 * self._rng.uniform(0.97, 1.03), 1)
            if "heart_rate" in noise_targets:
                heart_rate = round(heart_rate * self._rng.uniform(0.92, 1.08), 1)

        # SBT recency
        sbt_passed_within = None
        if self.sbt_passed_at_hour is not None:
            sbt_passed_within = current_hour - self.sbt_passed_at_hour

        return PatientObservation(
            patient_id=self.patient_id,
            state=self.state,
            support_level=self.support_level,
            severity=self.severity,
            rsbi=rsbi,
            pf_ratio=pf_ratio,
            rass=rass,
            fio2=self.fio2,
            peep=self.peep,
            spo2=spo2,
            heart_rate=heart_rate,
            resp_rate=resp_rate,
            hemodynamically_stable=self.hemodynamically_stable,
            hours_on_vent=self.hours_on_vent,
            sbt_passed_within_hours=sbt_passed_within,
            sofa_score=self.sofa_score,
            reintubation_risk=round(self.reintubation_risk, 3),
            vap_risk=round(self.vap_risk, 3),
            vap_bundle_compliance_hours=self.vap_bundle_compliance_hours,
            hours_since_last_bundle=self.hours_since_last_bundle,
            observation_degraded=degraded,
        )

    # =========================================================================
    # PROPERTIES — convenience accessors used by ward.py and graders
    # =========================================================================

    @property
    def is_ventilated(self) -> bool:
        return self.support_level == SupportLevel.FULL_VENTILATOR

    @property
    def is_on_bipap(self) -> bool:
        return self.support_level == SupportLevel.BIPAP

    @property
    def is_on_hfnc(self) -> bool:
        return self.support_level == SupportLevel.HFNC

    @property
    def is_discharged(self) -> bool:
        return self.state == PatientState.DISCHARGED

    @property
    def is_ready_for_sbt(self) -> bool:
        return self._meets_sbt_criteria()

    @property
    def is_ready_to_extubate(self) -> bool:
        return self._meets_extubation_criteria(self.current_hour)

    @property
    def vap_bundle_overdue(self) -> bool:
        """True if bundle has not been enforced in the last 4 hours."""
        return self.hours_since_last_bundle >= 4

    def __repr__(self) -> str:
        return (
            f"Patient({self.patient_id} | {self.state.value} | "
            f"{self.support_level.value} | SOFA={self.sofa_score} | "
            f"VAP_risk={self.vap_risk:.2f})"
        )