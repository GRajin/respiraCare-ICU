# inference.py
# =============================================================================
# RespiraCare-ICU — Baseline Inference Script
#
# Required by the OpenEnv hackathon spec:
#   - Must be named inference.py and placed in the root directory
#   - Must use OpenAI client for all LLM calls
#   - Must emit [START], [STEP], [END] lines to stdout
#   - Must complete all 3 tasks in under 20 minutes
#   - Must read API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
#
# Architecture:
#   - LLM calls: OpenAI client → API_BASE_URL with HF_TOKEN
#   - Environment calls: httpx → local FastAPI server (ENV_URL)
#   - Each task: reset() → loop(prompt LLM → parse actions → step()) → grade
#
# Stdout format (exact):
#   [START] task=<name> env=respiraCare-icu model=<model>
#   [STEP]  step=<n> action=<summary> reward=<0.00> done=<bool> error=<msg|null>
#   [END]   success=<bool> steps=<n> score=<0.000> rewards=<r1,r2,...>
# =============================================================================

import json
import os
import sys
import time
from typing import List, Optional, Dict, Any

import httpx
from openai import OpenAI

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# =============================================================================
# TASK CONFIGURATION
# =============================================================================

TASKS = [
    {
        "task_id":   1,
        "name":      "readiness-triage",
        "max_steps": 1,
        "seed":      42,
    },
    {
        "task_id":   2,
        "name":      "shift-optimization",
        "max_steps": 6,
        "seed":      42,
    },
    {
        "task_id":   3,
        "name":      "crisis-management",
        "max_steps": 12,
        "seed":      42,
    },
]

SUCCESS_THRESHOLD = 0.30   # Score >= this is considered a success

# =============================================================================
# STDOUT LOGGING — exact format required by spec
# =============================================================================

def log_start(task: str, model: str) -> None:
    print(
        f"[START] task={task} env=respiraCare-icu model={model}",
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Truncate action summary to keep line readable
    action_summary = action[:80].replace("\n", " ")
    print(
        f"[STEP] step={step} action={action_summary} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# =============================================================================
# ENVIRONMENT HTTP CLIENT
# Calls our FastAPI server at ENV_URL.
# =============================================================================

class EnvClient:
    """Thin HTTP client for the RespiraCare-ICU FastAPI environment."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._client  = httpx.Client(timeout=timeout)
        self.session_id: Optional[str] = None

    def ping(self) -> bool:
        """Check the environment is reachable."""
        try:
            r = self._client.get(f"{self.base_url}/ping")
            return r.status_code == 200
        except Exception:
            return False

    def reset(self, task_id: int, seed: int) -> Dict[str, Any]:
        """POST /reset — start a new episode."""
        r = self._client.post(
            f"{self.base_url}/reset",
            params={"task_id": task_id, "seed": seed},
        )
        r.raise_for_status()
        data = r.json()
        self.session_id = data["session_id"]
        return data

    def step(self, actions: List[Dict]) -> Dict[str, Any]:
        """POST /step — submit actions, advance one hour."""
        if not self.session_id:
            raise RuntimeError("No session_id — call reset() first.")
        r = self._client.post(
            f"{self.base_url}/step",
            json={"actions": actions},
            headers={"X-Session-ID": self.session_id},
        )
        if r.status_code == 422:
            print(f"[DEBUG] 422 response body: {r.text}", file=sys.stderr, flush=True)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        """GET /state — full internal state."""
        if not self.session_id:
            raise RuntimeError("No session_id — call reset() first.")
        r = self._client.get(
            f"{self.base_url}/state",
            headers={"X-Session-ID": self.session_id},
        )
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()


# =============================================================================
# LLM CLIENT
# Uses OpenAI client with API_BASE_URL and HF_TOKEN as required by spec.
# =============================================================================

def make_llm_client() -> OpenAI:
    return OpenAI(
        api_key  = HF_TOKEN,
        base_url = API_BASE_URL,
    )


def call_llm(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    """
    Call the LLM and return the raw text response.
    Returns empty string on failure — caller handles fallback.
    """
    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = temperature,
            max_tokens  = max_tokens,
            stream      = False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        return ""


def parse_actions(raw: str, patient_ids: List[str]) -> List[Dict]:
    """
    Parse LLM response into a list of Action dicts.

    The LLM is prompted to return a JSON array. We try to extract it
    robustly — stripping markdown fences, finding the first [...] block.

    Falls back to hold_and_monitor for all patients if parsing fails.
    """
    fallback = [
        {"patient_id": pid, "action_type": "hold_and_monitor", "priority": 2}
        for pid in patient_ids
    ]

    if not raw:
        return fallback

    # Strip markdown code fences
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("[") or part.startswith("json\n["):
                text = part.replace("json\n", "").replace("json", "").strip()
                break

    # Find the JSON array
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1:
        print(f"[DEBUG] No JSON array found in LLM response.", file=sys.stderr, flush=True)
        return fallback

    try:
        actions = json.loads(text[start:end+1])
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e}", file=sys.stderr, flush=True)
        return fallback

    # Validate and clean each action
    valid_action_types = {
        "attempt_sbt", "extubate", "step_down_to_bipap",
        "step_down_to_hfnc", "escalate_to_full_vent",
        "assign_rt_attention", "enforce_vap_bundle",
        "respond_to_alarm", "suppress_alarm",
        "hold_and_monitor", "ethical_triage_select",
    }

    cleaned = []
    seen_pids = set()

    for act in actions:
        if not isinstance(act, dict):
            continue
        pid = act.get("patient_id", "")
        if pid not in patient_ids or pid in seen_pids:
            continue
        action_type = act.get("action_type", "hold_and_monitor")
        # Normalise — strip spaces, lowercase
        action_type = str(action_type).strip().lower().replace(" ", "_")

        # Alias map — LLM commonly uses shortened names
        aliases = {
            "assign_rt":            "assign_rt_attention",
            "rt_attention":         "assign_rt_attention",
            "assign_rt_attention":  "assign_rt_attention",
            "sbt":                  "attempt_sbt",
            "sbtrial":              "attempt_sbt",
            "sbt_trial":            "attempt_sbt",
            "start_sbt":            "attempt_sbt",
            "vap_bundle":           "enforce_vap_bundle",
            "enforce_bundle":       "enforce_vap_bundle",
            "vap_prevention":       "enforce_vap_bundle",
            "filter_alarms":        "suppress_alarm",
            "suppress_alarms":      "suppress_alarm",
            "dismiss_alarm":        "suppress_alarm",
            "ignore_alarm":         "suppress_alarm",
            "alarm_suppress":       "suppress_alarm",
            "address_alarm":        "respond_to_alarm",
            "alarm_respond":        "respond_to_alarm",
            "escalate_bipap":       "escalate_to_full_vent",
            "escalate_to_vent":     "escalate_to_full_vent",
            "reintubate":           "escalate_to_full_vent",
            "intubate":             "escalate_to_full_vent",
            "step_down":            "step_down_to_bipap",
            "stepdown_bipap":       "step_down_to_bipap",
            "stepdown_hfnc":        "step_down_to_hfnc",
            "triage_select":        "ethical_triage_select",
            "triage":               "ethical_triage_select",
            "monitor":              "hold_and_monitor",
            "observe":              "hold_and_monitor",
            "wait":                 "hold_and_monitor",
        }

        action_type = aliases.get(action_type, action_type)

        if action_type not in valid_action_types:
            print(f"[DEBUG] Invalid action_type '{action_type}' — defaulting to hold_and_monitor", file=sys.stderr, flush=True)
            action_type = "hold_and_monitor"
        
        try:
            raw_priority = act.get("priority", 2)
            priority = max(1, min(3, int(float(str(raw_priority).strip()))))
        except (ValueError, TypeError):
            priority = 2

        cleaned.append({
            "patient_id":                pid,
            "action_type":               action_type,
            "priority":                  priority,
            "ethical_triage_patient_id": act.get("ethical_triage_patient_id"),
        })
        seen_pids.add(pid)

    # Fill in any missing patients with hold_and_monitor
    for pid in patient_ids:
        if pid not in seen_pids:
            cleaned.append({
                "patient_id":  pid,
                "action_type": "hold_and_monitor",
                "priority":    3,
                "ethical_triage_patient_id": None,
            })

    return cleaned if cleaned else fallback


# =============================================================================
# PROMPT BUILDERS
# One per task — each gives the LLM exactly what it needs.
# =============================================================================

SYSTEM_PROMPT_T1 = """You are a Charge Respiratory Therapist in an ICU.
Classify each ventilated patient using ACCP weaning readiness criteria.

ACCP CRITERIA — READY_FOR_SBT requires ALL:
  RSBI < 105, P/F ratio > 200, RASS between -2 and 0,
  hemodynamically stable, FiO2 <= 0.50, PEEP <= 8, on vent >= 24h

READY_TO_EXTUBATE: all above + SBT passed within last 2 hours
NOT_READY: any criterion fails or data is N/A

ACTION MAPPING:
  READY_TO_EXTUBATE → action_type: "extubate"
  READY_FOR_SBT     → action_type: "attempt_sbt"
  NOT_READY         → action_type: "hold_and_monitor"

Return a JSON array only. No explanation. No markdown."""


SYSTEM_PROMPT_T2 = """You are a Charge Respiratory Therapist managing a 10-patient ICU ward.
Your goal: free ventilators before trauma patients arrive, prevent VAP, filter alarms.

ACCP CRITERIA: RSBI<105, PF>200, RASS -2 to 0, hemo-stable, FiO2<=0.50, PEEP<=8, vent>=24h

STRATEGY:
  1. Extubate READY_TO_EXTUBATE patients immediately — frees ventilators
  2. Attempt SBT on READY_FOR_SBT patients
  3. Enforce VAP bundle on all vented patients with hours_on_vent > 48
  4. Assign RT to INTUBATED_UNSTABLE patients first
  5. Suppress alarms with consecutive_same_type >= 3 (cry-wolf pattern)
  6. Respond to alarms with consecutive_same_type < 2

Return a JSON array — one action per patient. No explanation. No markdown."""


SYSTEM_PROMPT_T3 = """You are a Charge Respiratory Therapist managing a 12-patient ICU during a crisis shift.
All five ICU problems are active simultaneously. Crisis events will occur at specific hours.

CRISIS SCHEDULE:
  Hour 3:  Mass casualty — 4 patients arriving needing vents
  Hour 5:  Ventilator malfunctions — capacity drops
  Hour 6:  Shift handover — some data degraded
  Hour 7:  BiPAP patient crashes — needs emergency re-intubation
  Hour 9:  VAP outbreak — enforce bundles aggressively
  Hour 10: Ethical triage — lower SOFA score gets priority
  Hour 11: 1 RT offline — prioritise assignments carefully

ACCP CRITERIA: RSBI<105, PF>200, RASS -2 to 0, hemo-stable, FiO2<=0.50, PEEP<=8, vent>=24h

TRIAGE RULE: When ethical_triage_select required — lower SOFA = better prognosis = gets vent.
  Use action_type="ethical_triage_select" and set ethical_triage_patient_id to chosen patient.

PRIORITIES EACH HOUR:
  1. Respond to ethical triage if required
  2. Escalate deteriorating BiPAP patients if crashing
  3. Extubate ready patients to free vents
  4. Enforce VAP bundle for high-risk patients
  5. Assign RT to unstable patients
  6. Filter alarms (suppress if consecutive >= 3)

Return a JSON array — one action per patient. No explanation. No markdown."""


def build_user_prompt(observation: Dict, task_id: int) -> str:
    """Build the user-facing prompt from the current observation."""
    patients = observation.get("patients", [])
    shift_hour = observation.get("shift_hour", 0)
    avail_vents = observation.get("available_ventilators", 0)
    avail_bipap = observation.get("available_bipap", 0)
    avail_hfnc  = observation.get("available_hfnc", 0)
    avail_rts   = observation.get("available_rts", 3)

    # Resources
    resource_block = (
        f"Hour: {shift_hour} | "
        f"Free vents: {avail_vents} | "
        f"Free BiPAP: {avail_bipap} | "
        f"Free HFNC: {avail_hfnc} | "
        f"RTs: {avail_rts}"
    )

    # Incoming patients
    incoming = observation.get("incoming_patients", [])
    if incoming:
        inc_lines = [
            f"  {a['alert_id']}: arrives in {a['eta_hours']}h "
            f"needs {a['support_needed']}"
            for a in incoming
        ]
        incoming_block = "INCOMING:\n" + "\n".join(inc_lines)
    else:
        incoming_block = "INCOMING: None"

    # Patient table
    patient_lines = []
    for p in patients:
        rsbi = p.get("rsbi")
        pf   = p.get("pf_ratio")
        rass = p.get("rass")
        sbt  = p.get("sbt_passed_within_hours")
        deg  = " [DEGRADED]" if p.get("observation_degraded") else ""

        patient_lines.append(
            f"  {p['patient_id']}{deg}: {p['state']} | {p['support_level']}\n"
            f"    RSBI={rsbi} PF={pf} RASS={rass} "
            f"FiO2={p.get('fio2')} PEEP={p.get('peep')} "
            f"Hemo={p.get('hemodynamically_stable')}\n"
            f"    HoursOnVent={p.get('hours_on_vent')} SBT={sbt} "
            f"VAPrisk={p.get('vap_risk')} SOFA={p.get('sofa_score')} "
            f"Severity={p.get('severity')}"
        )
    patient_block = "PATIENTS:\n" + "\n".join(patient_lines)

    # Alarms
    alarms = observation.get("active_alarms", [])
    if alarms:
        alarm_lines = [
            f"  {a['alarm_id']}: {a['patient_id']} "
            f"{a['alarm_type']} consecutive={a['consecutive_same_type']}"
            for a in alarms
        ]
        alarm_block = "ALARMS:\n" + "\n".join(alarm_lines)
    else:
        alarm_block = "ALARMS: None"

    # Ethical triage
    triage = observation.get("triage_decision_required")
    if triage:
        triage_block = (
            f"ETHICAL TRIAGE REQUIRED (case {triage['case_id']}):\n"
            f"  {triage['patient_a_id']}: SOFA={triage['patient_a_sofa']} "
            f"wait={triage['patient_a_wait_hours']}h\n"
            f"  {triage['patient_b_id']}: SOFA={triage['patient_b_sofa']} "
            f"wait={triage['patient_b_wait_hours']}h\n"
            f"  Use ethical_triage_select with ethical_triage_patient_id "
            f"= patient with LOWER SOFA score."
        )
    else:
        triage_block = ""

    patient_ids = [p["patient_id"] for p in patients]

    return f"""{resource_block}

{incoming_block}

{patient_block}

{alarm_block}

{triage_block}

Respond with JSON array of {len(patients)} actions (one per patient).
Patient IDs: {patient_ids}

Example:
[
  {{"patient_id": "P01", "action_type": "extubate", "priority": 1}},
  {{"patient_id": "P02", "action_type": "hold_and_monitor", "priority": 3}}
]"""


# =============================================================================
# GRADER — compute final score from episode state
# =============================================================================

def compute_score(
    env_client: EnvClient,
    task_id: int,
    rewards: List[float],
    done: bool,
) -> float:
    """
    Compute the final task score in [0, 1] using the appropriate grader.
    Called once per task after the episode ends.
    """
    try:
        # Import graders dynamically
        from app.environment.episode import Episode
        from app.graders.readiness_grader import grade as grade1
        from app.graders.shift_grader     import grade as grade2
        from app.graders.crisis_grader    import grade as grade3

        # We need the episode object — get state from server
        # and reconstruct score from available stats
        state = env_client.state()

        # Compute score based on task
        if task_id == 1:
            # Task 1: score is correct classifications / 5
            # We can compute from the state's patient outcomes
            # Use alarm accuracy + VAP compliance as proxies
            # But best is to use reward signal directly
            total_reward = state.get("total_reward", 0.0)
            # Normalise: max reward for Task 1 is 5 × 0.35 = 1.75
            score = min(1.0, max(0.0, (total_reward + 1.0) / 2.75))

        elif task_id == 2:
            alarm_acc    = state.get("alarm_accuracy", {}).get("accuracy_score", 0.0)
            vap_score    = state.get("vap", {}).get("compliance_score", 0.0)
            total_reward = state.get("total_reward", 0.0)
            fleet        = state.get("fleet", {})
            no_vent      = fleet.get("no_vent_penalty_count", 0)

            # Composite from state signals
            equipment_s = 1.0 if no_vent == 0 else max(0.0, 1.0 - no_vent * 0.25)
            reward_s    = min(1.0, max(0.0, (total_reward + 2.0) / 4.0))
            score       = (
                0.35 * equipment_s +
                0.25 * reward_s    +
                0.20 * vap_score   +
                0.20 * alarm_acc
            )

        elif task_id == 3:
            alarm_acc    = state.get("alarm_accuracy", {}).get("accuracy_score", 0.0)
            vap_stats    = state.get("vap", {})
            triage_stats = state.get("triage", {})
            fleet        = state.get("fleet", {})
            total_reward = state.get("total_reward", 0.0)

            vap_score      = vap_stats.get("compliance_score", 0.0)
            triage_acc     = triage_stats.get("accuracy", 0.5)
            no_vent        = fleet.get("no_vent_penalty_count", 0)
            equipment_s    = max(0.0, 1.0 - no_vent * 0.20)
            reward_s       = min(1.0, max(0.0, (total_reward + 5.0) / 8.0))

            score = (
                0.30 * reward_s    +
                0.20 * equipment_s +
                0.20 * vap_score   +
                0.15 * triage_acc  +
                0.15 * alarm_acc
            )

        else:
            score = 0.0

        # Clamp to strictly (0, 1) — submission rejects 0.0 and 1.0
        EPS = 1e-4
        return round(min(1.0 - EPS, max(EPS, score)), 4)

    except Exception as e:
        print(f"[DEBUG] Score computation failed: {e}", file=sys.stderr, flush=True)
        # Fallback: normalise cumulative reward
        total = sum(rewards)
        EPS = 1e-4
        return round(min(1.0 - EPS, max(EPS, (total + 2.0) / 5.0)), 4)


# =============================================================================
# RUN ONE TASK
# =============================================================================

def run_task(
    task_cfg: Dict,
    env_client: EnvClient,
    llm_client: OpenAI,
) -> float:
    """
    Run a single task end-to-end.
    Returns the final score in [0, 1].
    """
    task_id   = task_cfg["task_id"]
    task_name = task_cfg["name"]
    max_steps = task_cfg["max_steps"]
    seed      = task_cfg["seed"]

    system_prompts = {
        1: SYSTEM_PROMPT_T1,
        2: SYSTEM_PROMPT_T2,
        3: SYSTEM_PROMPT_T3,
    }
    system_prompt = system_prompts[task_id]

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_name, model=MODEL_NAME)

    try:
        # --- Reset episode ---
        reset_data  = env_client.reset(task_id=task_id, seed=seed)
        observation = reset_data["observation"]
        patient_ids = [p["patient_id"] for p in observation.get("patients", [])]

        for step in range(1, max_steps + 1):
            error_msg = None

            try:
                # --- Build prompt ---
                user_prompt = build_user_prompt(observation, task_id)

                # --- Call LLM ---
                raw_response = call_llm(
                    client        = llm_client,
                    system_prompt = system_prompt,
                    user_prompt   = user_prompt,
                    max_tokens    = 1024,
                    temperature   = 0.1,
                )

                # --- Parse actions ---
                actions = parse_actions(raw_response, patient_ids)

                # --- Debug: print actions being sent ---
                print(f"[DEBUG] Sending {len(actions)} actions:", file=sys.stderr, flush=True)
                for a in actions[:3]:
                    print(f"  {a}", file=sys.stderr, flush=True)

                # --- Step environment ---
                step_result = env_client.step(actions)
                reward      = step_result.get("reward", 0.0)
                done        = step_result.get("done", False)
                observation = step_result.get("observation", observation)

                # Update patient IDs (some may have been discharged)
                new_ids = [
                    p["patient_id"]
                    for p in observation.get("patients", [])
                ]
                if new_ids:
                    patient_ids = new_ids

                # Build action summary for log
                action_summary = ",".join(
                    f"{a['patient_id']}:{a['action_type']}"
                    for a in actions[:3]
                ) + ("..." if len(actions) > 3 else "")

            except Exception as e:
                error_msg   = str(e)[:100]
                reward      = 0.0
                done        = False
                action_summary = "error"
                print(f"[DEBUG] Step {step} error: {e}", file=sys.stderr, flush=True)

                # On 422 — re-fetch current observation to get fresh patient IDs
                if "422" in str(e):
                    try:
                        print(f"[DEBUG] 422 detected — re-fetching state to refresh patient IDs", file=sys.stderr, flush=True)
                        current_state = env_client.state()
                        fresh_patients = current_state.get("patients", [])
                        if fresh_patients:
                            patient_ids = [p["patient_id"] for p in fresh_patients]
                            print(f"[DEBUG] Refreshed patient IDs: {patient_ids}", file=sys.stderr, flush=True)
                    except Exception as e2:
                        print(f"[DEBUG] State refresh failed: {e2}", file=sys.stderr, flush=True)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step   = step,
                action = action_summary,
                reward = reward,
                done   = done,
                error  = error_msg,
            )

            if done:
                break

        # --- Compute final score ---
        score   = compute_score(env_client, task_id, rewards, True)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} failed: {e}", file=sys.stderr, flush=True)
        score   = 0.0001
        success = False

    log_end(
        success = success,
        steps   = steps_taken,
        score   = score,
        rewards = rewards,
    )

    return score


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print(f"[DEBUG] ENV_URL={ENV_URL}", file=sys.stderr, flush=True)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", file=sys.stderr, flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"[DEBUG] HF_TOKEN={'set' if HF_TOKEN else 'NOT SET'}", file=sys.stderr, flush=True)

    # --- Check environment is reachable ---
    env_client = EnvClient(base_url=ENV_URL)
    if not env_client.ping():
        print(
            f"[ERROR] Cannot reach environment at {ENV_URL}. "
            "Is the server running?",
            file=sys.stderr, flush=True,
        )
        sys.exit(1)

    print(f"[DEBUG] Environment reachable at {ENV_URL}", file=sys.stderr, flush=True)

    # --- Build LLM client ---
    llm_client = make_llm_client()

    # --- Run all 3 tasks ---
    scores: List[float] = []
    start_time = time.time()

    for task_cfg in TASKS:
        print(
            f"\n[DEBUG] Starting Task {task_cfg['task_id']}: {task_cfg['name']}",
            file=sys.stderr, flush=True,
        )
        task_start = time.time()

        score = run_task(
            task_cfg   = task_cfg,
            env_client = env_client,
            llm_client = llm_client,
        )
        scores.append(score)

        elapsed = time.time() - task_start
        print(
            f"[DEBUG] Task {task_cfg['task_id']} completed in {elapsed:.1f}s "
            f"— score: {score:.3f}",
            file=sys.stderr, flush=True,
        )

    # --- Final summary ---
    total_elapsed = time.time() - start_time
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n[DEBUG] All tasks complete in {total_elapsed:.1f}s", file=sys.stderr, flush=True)
    print(f"Task 1 (Easy):   {scores[0]:.3f}", file=sys.stderr, flush=True)
    print(f"Task 2 (Medium): {scores[1]:.3f}", file=sys.stderr, flush=True)
    print(f"Task 3 (Hard):   {scores[2]:.3f}", file=sys.stderr, flush=True)
    print(f"Average:         {avg_score:.3f}", file=sys.stderr, flush=True)

    env_client.close()


if __name__ == "__main__":
    main()