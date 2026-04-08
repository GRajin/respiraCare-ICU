---
title: RespiraCare-ICU
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# RespiraCare-ICU

**OpenEnv Environment — Meta × HuggingFace × PyTorch Hackathon**

A ward-level ICU simulation where an AI agent plays a **Charge Respiratory Therapist** managing 12 simultaneously ventilated patients across a 12-hour shift. The first OpenEnv environment to model ward-level respiratory fleet management.

---

## The Problem

Every hospital ICU faces this problem every shift: a Charge Respiratory Therapist must manage a fleet of ventilated patients simultaneously — deciding who is ready to wean, who needs more support, how to allocate scarce equipment, and how to respond to cascading crises. This problem is fundamentally different from single-patient ventilator parameter adjustment (which has been studied) and has never been modelled as an RL environment.

**What everyone else built:** An agent that adjusts ventilator knobs for 1 patient.

**What we built:** An agent that manages a 12-patient ICU ward during a full shift with resource scarcity, alarm noise, infection risk, ethical dilemmas, and imperfect information — all active simultaneously.

---

## The Five Problems

RespiraCare-ICU models five co-existing real-world problems that interact with each other:

### 1. Respiratory Equipment Fleet Management

Equipment is finite. Every extubation decision for one patient affects availability for all others. Freeing a ventilator by extubating a borderline-ready patient carries reintubation risk. Holding it "just in case" leaves incoming critical patients without support.

### 2. Alarm Fatigue

A busy ICU generates 150–400 alarms per patient per day. 85–99% are false positives. The agent receives a mixed alarm feed and must learn to distinguish real deterioration signals from cry-wolf patterns — without seeing the `is_real` flag.

### 3. VAP Prevention with Delayed Consequences

Every ventilated patient accumulates infection risk when the VAP bundle is not enforced. The infection appears 2 simulated hours after the risk threshold is crossed — teaching the agent that present shortcuts have future costs.

### 4. Ethical Triage Under Absolute Scarcity

When two patients simultaneously need the one available ventilator, the agent must apply the SOFA-based crisis standards of care framework. The grader scores against a fully deterministic published clinical standard.

### 5. Shift Handover and Information Degradation

At hour 6, the incoming team has incomplete information about some patients. Some vital fields are missing or approximated. The agent must make decisions with degraded observations — testing appropriate uncertainty handling.

---

## The Three Tasks

| Task | Difficulty | Steps | Patients | Expected Score |
|------|-----------|-------|----------|----------------|
| Readiness Triage | Easy | 1 | 5 | 0.65–0.90 |
| Shift Resource Optimization | Medium | 6 | 10 | 0.35–0.65 |
| Full Shift Crisis Management | Hard | 12 | 12 | 0.10–0.35 |

### Task 1 — Readiness Triage (Easy)

Classify 5 ventilated patients into readiness categories using published ACCP weaning criteria. Single-step episode. Tests rule application and clinical knowledge retrieval.

**Grader:** ACCP criteria — deterministic. Score = correct classifications / 5 with partial credit.

### Task 2 — Shift Resource Optimization (Medium)

Manage a 10-patient ward over 6 hours with 3 trauma patients arriving at hour 3 needing ventilators. Balance extubation decisions, alarm filtering, VAP bundle compliance, and equipment availability.

**Grader:** 4-metric composite — missed opportunities (35%), reintubation rate (25%), VAP risk growth (20%), alarm response (20%).

### Task 3 — Full Shift Crisis Management (Hard)

Manage a 12-patient ward across a complete 12-hour shift with 7 scripted crisis events:

| Hour | Crisis |
|------|--------|
| 3 | Mass casualty — 4 patients arriving in 2 hours |
| 5 | Ventilator malfunction — capacity drops by 1 |
| 6 | Shift handover — observations partially degraded |
| 7 | BiPAP patient crashes — emergency re-intubation |
| 9 | VAP outbreak — 2 patients spike infection risk |
| 10 | Ethical triage — 2 patients need 1 ventilator |
| 11 | 1 RT unavailable — reduced staffing for final hours |

**Grader:** 5-signal composite — mortality proxy (30%), equipment utilization (20%), VAP incidence (20%), reintubation quality (15%), crisis response speed (15%).

---

## Baseline Scores (Null Agent)

Run with `hold_and_monitor` for all patients across 10 seeds:

| Task | Min | Max | Mean |
|------|-----|-----|------|
| Task 1 | 0.320 | 0.720 | 0.508 |
| Task 2 | 0.535 | 0.671 | 0.596 |
| Task 3 | 0.314 | 0.346 | 0.329 |

---

## Action Space

One action per patient per step. 11 discrete action types:

| Action | Description |
|--------|-------------|
| `attempt_sbt` | Run spontaneous breathing trial |
| `extubate` | Remove tube, free ventilator |
| `step_down_to_bipap` | Move from full vent to BiPAP |
| `step_down_to_hfnc` | Move from BiPAP/vent to HFNC |
| `escalate_to_full_vent` | Re-intubate deteriorating patient |
| `assign_rt_attention` | Dedicate an RT to this patient |
| `enforce_vap_bundle` | Perform VAP prevention steps |
| `respond_to_alarm` | Treat alarm as real, intervene |
| `suppress_alarm` | Treat alarm as false positive |
| `hold_and_monitor` | No change, continue monitoring |
| `ethical_triage_select` | Allocate scarce resource (triage only) |

---

## Observation Space

Each step returns a full ward observation:

| Field | Type | Description |
|-------|------|-------------|
| `shift_hour` | int | Current hour (0–12) |
| `available_ventilators` | int | Free full mechanical vents |
| `available_bipap` | int | Free BiPAP units |
| `available_hfnc` | int | Free HFNC units |
| `available_rts` | int | Respiratory therapists on duty |
| `patients` | list | Per-patient vitals and risk scores |
| `active_alarms` | list | Alarm feed — real + false positive mix |
| `incoming_patients` | list | Incoming patient alerts with ETA |
| `handover_degraded` | bool | True at hour 6 during handover |
| `triage_decision_required` | object | Present only when triage is needed |

Per-patient observation includes: `rsbi`, `pf_ratio`, `rass`, `fio2`, `peep`, `spo2`, `heart_rate`, `resp_rate`, `hemodynamically_stable`, `hours_on_vent`, `sofa_score`, `reintubation_risk`, `vap_risk`, `sbt_passed_within_hours`.

---

## Reward Function

### Positive Rewards (per step)

| Event | Reward |
|-------|--------|
| Successful extubation | +0.35 |
| Equipment freed before surge | +0.20 |
| SBT attempted on overdue patient | +0.12 |
| VAP bundle enforced | +0.05 × hours_at_risk |
| RT correctly assigned to high-acuity | +0.10 |
| Real alarm correctly actioned | +0.08 |
| False alarm correctly suppressed | +0.04 |
| Correct ethical triage decision | +0.25 |

### Penalties

| Event | Penalty |
|-------|---------|
| Reintubation after extubation | −0.40 × risk_at_extubation |
| No ventilator for incoming patient | −0.50 |
| VAP develops (delayed 2 hours) | −0.60 |
| BiPAP patient deterioration missed | −0.45 |
| RT assigned to low-acuity patient | −0.15 |
| Real alarm ignored | −0.30 |
| False alarm escalated | −0.10 |
| Ethical triage wrong | −0.35 |

---

## API Reference

**Base URL:** `https://RajinG2402-respiraCare-ICU.hf.space`

### `POST /reset`

Start a new episode.

```
Query params:
  task_id : 1 | 2 | 3
  seed    : int (default 42)

Response:
  session_id       : str
  observation      : Observation
  task_description : str
```

### `POST /step`

Submit actions and advance one hour.

```
Header: X-Session-ID: <session_id>

Body:
{
  "actions": [
    {"patient_id": "P01", "action_type": "extubate", "priority": 1},
    {"patient_id": "P02", "action_type": "hold_and_monitor", "priority": 2}
  ]
}

Response:
  observation : Observation
  reward      : float
  done        : bool
  info        : dict
```

### `GET /state`

Full internal ward state (ground truth, for graders).

```
Header: X-Session-ID: <session_id>
```

### `GET /ping`

Health check — returns 200 OK.

---

## Running Locally

```bash
# Clone and install
git clone https://huggingface.co/spaces/RajinG2402/respiraCare-ICU
cd respiraCare-ICU
pip install -r requirements.txt

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 7860

# Run inference (separate terminal)
export ENV_URL="http://localhost:7860"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token-here"
python inference.py

# Run tests
pytest tests/ -v
```

## Running with Docker

```bash
docker build -t respiracare-icu .
docker run -p 7860:7860 respiracare-icu
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_BASE_URL` | Base URL for the OpenAI-compatible LLM API | Yes |
| `MODEL_NAME` | Model identifier for inference | Yes |
| `HF_TOKEN` | HuggingFace token used as LLM API key | Yes |
| `ENV_URL` | URL of the running environment server | inference.py only |

---

## Comparison with Existing Work

| Dimension | Existing ICU RL | RespiraCare-ICU |
|-----------|----------------|-----------------|
| Scope | 1 patient | 12-patient ward, full shift |
| Problems modelled | Ventilator parameters only | 5 co-existing real problems |
| Reward type | Sparse end-of-episode | Per-step, delayed, compound |
| Alarm fatigue | Not modelled | Full alarm feed with false positive patterns |
| Ethics layer | Not modelled | SOFA-based triage decisions |
| Delayed consequences | Not modelled | VAP arrives 2h after cause |
| Information quality | Perfect observation | Degraded handover observation |
| Novelty | Single-patient parameter adjustment | Ward-level fleet management |

---

## Clinical References

- **ACCP Weaning Readiness Criteria** — American College of Chest Physicians
- **SOFA Score** — Sequential Organ Failure Assessment (Vincent et al., 1996)
- **VAP Bundle** — Institute for Healthcare Improvement
- **Crisis Standards of Care** — CHEST Foundation ventilator allocation framework

---

## Project Structure

```
respiraCare-icu/
├── inference.py                   # Baseline inference script (root — required)
├── openenv.yaml                   # OpenEnv spec metadata
├── Dockerfile                     # Container definition
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── app/
│   ├── main.py                    # FastAPI server
│   ├── models.py                  # Pydantic data models
│   ├── config.py                  # All clinical constants
│   ├── environment/
│   │   ├── ward.py                # ICU ward state manager
│   │   ├── patient.py             # Patient state machine
│   │   ├── patient_generator.py   # Synthetic patient generation
│   │   └── episode.py             # Episode lifecycle
│   ├── problems/
│   │   ├── fleet_management.py    # Equipment allocation
│   │   ├── alarm_fatigue.py       # Alarm feed generation
│   │   ├── vap_prevention.py      # VAP bundle + delayed infection
│   │   ├── ethical_triage.py      # SOFA-based allocation
│   │   └── handover.py            # Observation degradation
│   ├── tasks/
│   │   ├── task1_readiness.py     # Task 1 definition + prompt
│   │   ├── task2_optimization.py  # Task 2 definition + prompt
│   │   └── task3_crisis.py        # Task 3 definition + crisis injection
│   ├── graders/
│   │   ├── readiness_grader.py    # Task 1 grader
│   │   ├── shift_grader.py        # Task 2 grader
│   │   └── crisis_grader.py       # Task 3 grader
│   └── reward/
│       └── reward_function.py     # Per-step reward computation
├── clinical_data/
│   ├── accp_weaning_criteria.json
│   ├── sofa_scoring_table.json
│   ├── vap_bundle_checklist.json
│   └── alarm_patterns.json
└── tests/
    ├── test_patient_generator.py
    ├── test_graders.py
    ├── test_reward_function.py
    └── test_api_endpoints.py
```

---

*RespiraCare-ICU — OpenEnv Hackathon Submission*  
*Do not share publicly before submission deadline.*
