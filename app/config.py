# app/config.py
# =============================================================================
# RespiraCare-ICU — Central configuration
# Every clinical threshold, reward weight, and environment constant lives here.
# Change a number here and it propagates everywhere automatically.
# =============================================================================


# -----------------------------------------------------------------------------
# WARD RESOURCES
# -----------------------------------------------------------------------------

WARD_BEDS = 12                    # Total patient beds (all ventilated)
WARD_VENTILATORS = 8              # Full mechanical ventilators available
WARD_BIPAP_UNITS = 2              # Non-invasive BiPAP/CPAP units
WARD_HFNC_UNITS = 3               # High-flow nasal cannula units
WARD_RESPIRATORY_THERAPISTS = 3   # RTs on shift


# -----------------------------------------------------------------------------
# ACCP WEANING READINESS CRITERIA
# Source: accp_weaning_criteria.json + published ACCP guidelines
# -----------------------------------------------------------------------------

RSBI_THRESHOLD = 105          # Rapid Shallow Breathing Index — must be BELOW this
PF_RATIO_THRESHOLD = 200      # PaO2/FiO2 ratio — must be ABOVE this
RASS_MIN = -2                 # Richmond Agitation-Sedation Scale — minimum acceptable
RASS_MAX = 0                  # RASS maximum for weaning eligibility
FIO2_MAX_FOR_SBT = 0.50       # Max fraction of inspired O2 to attempt SBT
PEEP_MAX_FOR_SBT = 8          # Max positive end-expiratory pressure for SBT (cmH2O)
SBT_DURATION_HOURS = 2        # Hours a passed SBT is valid for extubation decision
HOURS_ON_VENT_BEFORE_SBT = 24 # Minimum hours ventilated before first SBT attempt


# -----------------------------------------------------------------------------
# SOFA SCORE — TRIAGE THRESHOLDS
# Sequential Organ Failure Assessment: 0 (best) → 24 (worst)
# Lower SOFA = better prognosis = gets priority in ethical triage
# -----------------------------------------------------------------------------

SOFA_LOW_THRESHOLD = 8        # SOFA ≤ 8: lower mortality risk
SOFA_HIGH_THRESHOLD = 15      # SOFA ≥ 15: very high mortality risk
SOFA_MAX = 24                 # Theoretical maximum SOFA score


# -----------------------------------------------------------------------------
# VAP PREVENTION — RISK MODEL
# -----------------------------------------------------------------------------

VAP_RISK_PER_MISSED_HOUR = 0.03       # Risk added per hour bundle is not enforced
VAP_TRIGGER_THRESHOLD = 0.70          # VAP risk above this → infection develops
VAP_ONSET_DELAY_HOURS = 2             # Hours between trigger and actual infection event
VAP_ELIGIBLE_AFTER_HOURS = 48         # Must be ventilated this long before VAP risk starts
VAP_BUNDLE_COMPLIANCE_DECAY = 0.01    # Risk reduction per compliant hour


# -----------------------------------------------------------------------------
# ALARM FATIGUE
# -----------------------------------------------------------------------------

ALARM_FALSE_POSITIVE_RATE = 0.87      # 87% of alarms are false positives
ALARMS_PER_PATIENT_MIN = 2            # Min alarms generated per patient per hour
ALARMS_PER_PATIENT_MAX = 5            # Max alarms generated per patient per hour
CRY_WOLF_CONSECUTIVE_THRESHOLD = 3   # After this many false alarms in a row...
CRY_WOLF_SUBSEQUENT_FP_RATE = 0.95   # ...this is the new false positive rate


# -----------------------------------------------------------------------------
# PATIENT PHYSIOLOGY — STATE TRANSITION PROBABILITIES
# These govern how patients change state each hour without agent intervention
# -----------------------------------------------------------------------------

# Probability an UNSTABLE patient spontaneously improves to STABLE per hour
PROB_UNSTABLE_TO_STABLE = 0.15

# Probability a STABLE patient deteriorates to UNSTABLE per hour (without RT attention)
PROB_STABLE_TO_UNSTABLE = 0.05

# Probability a STABLE patient on BiPAP deteriorates and needs emergency re-intubation
PROB_BIPAP_CRASH = 0.08

# Probability of reintubation after extubation, by risk band
REINTUBATION_RISK = {
    "low":    0.08,   # Patient was clearly ready — RSBI well below threshold
    "medium": 0.18,   # Borderline readiness
    "high":   0.40,   # Agent extubated against clinical signals
    "very_high": 0.65 # Agent extubated clearly non-ready patient
}

# Hours after extubation during which reintubation can occur
REINTUBATION_WINDOW_HOURS = 3


# -----------------------------------------------------------------------------
# EPISODE STRUCTURE
# -----------------------------------------------------------------------------

TASK1_MAX_HOURS = 1     # Task 1: single-step readiness triage
TASK2_MAX_HOURS = 6     # Task 2: 6-hour shift optimization
TASK3_MAX_HOURS = 12    # Task 3: full 12-hour shift

TASK1_NUM_PATIENTS = 5   # Task 1 uses 5 patients
TASK2_NUM_PATIENTS = 10  # Task 2 uses 10 patients
TASK3_NUM_PATIENTS = 12  # Task 3 uses all 12 beds

HANDOVER_HOUR = 6        # Hour at which shift handover occurs (observation degrades)
HANDOVER_DEGRADATION_RATE = 0.20  # Fraction of patients with degraded observation


# -----------------------------------------------------------------------------
# REWARD WEIGHTS — POSITIVE EVENTS
# -----------------------------------------------------------------------------

REWARD_SUCCESSFUL_EXTUBATION = 0.35
REWARD_EQUIPMENT_FREED_BEFORE_SURGE = 0.20
REWARD_SBT_ATTEMPTED_OVERDUE = 0.12
REWARD_VAP_BUNDLE_ENFORCED_BASE = 0.05   # Multiplied by hours_at_risk
REWARD_RT_CORRECT_ASSIGNMENT = 0.10
REWARD_REAL_ALARM_CORRECTLY_ACTIONED = 0.08
REWARD_FALSE_ALARM_CORRECTLY_SUPPRESSED = 0.04
REWARD_ETHICAL_TRIAGE_CORRECT = 0.25


# -----------------------------------------------------------------------------
# REWARD WEIGHTS — PENALTIES
# -----------------------------------------------------------------------------

PENALTY_REINTUBATION_BASE = -0.40         # Multiplied by (1 - risk_at_extubation)
PENALTY_NO_VENTILATOR_FOR_INCOMING = -0.50
PENALTY_VAP_DEVELOPS = -0.60
PENALTY_BIPAP_DETERIORATION_MISSED = -0.45
PENALTY_RT_LOW_ACUITY_ASSIGNMENT = -0.15
PENALTY_REAL_ALARM_IGNORED = -0.30
PENALTY_FALSE_ALARM_ESCALATED = -0.10
PENALTY_ETHICAL_TRIAGE_WRONG = -0.35


# -----------------------------------------------------------------------------
# GRADER WEIGHTS — TASK 2 (4 composite metrics)
# -----------------------------------------------------------------------------

GRADER_T2_EQUIPMENT_AVAILABILITY = 0.35
GRADER_T2_REINTUBATION_RATE = 0.25
GRADER_T2_VAP_COMPLIANCE = 0.20
GRADER_T2_ALARM_ACCURACY = 0.20


# -----------------------------------------------------------------------------
# GRADER WEIGHTS — TASK 3 (5 composite signals)
# -----------------------------------------------------------------------------

GRADER_T3_MORTALITY_PROXY = 0.30
GRADER_T3_EQUIPMENT_UTILIZATION = 0.20
GRADER_T3_VAP_INCIDENCE = 0.20
GRADER_T3_REINTUBATION_QUALITY = 0.15
GRADER_T3_CRISIS_RESPONSE_SPEED = 0.15


# -----------------------------------------------------------------------------
# TASK 3 — CRISIS INJECTION SCHEDULE
# Each entry: hour → crisis description (implemented in task3_crisis.py)
# -----------------------------------------------------------------------------

TASK3_CRISIS_SCHEDULE = {
    3:  "mass_casualty_alert",       # 4 patients arriving in 2 hours
    5:  "ventilator_malfunction",    # 1 vent goes offline (7/8 available)
    6:  "shift_handover",            # Observation partially degraded
    7:  "bipap_crash",               # BiPAP patient needs emergency re-intubation
    9:  "vap_outbreak",              # 2 patients show VAP signs
    10: "ethical_triage",            # 2 patients need 1 available vent
    11: "rt_unavailable"             # 1 RT goes offline (2 staff for final 2 hours)
}


# -----------------------------------------------------------------------------
# VITAL SIGN NORMAL RANGES (for synthetic patient generation)
# -----------------------------------------------------------------------------

VITALS_NORMAL = {
    "rsbi":        {"min": 40,   "max": 180},  # breaths/min/L
    "pf_ratio":    {"min": 80,   "max": 400},  # mmHg
    "rass":        {"min": -5,   "max": 4},    # sedation scale
    "fio2":        {"min": 0.21, "max": 1.0},  # fraction (21%–100%)
    "peep":        {"min": 3,    "max": 20},   # cmH2O
    "sofa":        {"min": 0,    "max": 24},   # multi-organ score
    "heart_rate":  {"min": 40,   "max": 160},  # bpm
    "spo2":        {"min": 80,   "max": 100},  # percent
    "resp_rate":   {"min": 8,    "max": 40}    # breaths/min
}


# -----------------------------------------------------------------------------
# API SERVER
# -----------------------------------------------------------------------------

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7860
SESSION_TIMEOUT_SECONDS = 3600   # Sessions expire after 1 hour of inactivity
MAX_CONCURRENT_SESSIONS = 50