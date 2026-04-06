# app/main.py
# =============================================================================
# RespiraCare-ICU — FastAPI Server
#
# Exposes the OpenEnv standard interface over HTTP:
#   POST /reset          → start a new episode, get initial observation
#   POST /step           → submit actions, get new observation + reward
#   GET  /state          → get full internal ward state (ground truth)
#   GET  /ping           → health check (required by HuggingFace Space)
#
# Session management:
#   Every client gets a session_id from /reset.
#   All subsequent requests must include X-Session-ID header.
#   Sessions are stored in memory — one Episode object per session.
#   Sessions expire after SESSION_TIMEOUT_SECONDS of inactivity.
#
# Concurrency:
#   Multiple judges can run simultaneously — each gets its own session.
#   The session store is a plain dict protected by no lock (FastAPI is
#   async but our simulation is synchronous — no race conditions in practice
#   since each session is fully independent).
# =============================================================================

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.environment.episode import Episode
from app.models import Action, ActionType
from app import config


# =============================================================================
# SESSION STORE
# =============================================================================

# Global session store: session_id (str) → Episode
_sessions: Dict[str, Episode] = {}

# Track last access time for expiry: session_id → timestamp
_session_last_access: Dict[str, float] = {}


def _get_session(session_id: str) -> Episode:
    """
    Retrieve a session by ID. Raises 404 if not found, 410 if expired.
    Updates the last-access timestamp on every successful retrieval.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Session '{session_id}' not found. "
                "Call POST /reset to start a new episode."
            ),
        )

    # Check expiry
    last_access = _session_last_access.get(session_id, 0)
    if time.time() - last_access > config.SESSION_TIMEOUT_SECONDS:
        _sessions.pop(session_id, None)
        _session_last_access.pop(session_id, None)
        raise HTTPException(
            status_code=410,
            detail=(
                f"Session '{session_id}' has expired after "
                f"{config.SESSION_TIMEOUT_SECONDS}s of inactivity. "
                "Call POST /reset to start a new episode."
            ),
        )

    # Refresh access time
    _session_last_access[session_id] = time.time()
    return _sessions[session_id]


def _purge_expired_sessions() -> int:
    """
    Remove all sessions that have exceeded SESSION_TIMEOUT_SECONDS.
    Called automatically on each /reset to prevent unbounded memory growth.
    Returns the number of sessions purged.
    """
    now = time.time()
    expired = [
        sid for sid, t in _session_last_access.items()
        if now - t > config.SESSION_TIMEOUT_SECONDS
    ]
    for sid in expired:
        _sessions.pop(sid, None)
        _session_last_access.pop(sid, None)
    return len(expired)


def _enforce_session_limit() -> None:
    """
    If the session store is at capacity, reject new sessions.
    Prevents resource exhaustion during load testing.
    """
    if len(_sessions) >= config.MAX_CONCURRENT_SESSIONS:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Server at capacity ({config.MAX_CONCURRENT_SESSIONS} "
                "concurrent sessions). Try again later."
            ),
        )


# =============================================================================
# LIFESPAN — startup and shutdown
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Runs startup logic before the server accepts requests,
    and shutdown logic when the server stops.
    """
    # Startup
    _sessions.clear()
    _session_last_access.clear()
    print("RespiraCare-ICU environment server starting...")
    print(f"  Max concurrent sessions : {config.MAX_CONCURRENT_SESSIONS}")
    print(f"  Session timeout         : {config.SESSION_TIMEOUT_SECONDS}s")
    print(f"  Ward beds               : {config.WARD_BEDS}")
    print(f"  Ward ventilators        : {config.WARD_VENTILATORS}")
    print("Server ready.")

    yield  # Server runs here

    # Shutdown
    _sessions.clear()
    _session_last_access.clear()
    print("RespiraCare-ICU environment server stopped.")


# =============================================================================
# APP INSTANCE
# =============================================================================

app = FastAPI(
    title="RespiraCare-ICU",
    description=(
        "OpenEnv environment: ICU ward-level respiratory management simulation. "
        "An AI agent plays a Charge Respiratory Therapist managing 12 ventilated "
        "patients across a 12-hour shift."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class StepRequest(BaseModel):
    """
    Request body for POST /step.
    Contains the list of actions the agent is taking this step.
    Patients not mentioned receive HOLD_AND_MONITOR automatically.
    """
    actions: List[Action]

    class Config:
        json_schema_extra = {
            "example": {
                "actions": [
                    {
                        "patient_id": "P01",
                        "action_type": "attempt_sbt",
                        "priority": 1,
                        "ethical_triage_patient_id": None,
                    },
                    {
                        "patient_id": "P02",
                        "action_type": "hold_and_monitor",
                        "priority": 2,
                        "ethical_triage_patient_id": None,
                    },
                ]
            }
        }


class SessionInfo(BaseModel):
    """Minimal session info returned by utility endpoints."""
    session_id: str
    task_id: int
    current_hour: int
    is_done: bool
    total_reward: float
    step_count: int


# =============================================================================
# ENDPOINTS
# =============================================================================

# -----------------------------------------------------------------------------
# GET /ping — health check
# Required by HuggingFace Space to confirm the container is running.
# Must return HTTP 200. No authentication required.
# -----------------------------------------------------------------------------

@app.get("/ping", tags=["Health"])
def ping():
    """
    Health check endpoint. Returns 200 OK when the server is running.
    Called by HuggingFace Space deployment to verify the container started.
    """
    return {
        "status": "ok",
        "environment": "RespiraCare-ICU",
        "version": "1.0.0",
        "active_sessions": len(_sessions),
    }


# -----------------------------------------------------------------------------
# POST /reset — start a new episode
# Creates a new Episode, stores it in the session store, returns session_id
# and the initial Observation.
# -----------------------------------------------------------------------------

@app.post("/reset", tags=["Environment"])
def reset(
    task_id: int = Query(
        default=1,
        ge=1,
        le=3,
        description="Task ID: 1 (easy), 2 (medium), 3 (hard)",
    ),
    seed: int = Query(
        default=42,
        description="Random seed for reproducibility. Same seed → same episode.",
    ),
):
    """
    Start a new episode.

    Creates a fresh Episode for the given task and seed.
    Returns a session_id that must be included as X-Session-ID header
    in all subsequent /step and /state requests.

    The session_id is also the episode_id — it appears in every Observation.
    """
    # Purge expired sessions before creating a new one
    purged = _purge_expired_sessions()
    if purged > 0:
        print(f"Purged {purged} expired sessions.")

    # Enforce session limit
    _enforce_session_limit()

    # Create and initialise episode
    episode = Episode()
    reset_response = episode.reset(task_id=task_id, seed=seed)

    # Store in session map
    session_id = reset_response.session_id
    _sessions[session_id] = episode
    _session_last_access[session_id] = time.time()

    print(
        f"[{session_id}] New episode: task={task_id} seed={seed} "
        f"patients={len(reset_response.observation.patients)}"
    )

    # Return full ResetResponse as dict
    return {
        "session_id": session_id,
        "observation": reset_response.observation.model_dump(),
        "task_description": reset_response.task_description,
    }


# -----------------------------------------------------------------------------
# POST /step — submit actions, advance one hour
# Requires X-Session-ID header.
# Returns EpisodeResponse: observation + reward + done + info.
# -----------------------------------------------------------------------------

@app.post("/step", tags=["Environment"])
def step(
    request: StepRequest,
    x_session_id: str = Header(
        ...,
        description="Session ID returned by POST /reset",
    ),
):
    """
    Submit agent actions and advance the simulation one hour.

    Include the session_id (from /reset) as the X-Session-ID header.
    The request body contains a list of Action objects — one per patient.
    Patients not included receive HOLD_AND_MONITOR automatically.

    Returns the new observation, step reward, done flag, and info dict.
    Raises 409 if the episode is already complete.
    """
    episode = _get_session(x_session_id)

    if episode.is_done:
        raise HTTPException(
            status_code=409,
            detail=(
                "This episode is already complete (done=True). "
                "Call POST /reset to start a new episode."
            ),
        )

    # Validate action patient IDs exist in current observation
    current_obs = episode.ward.get_observation(episode.ward.current_hour)
    valid_patient_ids = {p.patient_id for p in current_obs.patients}

    for action in request.actions:
        if action.patient_id not in valid_patient_ids:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Action references unknown patient_id '{action.patient_id}'. "
                    f"Valid IDs for this session: {sorted(valid_patient_ids)}"
                ),
            )

    # Execute step
    response = episode.step(request.actions)

    print(
        f"[{x_session_id}] Step {episode.step_count}: "
        f"hour={response.observation.shift_hour} "
        f"reward={response.reward:.4f} "
        f"done={response.done}"
    )

    return {
        "observation": response.observation.model_dump(),
        "reward": response.reward,
        "done": response.done,
        "info": response.info,
    }


# -----------------------------------------------------------------------------
# GET /state — full internal ward state
# Requires X-Session-ID header.
# Returns ground-truth state including hidden fields (is_real alarms, etc.)
# Used by judges and the inference script to inspect the simulation.
# -----------------------------------------------------------------------------

@app.get("/state", tags=["Environment"])
def state(
    x_session_id: str = Header(
        ...,
        description="Session ID returned by POST /reset",
    ),
):
    """
    Return the full internal state of the ward.

    Unlike /step which returns only agent-visible observations, /state
    returns ground-truth data including alarm is_real flags, VAP risk
    values, and the full event log.

    Used by graders and the inference script to verify environment state.
    """
    episode = _get_session(x_session_id)
    return episode.get_state()


# -----------------------------------------------------------------------------
# GET /sessions — list active sessions (debug utility)
# No authentication required — useful during development.
# -----------------------------------------------------------------------------

@app.get("/sessions", tags=["Debug"])
def list_sessions():
    """
    List all active sessions. Development utility — not part of OpenEnv spec.
    """
    now = time.time()
    return {
        "active_sessions": len(_sessions),
        "max_sessions": config.MAX_CONCURRENT_SESSIONS,
        "sessions": [
            {
                "session_id": sid,
                "task_id": ep.task_id,
                "step_count": ep.step_count,
                "is_done": ep.is_done,
                "idle_seconds": round(now - _session_last_access.get(sid, now), 1),
            }
            for sid, ep in _sessions.items()
        ],
    }


# -----------------------------------------------------------------------------
# DELETE /session — manually close a session (debug utility)
# -----------------------------------------------------------------------------

@app.delete("/session", tags=["Debug"])
def delete_session(
    x_session_id: str = Header(...),
):
    """
    Manually delete a session. Development utility.
    """
    if x_session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    _sessions.pop(x_session_id, None)
    _session_last_access.pop(x_session_id, None)
    return {"deleted": x_session_id}