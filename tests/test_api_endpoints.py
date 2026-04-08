# tests/test_api_endpoints.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["environment"] == "RespiraCare-ICU"


def test_reset_task1():
    response = client.post("/reset?task_id=1&seed=42")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "observation" in data
    assert "task_description" in data
    assert len(data["observation"]["patients"]) == 5


def test_reset_task2():
    response = client.post("/reset?task_id=2&seed=42")
    assert response.status_code == 200
    data = response.json()
    assert len(data["observation"]["patients"]) == 10


def test_reset_task3():
    response = client.post("/reset?task_id=3&seed=42")
    assert response.status_code == 200
    data = response.json()
    assert len(data["observation"]["patients"]) == 12


def test_reset_invalid_task():
    response = client.post("/reset?task_id=99&seed=42")
    assert response.status_code == 422


def test_step_requires_session_header():
    response = client.post(
        "/step",
        json={"actions": []},
    )
    assert response.status_code == 422


def test_step_invalid_session():
    response = client.post(
        "/step",
        json={"actions": []},
        headers={"X-Session-ID": "invalid-session-id"},
    )
    assert response.status_code == 404


def test_full_task1_episode():
    reset = client.post("/reset?task_id=1&seed=42").json()
    session_id = reset["session_id"]
    patients = reset["observation"]["patients"]

    actions = [
        {"patient_id": p["patient_id"], "action_type": "hold_and_monitor", "priority": 2}
        for p in patients
    ]

    response = client.post(
        "/step",
        json={"actions": actions},
        headers={"X-Session-ID": session_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert "reward" in data
    assert "done" in data
    assert "observation" in data
    assert data["done"] is True   # Task 1 ends after 1 step


def test_step_on_done_episode_returns_409():
    reset = client.post("/reset?task_id=1&seed=42").json()
    session_id = reset["session_id"]
    patients = reset["observation"]["patients"]
    actions = [
        {"patient_id": p["patient_id"], "action_type": "hold_and_monitor", "priority": 2}
        for p in patients
    ]
    # First step — completes episode
    client.post(
        "/step",
        json={"actions": actions},
        headers={"X-Session-ID": session_id},
    )
    # Second step — should fail
    response = client.post(
        "/step",
        json={"actions": actions},
        headers={"X-Session-ID": session_id},
    )
    assert response.status_code == 409


def test_state_endpoint():
    reset = client.post("/reset?task_id=2&seed=42").json()
    session_id = reset["session_id"]

    response = client.get(
        "/state",
        headers={"X-Session-ID": session_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert "current_hour" in data
    assert "fleet" in data
    assert "patients" in data


def test_sessions_endpoint():
    response = client.get("/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "active_sessions" in data


def test_reproducibility_same_seed():
    r1 = client.post("/reset?task_id=1&seed=77").json()
    r2 = client.post("/reset?task_id=1&seed=77").json()
    states1 = [p["state"] for p in r1["observation"]["patients"]]
    states2 = [p["state"] for p in r2["observation"]["patients"]]
    assert states1 == states2