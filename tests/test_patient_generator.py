# tests/test_patient_generator.py
import pytest
from app.environment.patient_generator import generate_ward, generate_patient
from app.models import PatientState, SupportLevel, Severity
import random


def test_task1_generates_5_patients():
    patients = generate_ward(task_id=1, seed=42)
    assert len(patients) == 5


def test_task2_generates_10_patients():
    patients = generate_ward(task_id=2, seed=42)
    assert len(patients) == 10


def test_task3_generates_12_patients():
    patients = generate_ward(task_id=3, seed=42)
    assert len(patients) == 12


def test_reproducibility_same_seed():
    ward_a = generate_ward(task_id=3, seed=99)
    ward_b = generate_ward(task_id=3, seed=99)
    states_a = [p.state.value for p in ward_a]
    states_b = [p.state.value for p in ward_b]
    assert states_a == states_b


def test_diversity_different_seeds():
    ward_a = generate_ward(task_id=3, seed=1)
    ward_b = generate_ward(task_id=3, seed=2)
    sofa_a = [p.sofa_score for p in ward_a]
    sofa_b = [p.sofa_score for p in ward_b]
    assert sofa_a != sofa_b


def test_invalid_task_raises():
    with pytest.raises(ValueError):
        generate_ward(task_id=99, seed=42)


def test_patient_ids_are_unique():
    patients = generate_ward(task_id=3, seed=42)
    ids = [p.patient_id for p in patients]
    assert len(ids) == len(set(ids))


def test_generate_patient_has_valid_vitals():
    rng = random.Random(42)
    p = generate_patient(
        patient_id="P01",
        severity=Severity.MEDIUM,
        seed=42,
    )
    assert 0.0 <= p.fio2 <= 1.0
    assert 0 <= p.sofa_score <= 24
    assert 0.0 <= p.vap_risk <= 1.0
    assert p.hours_on_vent >= 0