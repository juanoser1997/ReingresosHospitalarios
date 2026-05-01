"""Tests unitarios para src/api/main.py"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, _get_risk_level, _get_recommendation


@pytest.fixture
def mock_model():
    """Modelo mockeado que retorna probabilidades bajas."""
    m = MagicMock()
    m.predict.return_value = np.array([[0.85, 0.15]])
    return m


@pytest.fixture
def client(mock_model):
    """TestClient con el modelo mockeado para no necesitar MLflow."""
    with patch("src.api.main._model", mock_model):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def valid_payload():
    return {
        "age": 72,
        "gender": "Female",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 5,
        "num_lab_procedures": 44,
        "num_procedures": 1,
        "num_medications": 14,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 1,
        "number_diagnoses": 9,
        "A1Cresult": ">8",
        "insulin": "Up",
        "diabetesMed": "Yes",
        "change": "Ch",
    }


# --- Tests de logica de negocio (sin HTTP) ---

def test_get_risk_level_bajo():
    assert _get_risk_level(0.10) == "BAJO"
    assert _get_risk_level(0.19) == "BAJO"


def test_get_risk_level_medio():
    assert _get_risk_level(0.20) == "MEDIO"
    assert _get_risk_level(0.39) == "MEDIO"


def test_get_risk_level_alto():
    assert _get_risk_level(0.40) == "ALTO"
    assert _get_risk_level(0.95) == "ALTO"


def test_get_recommendation_contains_action():
    rec_alto = _get_recommendation("ALTO")
    assert "INTERVENIR" in rec_alto or "evaluacion" in rec_alto.lower()

    rec_bajo = _get_recommendation("BAJO")
    assert "seguimiento" in rec_bajo.lower() or "control" in rec_bajo.lower()


# --- Tests de endpoints HTTP ---

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_model_loaded_true(client):
    data = response = client.get("/health").json()
    assert data["model_loaded"] is True
    assert data["status"] == "ok"


def test_predict_returns_200(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_predict_response_schema(client, valid_payload):
    data = client.post("/predict", json=valid_payload).json()
    assert "readmission_probability" in data
    assert "high_risk" in data
    assert "risk_level" in data
    assert "recommendation" in data
    assert data["risk_level"] in ("BAJO", "MEDIO", "ALTO")


def test_predict_probability_between_0_and_1(client, valid_payload):
    data = client.post("/predict", json=valid_payload).json()
    assert 0.0 <= data["readmission_probability"] <= 1.0


def test_predict_invalid_age_returns_422(client, valid_payload):
    valid_payload["age"] = 150   # fuera del rango permitido
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_invalid_gender_returns_422(client, valid_payload):
    valid_payload["gender"] = "Robot"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_batch_returns_list(client, valid_payload):
    payload = [{"patient_id": "P001", "features": valid_payload}]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert data["total"] == 1


def test_predict_batch_high_risk_count(client, valid_payload):
    """high_risk_count debe ser consistente con las predicciones individuales."""
    payload = [{"features": valid_payload}] * 3
    data = client.post("/predict/batch", json=payload).json()
    computed_high = sum(1 for p in data["predictions"] if p["high_risk"])
    assert data["high_risk_count"] == computed_high


def test_predict_batch_limit_exceeded(client, valid_payload):
    """Mas de 200 pacientes debe retornar 422."""
    payload = [{"features": valid_payload}] * 201
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422
