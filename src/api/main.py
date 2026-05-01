"""
API REST para prediccion de reingreso hospitalario en pacientes diabeticos.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from src.api.ui import UI_HTML
from src.api.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    HealthResponse,
    PatientFeatures,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-readmission-xgboost")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.3"))
APP_VERSION = "1.1.0"
TRAIN_COLUMNS_PATH = Path(os.getenv("TRAIN_COLUMNS_PATH", "data/processed/X_train.csv"))

_model = None
_train_cols = None
_model_uri_loaded = None


def _get_risk_level(prob: float) -> str:
    if prob < 0.2:
        return "BAJO"
    if prob < 0.4:
        return "MEDIO"
    return "ALTO"


def _get_recommendation(risk_level: str) -> str:
    recommendations = {
        "BAJO": "Seguimiento ambulatorio estandar.",
        "MEDIO": "Seguimiento temprano y control de adherencia.",
        "ALTO": "Intervencion inmediata antes del alta.",
    }
    return recommendations[risk_level]


def _load_model_from_registry():
    """Carga el modelo de MLflow con fallback a la ultima version registrada."""
    global _model_uri_loaded
    mlflow.set_tracking_uri(MLFLOW_URI)

    candidate_uris = [
        f"models:/{MODEL_NAME}/{MODEL_STAGE}",
        f"models:/{MODEL_NAME}/latest",
    ]

    last_error = None
    for uri in candidate_uris:
        try:
            model = mlflow.pyfunc.load_model(uri)
            _model_uri_loaded = uri
            logger.info("Modelo cargado correctamente desde %s", uri)
            return model
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("No se pudo cargar %s: %s", uri, exc)

    raise RuntimeError(
        f"No se pudo cargar el modelo '{MODEL_NAME}'. Ultimo error: {last_error}"
    )


def _load_train_columns() -> pd.Index:
    """Carga columnas de entrenamiento compartidas por el volumen Docker."""
    if not TRAIN_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"No existe {TRAIN_COLUMNS_PATH}. El servicio trainer debe generar "
            "data/processed/X_train.csv antes de iniciar la API."
        )
    cols = pd.read_csv(TRAIN_COLUMNS_PATH, nrows=1).columns
    logger.info("Columnas de entrenamiento cargadas desde %s (%d columnas).", TRAIN_COLUMNS_PATH, len(cols))
    return cols


def _features_to_dataframe(features: PatientFeatures) -> pd.DataFrame:
    data = features.model_dump()

    insulin_str = str(data.get("insulin", "No")).strip().lower()
    data["insulin_changed"] = int(insulin_str != "no")

    a1c_str = str(data.get("A1Cresult", "None")).strip().lower()
    data["a1c_tested"] = int(a1c_str not in ["none", "nottested"])

    med_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 2}
    data["insulin"] = med_map.get(str(data.get("insulin", "No")), 0)

    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
    data["A1Cresult"] = a1c_map.get(str(data.get("A1Cresult", "None")), 0)

    gender_map = {"Male": 1, "Female": 0, "Unknown": 2}
    data["gender"] = gender_map.get(str(data.get("gender", "Unknown")), 2)

    change_map = {"No": 0, "Ch": 1}
    data["change"] = change_map.get(str(data.get("change", "No")), 0)

    med_bin_map = {"No": 0, "Yes": 1}
    data["diabetesMed"] = med_bin_map.get(str(data.get("diabetesMed", "Yes")), 1)

    data["total_prior_visits"] = (
        data.get("number_outpatient", 0)
        + data.get("number_emergency", 0)
        + data.get("number_inpatient", 0)
    )
    data["polypharmacy"] = int(data.get("num_medications", 0) >= 10)
    data["lab_procedures_per_day"] = data.get("num_lab_procedures", 0) / (
        data.get("time_in_hospital", 1) + 1
    )
    nd = data.get("number_diagnoses", 1)
    data["comorbidity_score"] = 0 if nd <= 3 else (1 if nd <= 6 else 2)
    data["had_emergency"] = int(data.get("number_emergency", 0) >= 1)

    df = pd.DataFrame([data])
    if _train_cols is None:
        raise RuntimeError("Columnas de entrenamiento no disponibles.")
    return df.reindex(columns=_train_cols, fill_value=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _train_cols

    try:
        _train_cols = _load_train_columns()
        _model = _load_model_from_registry()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error inicializando API: %s", exc)
        _model = None
        _train_cols = None

    yield

    _model = None
    _train_cols = None


app = FastAPI(
    title="API de Prediccion de Reingreso Hospitalario",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return {"service": "mlops-alto-costo", "status": "ok", "docs": "/docs", "ui": "/ui"}


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui():
    """Interfaz web interactiva para hacer predicciones sin escribir JSON."""
    return HTMLResponse(content=UI_HTML)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok" if _model is not None and _train_cols is not None else "degraded",
        version=APP_VERSION,
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Revisa logs del servicio trainer.")

    try:
        input_df = _features_to_dataframe(features)
        raw_pred = _model.predict(input_df)
        prob = float(raw_pred[0, 1]) if hasattr(raw_pred, "ndim") and raw_pred.ndim == 2 else float(raw_pred[0])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error en prediccion: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    risk_level = _get_risk_level(prob)
    return PredictionResponse(
        readmission_probability=round(prob, 4),
        high_risk=prob >= THRESHOLD,
        risk_level=risk_level,
        threshold_used=THRESHOLD,
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
        recommendation=_get_recommendation(risk_level),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(items: List[BatchPredictionItem]):
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Revisa logs del servicio trainer.")
    if len(items) > 200:
        raise HTTPException(status_code=422, detail="Maximo 200 pacientes.")

    try:
        batch_df = pd.concat([_features_to_dataframe(item.features) for item in items], ignore_index=True)
        raw_preds = _model.predict(batch_df)
        probs = raw_preds[:, 1].tolist() if hasattr(raw_preds, "ndim") and raw_preds.ndim == 2 else raw_preds.tolist()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error en prediccion batch: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    predictions = []
    for prob in probs:
        risk_level = _get_risk_level(float(prob))
        predictions.append(
            PredictionResponse(
                readmission_probability=round(float(prob), 4),
                high_risk=float(prob) >= THRESHOLD,
                risk_level=risk_level,
                threshold_used=THRESHOLD,
                model_name=MODEL_NAME,
                model_stage=MODEL_STAGE,
                recommendation=_get_recommendation(risk_level),
            )
        )

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=sum(p.high_risk for p in predictions),
    )
