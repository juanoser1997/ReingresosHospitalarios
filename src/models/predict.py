"""
Modulo de inferencia: carga el modelo desde MLflow Registry y genera predicciones.
"""

import logging
import os
from typing import Union

import mlflow.pyfunc
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-readmission-xgboost")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

_model_cache = None


def load_model(
    model_name: str = MODEL_NAME,
    stage: str = MODEL_STAGE,
):
    """
    Carga el modelo desde el MLflow Model Registry.
    Usa cache en memoria para evitar recargas en la API.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    mlflow.set_tracking_uri(MLFLOW_URI)
    errors = []
    for model_uri in [f"models:/{model_name}/{stage}", f"models:/{model_name}/latest"]:
        try:
            logger.info("Cargando modelo desde: %s", model_uri)
            _model_cache = mlflow.pyfunc.load_model(model_uri)
            return _model_cache
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{model_uri}: {exc}")
    raise RuntimeError("No se pudo cargar el modelo desde MLflow. " + " | ".join(errors))


def predict_proba(
    input_data: Union[pd.DataFrame, dict, list],
    threshold: float = 0.3,
) -> dict:
    """
    Genera predicciones de probabilidad de reingreso en menos de 30 dias.

    El umbral por defecto es 0.3 (no 0.5) para priorizar el recall en un
    contexto clinico donde los falsos negativos tienen mayor costo.

    Args:
        input_data: Features del paciente. Dict, lista de dicts o DataFrame.
        threshold: Umbral de decision para la clase positiva.

    Returns:
        Diccionario con probabilidad y prediccion binaria.
    """
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_data = pd.DataFrame(input_data)

    model = load_model()
    proba = model.predict(input_data)

    if hasattr(proba, "ndim") and proba.ndim == 2:
        proba = proba[:, 1]

    predictions = (proba >= threshold).astype(int)

    return {
        "probabilities": proba.tolist(),
        "predictions": predictions.tolist(),
        "threshold_used": threshold,
    }
