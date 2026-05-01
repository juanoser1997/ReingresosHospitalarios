"""
Promueve la version mas reciente del modelo al stage Production en MLflow.
Tambien asigna el alias 'champion' cuando la version de MLflow lo soporta.
"""

import logging
import os
import sys
import time

import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [promote_model] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-readmission-xgboost")
TARGET_STAGE = os.getenv("MODEL_STAGE", "Production")
MAX_RETRIES = 20
RETRY_DELAY = 3


def wait_for_mlflow() -> mlflow.tracking.MlflowClient:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
            client.search_experiments(max_results=1)
            logger.info("MLflow disponible en %s", MLFLOW_URI)
            return client
        except Exception as exc:  # noqa: BLE001
            logger.warning("Intento %d/%d: MLflow no disponible (%s)", attempt, MAX_RETRIES, exc)
            time.sleep(RETRY_DELAY)
    logger.error("MLflow no respondio tras %d intentos. Abortando.", MAX_RETRIES)
    sys.exit(1)


def get_latest_version(client: mlflow.tracking.MlflowClient):
    for attempt in range(1, MAX_RETRIES + 1):
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if versions:
            return max(versions, key=lambda v: int(v.version))
        logger.warning("Aun no aparece una version registrada de %s (%d/%d).", MODEL_NAME, attempt, MAX_RETRIES)
        time.sleep(RETRY_DELAY)
    logger.error("No se encontro ninguna version registrada de '%s'.", MODEL_NAME)
    sys.exit(1)


def promote_latest_version(client: mlflow.tracking.MlflowClient) -> None:
    latest = get_latest_version(client)
    logger.info("Version mas reciente de '%s': v%s", MODEL_NAME, latest.version)

    if latest.current_stage != TARGET_STAGE:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest.version,
            stage=TARGET_STAGE,
            archive_existing_versions=True,
        )
        logger.info("Modelo '%s' v%s promovido a %s.", MODEL_NAME, latest.version, TARGET_STAGE)
    else:
        logger.info("La version v%s ya esta en %s.", latest.version, TARGET_STAGE)

    try:
        client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)
        logger.info("Alias 'champion' asignado a v%s.", latest.version)
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo asignar alias champion (no critico): %s", exc)


if __name__ == "__main__":
    promote_latest_version(wait_for_mlflow())
    logger.info("Promocion completada. El modelo esta listo para la API.")
