"""
Orquestador robusto del pipeline de entrenamiento.

Correcciones clave:
1. Espera MLflow con retries via HTTP (consistente con wait_for_mlflow.py).
2. Garantiza SIEMPRE que existan data/raw/diabetic_data.csv y data/processed/X_train.csv.
3. Solo salta el entrenamiento si ya existe modelo en Production Y los datos procesados existen.
4. Si faltan datos procesados, ejecuta preprocesamiento aunque el modelo ya exista.
"""

import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [trainer] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-readmission-xgboost")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

RAW_DATA = Path("data/raw/diabetic_data.csv")
X_TRAIN = Path("data/processed/X_train.csv")
X_TEST = Path("data/processed/X_test.csv")
Y_TRAIN = Path("data/processed/y_train.csv")
Y_TEST = Path("data/processed/y_test.csv")

MLFLOW_MAX_RETRIES = int(os.getenv("MLFLOW_MAX_RETRIES", "30"))
MLFLOW_RETRY_DELAY = int(os.getenv("MLFLOW_RETRY_DELAY", "5"))


def run(cmd: list[str], step_name: str) -> None:
    logger.info("=== Ejecutando paso: %s ===", step_name)
    logger.info("Comando: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Paso '%s' fallo con codigo %d.", step_name, result.returncode)
        sys.exit(result.returncode)
    logger.info("Paso '%s' completado exitosamente.", step_name)


def wait_for_mlflow() -> mlflow.tracking.MlflowClient:
    """
    Espera a MLflow usando HTTP directo (consistente con wait_for_mlflow.py).
    Evita usar MlflowClient.search_experiments() para el chequeo inicial
    porque ese endpoint REST puede tardar mas en inicializarse que la UI HTTP.
    """
    http_uri = MLFLOW_URI.rstrip("/")
    is_http = http_uri.startswith("http://") or http_uri.startswith("https://")

    logger.info("Conectando a MLflow en %s ...", http_uri)

    if not is_http:
        # URI local (sqlite://), conectar directamente sin HTTP check.
        mlflow.set_tracking_uri(http_uri)
        logger.info("MLflow URI local detectada, conectando directamente.")
        return mlflow.tracking.MlflowClient(tracking_uri=http_uri)

    for attempt in range(1, MLFLOW_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                f"{http_uri}/",
                method="GET",
                headers={"User-Agent": "trainer-wait-mlflow"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 400:
                    logger.info(
                        "MLflow HTTP disponible (intento %d/%d, status %d).",
                        attempt, MLFLOW_MAX_RETRIES, resp.status,
                    )
                    # Pausa de gracia: la UI responde antes que el API REST de tracking.
                    time.sleep(3)
                    mlflow.set_tracking_uri(http_uri)
                    return mlflow.tracking.MlflowClient(tracking_uri=http_uri)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
            logger.warning(
                "Intento %d/%d: MLflow no responde aun (%s). Esperando %ds...",
                attempt, MLFLOW_MAX_RETRIES, exc, MLFLOW_RETRY_DELAY,
            )
        time.sleep(MLFLOW_RETRY_DELAY)

    logger.error("MLflow no respondio. Abortando.")
    sys.exit(1)


def model_in_stage(client: mlflow.tracking.MlflowClient) -> bool:
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        for version in versions:
            if version.current_stage == MODEL_STAGE:
                logger.info(
                    "Modelo '%s' version %s encontrado en stage '%s'.",
                    MODEL_NAME,
                    version.version,
                    MODEL_STAGE,
                )
                return True
    except Exception as exc:
        logger.warning("No se pudo consultar Model Registry: %s", exc)
    return False


def processed_data_exists() -> bool:
    required = [X_TRAIN, X_TEST, Y_TRAIN, Y_TEST]
    missing = [str(p) for p in required if not p.exists() or p.stat().st_size == 0]
    if missing:
        logger.warning("Faltan archivos procesados: %s", missing)
        return False
    logger.info("Datos procesados encontrados en data/processed.")
    return True


def raw_data_exists() -> bool:
    if RAW_DATA.exists() and RAW_DATA.stat().st_size > 0:
        logger.info("Dataset crudo encontrado: %s", RAW_DATA)
        return True
    logger.warning("Dataset crudo no encontrado: %s", RAW_DATA)
    return False


def ensure_data_ready() -> None:
    """Garantiza que la base se descargue y que X_train.csv exista."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    if not raw_data_exists():
        run([sys.executable, "src/data/download.py"], "descarga-de-datos")

    if not processed_data_exists():
        run([sys.executable, "src/data/preprocess.py"], "preprocesamiento-de-datos")

    if not processed_data_exists():
        logger.error("El preprocesamiento termino, pero siguen faltando archivos procesados.")
        sys.exit(1)


def main() -> None:
    logger.info("=" * 70)
    logger.info("Trainer autonomo | Modelo: %s | Stage: %s", MODEL_NAME, MODEL_STAGE)
    logger.info("MLflow: %s", MLFLOW_URI)
    logger.info("=" * 70)

    client = wait_for_mlflow()

    # Importante: esto va ANTES de saltarse entrenamiento.
    # La API necesita X_train.csv para saber el orden de columnas.
    ensure_data_ready()

    if model_in_stage(client):
        logger.info("Modelo ya existe y datos procesados ya existen. No se reentrena.")
        sys.exit(0)

    logger.info("No hay modelo en '%s'. Iniciando entrenamiento completo.", MODEL_STAGE)
    run([sys.executable, "src/models/train.py"], "entrenamiento-mlflow-optuna")
    run([sys.executable, "scripts/promote_model.py"], "promocion-a-production")

    logger.info("Pipeline completado correctamente.")


if __name__ == "__main__":
    main()
