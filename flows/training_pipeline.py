"""
Pipeline de entrenamiento orquestado con Prefect 3.x.

Tasks:
    1. download_data       - Descarga el dataset desde UCI
    2. preprocess_data     - Preprocesamiento y feature engineering
    3. run_optuna_study    - Optimizacion de hiperparametros (20 trials)
    4. register_best_model - Registra el mejor modelo en MLflow
    5. validate_model      - Valida que el modelo cumpla las metricas minimas
    6. generate_report     - Crea un artefacto de Prefect con el resumen

Uso:
    # Ejecucion directa
    uv run python flows/training_pipeline.py

    # Con servidor de Prefect corriendo:
    uv run prefect server start
    uv run python flows/training_pipeline.py
"""

import logging
import os
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

logging.basicConfig(level=logging.INFO)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
OPTUNA_STORAGE = os.getenv("OPTUNA_STORAGE", "sqlite:///optuna.db")
N_TRIALS = int(os.getenv("N_TRIALS", "20"))
MIN_AUC = float(os.getenv("MIN_AUC", "0.68"))
MODEL_NAME = "diabetes-readmission-xgboost"
EXPERIMENT_NAME = "diabetes-readmission-prediction"


@task(name="download-dataset", retries=2, retry_delay_seconds=15)
def download_data_task() -> Path:
    """Descarga el dataset Diabetes 130-US Hospitals desde UCI."""
    log = get_run_logger()
    from src.data.download import download_from_ucimlrepo

    output_path = Path("data/raw/diabetic_data.csv")
    df = download_from_ucimlrepo(output_path)
    log.info("Dataset disponible: %d filas x %d columnas", *df.shape)
    return output_path


@task(name="preprocess-and-engineer")
def preprocess_data_task(raw_path: Path):
    """Aplica feature engineering y preprocesamiento. Guarda los splits en disco."""
    log = get_run_logger()
    import pandas as pd
    from src.data.preprocess import load_raw_data, preprocess, save_processed_data
    from src.features.engineering import apply_all_features

    df = load_raw_data(raw_path)
    df = apply_all_features(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    save_processed_data(X_train, X_test, y_train, y_test)

    log.info(
        "Preprocesamiento OK. X_train: %s | Positivos: %.1f%%",
        X_train.shape,
        y_train.mean() * 100,
    )
    return X_train.shape, float(y_train.mean())


@task(name="optuna-hyperparameter-search")
def run_optuna_study_task() -> dict:
    """
    Ejecuta el estudio de Optuna con MLflow tracking.
    Registra cada trial como un run hijo en MLflow.
    Retorna los mejores hiperparametros encontrados.
    """
    log = get_run_logger()
    import mlflow
    import optuna
    import pandas as pd
    from src.models.train import objective, compute_metrics
    from src.data.preprocess import TARGET

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    with mlflow.start_run(run_name="prefect_optuna_study") as parent_run:
        mlflow.set_tag("orchestration", "prefect")
        mlflow.set_tag("n_trials", str(N_TRIALS))

        study = optuna.create_study(
            study_name="diabetes-readmission-xgb",
            direction="maximize",
            storage=OPTUNA_STORAGE,
            load_if_exists=True,
        )

        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                X_test,
                y_train,
                y_test,
                parent_run.info.run_id,
            ),
            n_trials=N_TRIALS,
        )

        best_params = study.best_params
        best_auc = study.best_value

        mlflow.log_metric("best_roc_auc", best_auc)
        log.info("Mejor AUC tras %d trials: %.4f", N_TRIALS, best_auc)

    return {"best_params": best_params, "best_auc": best_auc}


@task(name="register-best-model")
def register_best_model_task(study_result: dict) -> str:
    """Entrena el modelo final y lo registra en el MLflow Model Registry."""
    log = get_run_logger()
    import mlflow
    import mlflow.xgboost
    import pandas as pd
    from xgboost import XGBClassifier
    from src.models.train import compute_metrics

    best_params = study_result["best_params"]

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    with mlflow.start_run(run_name="best_model_registered") as run:
        mlflow.set_tag("registered_by", "prefect")
        mlflow.log_params(best_params)

        model = XGBClassifier(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        run_id = run.info.run_id
        log.info(
            "Modelo registrado en MLflow. AUC: %.4f | Run: %s",
            metrics["roc_auc"],
            run_id,
        )

    return run_id


@task(name="validate-model-metrics")
def validate_model_task(study_result: dict) -> bool:
    """Valida que el modelo alcance el umbral minimo de AUC."""
    log = get_run_logger()
    best_auc = study_result["best_auc"]
    passes = best_auc >= MIN_AUC

    if passes:
        log.info("Validacion OK. AUC=%.4f >= umbral=%.4f", best_auc, MIN_AUC)
    else:
        log.warning(
            "VALIDACION FALLIDA. AUC=%.4f < umbral=%.4f. "
            "El modelo NO debe promoverse a Production.",
            best_auc,
            MIN_AUC,
        )
    return passes


@task(name="generate-pipeline-report")
def generate_report_task(
    study_result: dict,
    run_id: str,
    metrics_ok: bool,
    train_shape: tuple,
) -> None:
    """Crea un reporte Markdown del pipeline como artefacto de Prefect."""
    status = "EXITOSO" if metrics_ok else "FALLIDO (metricas por debajo del umbral)"
    best_auc = study_result["best_auc"]
    best_params = study_result["best_params"]

    params_md = "\n".join(
        f"- `{k}`: `{round(v, 6) if isinstance(v, float) else v}`"
        for k, v in best_params.items()
    )

    create_markdown_artifact(
        key="training-pipeline-report",
        markdown=f"""
# Reporte del Pipeline de Entrenamiento

**Estado:** {status}
**Dataset:** Diabetes 130-US Hospitals (UCI ID: 296)
**Filas de entrenamiento:** {train_shape[0]:,}
**Features:** {train_shape[1]}

## Resultado de la Optimizacion

- **Trials ejecutados:** {N_TRIALS}
- **Mejor ROC-AUC:** {best_auc:.4f}
- **Umbral minimo:** {MIN_AUC}
- **MLflow Run ID:** `{run_id}`

## Mejores Hiperparametros

{params_md}

## Proximos Pasos

1. Revisar los {N_TRIALS} trials en MLflow: http://localhost:5000
2. Si las metricas son satisfactorias, promover el modelo a Production:
   ```python
   client.transition_model_version_stage(
       name="{MODEL_NAME}", version=1, stage="Production"
   )
   ```
3. Levantar la API: `uv run uvicorn src.api.main:app --reload`
4. Verificar el endpoint de salud: `curl http://localhost:8000/health`
""",
        description="Reporte del pipeline de entrenamiento MLOps",
    )


@flow(name="diabetes-readmission-training-pipeline", log_prints=True)
def training_pipeline(
    n_trials: int = N_TRIALS,
    min_auc: float = MIN_AUC,
) -> dict:
    """
    Pipeline principal de entrenamiento para prediccion de reingreso hospitalario.

    Args:
        n_trials: Numero de trials de Optuna para la busqueda de hiperparametros.
        min_auc: ROC-AUC minimo para aprobar la validacion del modelo.

    Returns:
        Diccionario con el resultado del pipeline.
    """
    raw_path = download_data_task()
    train_shape, _ = preprocess_data_task(raw_path)
    study_result = run_optuna_study_task()
    run_id = register_best_model_task(study_result)
    metrics_ok = validate_model_task(study_result)
    generate_report_task(study_result, run_id, metrics_ok, train_shape)

    return {
        "status": "success" if metrics_ok else "warning",
        "best_auc": study_result["best_auc"],
        "run_id": run_id,
        "metrics_ok": metrics_ok,
    }


if __name__ == "__main__":
    result = training_pipeline()
    print(f"\nPipeline completado.")
    print(f"Estado: {result['status']}")
    print(f"Mejor AUC: {result['best_auc']:.4f}")
    print(f"MLflow Run ID: {result['run_id']}")
