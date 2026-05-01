"""
Entrenamiento del modelo con MLflow 3.x y Optuna para optimizacion de hiperparametros.

Flujo:
1. Carga y preprocesamiento de datos
2. Estudio de Optuna con 20 trials (XGBoost)
3. Cada trial es un run hijo en MLflow
4. El mejor trial registra el modelo en el MLflow Model Registry
5. Se generan reportes de feature importance como artefactos

Uso:
    uv run python src/models/train.py
"""

import logging
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.data.preprocess import load_raw_data, preprocess, save_processed_data
from src.features.engineering import apply_all_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/model_config.yaml")
EXPERIMENT_NAME = "diabetes-readmission-prediction"
MODEL_NAME = "diabetes-readmission-xgboost"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
OPTUNA_STORAGE = os.getenv("OPTUNA_STORAGE", "sqlite:///optuna.db")
N_TRIALS = int(os.getenv("N_TRIALS", "20"))


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calcula las metricas de clasificacion para modelos de salud."""
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    parent_run_id: str,
) -> float:
    """
    Funcion objetivo de Optuna. Cada llamada es un trial con hiperparametros
    distintos y se registra como un run hijo en MLflow.

    Optimiza ROC-AUC (metrica principal para clasificacion desbalanceada).
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 15.0),
    }

    with mlflow.start_run(
        run_name=f"trial_{trial.number:03d}",
        nested=True,
        tags={"trial_number": str(trial.number)},
    ):
        mlflow.log_params(params)

        model = XGBClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
            eval_metric="auc",
            early_stopping_rounds=30,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        logger.info(
            "Trial %d - AUC: %.4f | Recall: %.4f | F1: %.4f",
            trial.number,
            metrics["roc_auc"],
            metrics["recall"],
            metrics["f1"],
        )

    return metrics["roc_auc"]


def train_best_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_params: dict,
) -> None:
    """
    Entrena el modelo final con los mejores hiperparametros y lo registra en MLflow.
    """
    with mlflow.start_run(run_name="best_model_final"):
        mlflow.set_tag("model_type", "XGBClassifier")
        mlflow.set_tag("dataset", "diabetes_130_uci")
        mlflow.log_params(best_params)

        model = XGBClassifier(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("positive_rate_train", float(y_train.mean()))

        # Guardar feature importance como artefacto
        fi_path = Path("logs/feature_importance.csv")
        fi_path.parent.mkdir(exist_ok=True)
        fi_df = pd.DataFrame(
            {"feature": X_train.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        fi_df.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        # Registrar modelo en MLflow Model Registry
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        logger.info(
            "Modelo final - AUC: %.4f | Recall: %.4f | Precision: %.4f | F1: %.4f",
            metrics["roc_auc"],
            metrics["recall"],
            metrics["precision"],
            metrics["f1"],
        )
        return metrics


def run_training() -> None:
    """Pipeline completo de entrenamiento con Optuna + MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Cargando y preprocesando datos...")
    df = load_raw_data()
    df = apply_all_features(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    save_processed_data(X_train, X_test, y_train, y_test)

    logger.info(
        "Iniciando optimizacion con Optuna (%d trials)...", N_TRIALS
    )

    with mlflow.start_run(run_name="optuna_study") as parent_run:
        mlflow.set_tag("optimization", "optuna")
        mlflow.set_tag("n_trials", str(N_TRIALS))

        study = optuna.create_study(
            study_name="diabetes-readmission-xgb",
            direction="maximize",
            storage=OPTUNA_STORAGE,
            load_if_exists=True,
        )

        study.optimize(
            lambda trial: objective(
                trial, X_train, X_test, y_train, y_test, parent_run.info.run_id
            ),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_auc = study.best_value

        mlflow.log_metric("best_roc_auc", best_auc)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        logger.info(
            "Optimizacion completada. Mejor AUC: %.4f con params: %s",
            best_auc,
            best_params,
        )

    logger.info("Entrenando modelo final con los mejores hiperparametros...")
    metrics = train_best_model(X_train, X_test, y_train, y_test, best_params)

    logger.info("Entrenamiento completado. Ver resultados en http://localhost:5000")
    logger.info(
        "Siguiente paso: promover el modelo '%s' a Production en MLflow.", MODEL_NAME
    )
    return metrics


if __name__ == "__main__":
    run_training()
