"""
Tests de integracion para el pipeline completo.

Requieren que los datos procesados existan en data/processed/.
Ejecutar despues de correr el pipeline de entrenamiento:
    uv run python flows/training_pipeline.py

Ejecutar con:
    uv run pytest tests/integration/ -v -m integration
"""

from pathlib import Path

import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed")


@pytest.mark.integration
def test_processed_files_exist():
    """Los cuatro archivos de datos procesados deben existir."""
    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        assert (PROCESSED_DIR / fname).exists(), f"Falta: {PROCESSED_DIR / fname}"


@pytest.mark.integration
def test_train_test_shapes_consistent():
    """X_train y y_train deben tener el mismo numero de filas."""
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.shape[1] == X_test.shape[1], (
        f"Train tiene {X_train.shape[1]} columnas y test {X_test.shape[1]}"
    )


@pytest.mark.integration
def test_no_nulls_in_features():
    """No debe haber valores nulos en las features procesadas."""
    for fname in ["X_train.csv", "X_test.csv"]:
        df = pd.read_csv(PROCESSED_DIR / fname)
        nulls = df.isnull().sum().sum()
        assert nulls == 0, f"{fname} contiene {nulls} valores nulos"


@pytest.mark.integration
def test_target_is_binary():
    """El target debe contener solo 0 y 1."""
    for fname in ["y_train.csv", "y_test.csv"]:
        y = pd.read_csv(PROCESSED_DIR / fname).squeeze()
        unique_values = set(y.unique())
        assert unique_values <= {0, 1}, (
            f"{fname} contiene valores distintos de 0 y 1: {unique_values}"
        )


@pytest.mark.integration
def test_class_imbalance_within_expected_range():
    """
    La tasa de la clase positiva debe estar entre 8% y 15%
    segun lo conocido del dataset UCI Diabetes 130-US.
    """
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    positive_rate = y_train.mean()
    assert 0.08 <= positive_rate <= 0.15, (
        f"Tasa de positivos inesperada: {positive_rate:.3f} "
        "(esperado entre 0.08 y 0.15)"
    )


@pytest.mark.integration
def test_engineered_features_present():
    """Las features de engineering deben estar presentes en los datos procesados."""
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    expected_features = [
        "total_prior_visits",
        "polypharmacy",
        "lab_procedures_per_day",
        "comorbidity_score",
        "insulin_changed",
        "had_emergency",
    ]
    for feat in expected_features:
        assert feat in X_train.columns, (
            f"Feature de engineering faltante: '{feat}'"
        )


@pytest.mark.integration
def test_train_larger_than_test():
    """El conjunto de entrenamiento debe ser mas grande que el de prueba."""
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    assert len(X_train) > len(X_test)
