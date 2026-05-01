"""Tests unitarios para src/data/preprocess.py"""

import pandas as pd
import pytest

from src.data.preprocess import (
    create_binary_target,
    encode_age,
    encode_medications,
    handle_missing_values,
    DISCHARGE_EXCLUDE,
    TARGET,
    TARGET_RAW,
)


@pytest.fixture
def sample_df():
    """DataFrame minimo con la estructura del dataset diabetico."""
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4],
            "patient_nbr": [10, 20, 30, 40],
            "age": ["[70-80)", "[50-60)", "[30-40)", "[80-90)"],
            "gender": ["Male", "Female", "Male", "Female"],
            "time_in_hospital": [5, 3, 8, 2],
            "num_medications": [14, 8, 22, 5],
            "num_lab_procedures": [44, 31, 66, 20],
            "number_diagnoses": [9, 5, 12, 3],
            "number_emergency": [1, 0, 2, 0],
            "number_inpatient": [1, 0, 3, 0],
            "number_outpatient": [0, 1, 0, 0],
            "discharge_disposition_id": [1, 1, 1, 1],
            "insulin": ["Up", "No", "Steady", "Down"],
            "A1Cresult": [">8", "None", "Norm", ">7"],
            "diabetesMed": ["Yes", "Yes", "No", "Yes"],
            "change": ["Ch", "No", "Ch", "No"],
            TARGET_RAW: ["<30", ">30", "NO", "<30"],
        }
    )


def test_create_binary_target_marks_lt30(sample_df):
    """Los registros con readmitted=='<30' deben tener target=1."""
    result = create_binary_target(sample_df)
    assert TARGET in result.columns
    assert TARGET_RAW not in result.columns
    lt30_rows = result[TARGET] == 1
    assert lt30_rows.sum() == 2


def test_create_binary_target_excludes_deceased():
    """Los registros con discharge_disposition_id en DISCHARGE_EXCLUDE deben eliminarse."""
    df = pd.DataFrame(
        {
            "discharge_disposition_id": [1, 11, 19],
            TARGET_RAW: ["<30", ">30", "NO"],
        }
    )
    result = create_binary_target(df)
    assert len(result) == 1
    assert result.iloc[0]["discharge_disposition_id"] == 1


def test_handle_missing_values_fills_object_cols():
    """Las columnas de texto con nulos deben rellenarse con 'Unknown'."""
    df = pd.DataFrame({"gender": ["Male", None, "Female"], "age": [70, 60, 50]})
    result = handle_missing_values(df)
    assert result["gender"].isnull().sum() == 0
    assert "Unknown" in result["gender"].values


def test_handle_missing_values_fills_numeric_with_median():
    """Las columnas numericas con nulos deben rellenarse con la mediana."""
    df = pd.DataFrame({"num_medications": [5, None, 15, 10]})
    result = handle_missing_values(df)
    assert result["num_medications"].isnull().sum() == 0
    # Mediana de [5, 15, 10] = 10
    assert result.iloc[1]["num_medications"] == 10.0


def test_encode_age_converts_intervals():
    """Los intervalos de edad deben convertirse a valores numericos ordinales."""
    df = pd.DataFrame({"age": ["[0-10)", "[50-60)", "[90-100)"]})
    result = encode_age(df)
    assert result.iloc[0]["age"] == 5
    assert result.iloc[1]["age"] == 55
    assert result.iloc[2]["age"] == 95


def test_encode_age_unknown_maps_to_median():
    """Los valores de edad desconocidos deben mapearse a 50."""
    df = pd.DataFrame({"age": ["Unknown", "[70-80)"]})
    result = encode_age(df)
    assert result.iloc[0]["age"] == 50


def test_encode_medications_maps_correctly():
    """Las etiquetas de medicamentos deben mapearse a enteros."""
    df = pd.DataFrame({"insulin": ["No", "Steady", "Up", "Down"]})
    result = encode_medications(df)
    assert result.iloc[0]["insulin"] == 0   # No
    assert result.iloc[1]["insulin"] == 1   # Steady
    assert result.iloc[2]["insulin"] == 2   # Up
    assert result.iloc[3]["insulin"] == 2   # Down
