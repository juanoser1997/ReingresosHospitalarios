"""Tests unitarios para src/features/engineering.py"""

import pandas as pd
import pytest

from src.features.engineering import (
    add_comorbidity_score,
    add_high_emergency_flag,
    add_insulin_change_flag,
    add_polypharmacy_flag,
    add_service_utilization,
    add_total_visits,
    apply_all_features,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "number_outpatient": [0, 2, 0],
            "number_emergency": [1, 0, 3],
            "number_inpatient": [1, 0, 2],
            "num_medications": [14, 8, 5],
            "num_lab_procedures": [44, 20, 66],
            "time_in_hospital": [5, 3, 8],
            "number_diagnoses": [9, 4, 2],
            "insulin": [2, 0, 1],     # ya codificado: Up=2, No=0, Steady=1
            "A1Cresult": [3, 0, 1],   # ya codificado
        }
    )


def test_add_total_visits_sums_correctly(sample_df):
    result = add_total_visits(sample_df)
    assert "total_prior_visits" in result.columns
    assert result.iloc[0]["total_prior_visits"] == 2   # 0+1+1
    assert result.iloc[1]["total_prior_visits"] == 2   # 2+0+0
    assert result.iloc[2]["total_prior_visits"] == 5   # 0+3+2


def test_add_polypharmacy_flag_threshold(sample_df):
    result = add_polypharmacy_flag(sample_df, threshold=10)
    assert "polypharmacy" in result.columns
    assert result.iloc[0]["polypharmacy"] == 1   # 14 >= 10
    assert result.iloc[1]["polypharmacy"] == 0   # 8 < 10
    assert result.iloc[2]["polypharmacy"] == 0   # 5 < 10


def test_add_service_utilization_non_negative(sample_df):
    result = add_service_utilization(sample_df)
    assert "lab_procedures_per_day" in result.columns
    assert (result["lab_procedures_per_day"] >= 0).all()


def test_add_comorbidity_score_categories(sample_df):
    result = add_comorbidity_score(sample_df)
    assert "comorbidity_score" in result.columns
    assert result.iloc[0]["comorbidity_score"] == 2   # 9 diagnosticos -> alto
    assert result.iloc[1]["comorbidity_score"] == 1   # 4 -> medio
    assert result.iloc[2]["comorbidity_score"] == 0   # 2 -> bajo


def test_add_insulin_change_flag(sample_df):
    result = add_insulin_change_flag(sample_df)
    assert "insulin_changed" in result.columns
    assert result.iloc[0]["insulin_changed"] == 1   # Up=2 -> cambiado
    assert result.iloc[1]["insulin_changed"] == 0   # No=0 -> no cambiado
    assert result.iloc[2]["insulin_changed"] == 0   # Steady=1 -> no cambiado


def test_add_high_emergency_flag(sample_df):
    result = add_high_emergency_flag(sample_df)
    assert "had_emergency" in result.columns
    assert result.iloc[0]["had_emergency"] == 1   # 1 emergencia
    assert result.iloc[1]["had_emergency"] == 0   # 0 emergencias
    assert result.iloc[2]["had_emergency"] == 1   # 3 emergencias


def test_apply_all_features_increases_columns(sample_df):
    """apply_all_features debe agregar al menos 6 columnas nuevas."""
    original_cols = len(sample_df.columns)
    result = apply_all_features(sample_df)
    assert len(result.columns) >= original_cols + 6


def test_apply_all_features_no_nulls_introduced(sample_df):
    """apply_all_features no debe introducir nuevos nulos."""
    result = apply_all_features(sample_df)
    assert result.isnull().sum().sum() == 0
