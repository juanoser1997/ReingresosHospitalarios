"""
Feature engineering para el dataset Diabetes 130-US Hospitals.

Crea variables clinicas derivadas que capturan patrones de alto riesgo
de reingreso documentados en la literatura medica.

Referencias:
    - Strack et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates.
    - Donze et al. (2013). Potentially Avoidable 30-Day Hospital Readmissions.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_total_visits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Suma el total de visitas previas al hospital.
    Un alto numero de visitas previas es predictor conocido de reingreso.
    """
    df = df.copy()
    visit_cols = [
        c for c in ["number_outpatient", "number_emergency", "number_inpatient"]
        if c in df.columns
    ]
    if visit_cols:
        df["total_prior_visits"] = df[visit_cols].sum(axis=1)
        logger.info("Feature 'total_prior_visits' creada.")
    return df


def add_polypharmacy_flag(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    """
    Indica polifarmacia: uso de 10 o mas medicamentos distintos.
    La polifarmacia es un factor de riesgo independiente de reingreso en diabeticos.
    """
    df = df.copy()
    if "num_medications" in df.columns:
        df["polypharmacy"] = (df["num_medications"] >= threshold).astype(int)
        pct = df["polypharmacy"].mean() * 100
        logger.info(
            "Feature 'polypharmacy' creada. Pacientes con polifarmacia: %.1f%%", pct
        )
    return df


def add_service_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio entre procedimientos de laboratorio y duracion de la hospitalizacion.
    Un ratio alto indica mayor complejidad clinica.
    """
    df = df.copy()
    if "num_lab_procedures" in df.columns and "time_in_hospital" in df.columns:
        df["lab_procedures_per_day"] = (
            df["num_lab_procedures"] / (df["time_in_hospital"] + 1)
        )
        logger.info("Feature 'lab_procedures_per_day' creada.")
    return df


def add_comorbidity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score simplificado de comorbilidad basado en el numero de diagnosticos.
    Inspirado en el indice de Charlson adaptado para datos administrativos.
    Categories: bajo (1-3), medio (4-6), alto (>=7).
    """
    df = df.copy()
    if "number_diagnoses" in df.columns:
        df["comorbidity_score"] = pd.cut(
            df["number_diagnoses"],
            bins=[0, 3, 6, 100],
            labels=[0, 1, 2],
        ).astype(int)
        logger.info("Feature 'comorbidity_score' creada.")
    return df


def add_insulin_change_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indica si hubo uso o cambio en la insulina durante la hospitalizacion.

    ⚠️ IMPORTANTE:
    Esta version trabaja con valores STRING (ANTES del encoding).
    'No' = no uso de insulina
    'Up', 'Down', 'Steady' = uso/cambio de insulina
    """
    df = df.copy()
    if "insulin" in df.columns:
        insulin_str = df["insulin"].astype(str).str.strip().str.lower()

        # Todo lo que NO sea 'no' implica uso/cambio
        df["insulin_changed"] = (insulin_str != "no").astype(int)

        logger.info("Feature 'insulin_changed' creada (pre-encoding).")
    return df


def add_a1c_tested_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indica si se realizo la prueba de HbA1c durante la hospitalizacion.

    ⚠️ IMPORTANTE:
    Trabaja con strings antes del encoding.
    'None' o 'NotTested' = no se realizo la prueba
    """
    df = df.copy()
    if "A1Cresult" in df.columns:
        a1c_str = df["A1Cresult"].astype(str).str.strip().str.lower()

        # Si NO está en estos valores → sí se hizo prueba
        df["a1c_tested"] = (~a1c_str.isin(["none", "nottested"])).astype(int)

        logger.info("Feature 'a1c_tested' creada (pre-encoding).")
    return df


def add_high_emergency_flag(df: pd.DataFrame, threshold: int = 1) -> pd.DataFrame:
    """
    Indica si el paciente tuvo visitas de emergencia previas.
    Predictor fuerte de reingreso en pacientes cronicos.
    """
    df = df.copy()
    if "number_emergency" in df.columns:
        df["had_emergency"] = (df["number_emergency"] >= threshold).astype(int)
        logger.info("Feature 'had_emergency' creada.")
    return df


def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas las transformaciones de feature engineering al DataFrame.
    Debe llamarse ANTES del encoding y del split train/test.
    """
    logger.info("Aplicando feature engineering clinico...")
    df = add_total_visits(df)
    df = add_polypharmacy_flag(df)
    df = add_service_utilization(df)
    df = add_comorbidity_score(df)
    df = add_insulin_change_flag(df)
    df = add_a1c_tested_flag(df)
    df = add_high_emergency_flag(df)
    logger.info(
        "Feature engineering completado. Shape final: %s", df.shape
    )
    return df