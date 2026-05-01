"""
Monitoreo de data drift con Evidently para el modelo de reingreso hospitalario.

Compara la distribucion de los datos de produccion contra los datos de
entrenamiento para detectar cambios en la poblacion de pacientes.

Uso:
    uv run python src/monitoring/drift.py
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REFERENCE_DATA_PATH = Path("data/processed/X_train.csv")
REPORTS_PATH = Path("logs/drift_reports")

# Features clinicas clave para monitorear
KEY_FEATURES = [
    "age",
    "time_in_hospital",
    "num_medications",
    "num_lab_procedures",
    "number_diagnoses",
    "number_inpatient",
    "number_emergency",
    "total_prior_visits",
    "polypharmacy",
    "comorbidity_score",
]


def load_reference_data(path: Path = REFERENCE_DATA_PATH) -> pd.DataFrame:
    """Carga los datos de referencia (conjunto de entrenamiento)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Datos de referencia no encontrados en {path}. "
            "Ejecuta primero el pipeline de entrenamiento."
        )
    return pd.read_csv(path)


def compute_drift_report(
    current_data: pd.DataFrame,
    reference_data: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Genera un reporte HTML de data drift con Evidently.

    Args:
        current_data: Datos recientes de produccion (features de pacientes).
        reference_data: Datos de referencia. Si None, carga desde disco.
        output_path: Ruta para guardar el reporte HTML.

    Returns:
        Diccionario con el resultado del analisis de drift.
    """
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        logger.error(
            "Evidently no esta instalado. Ejecuta: uv add evidently"
        )
        return {"error": "evidently_not_installed"}

    if reference_data is None:
        reference_data = load_reference_data()

    # Usar solo features comunes y las mas relevantes clinicamente
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    key_available = [c for c in KEY_FEATURES if c in common_cols]
    cols_to_use = key_available if key_available else common_cols[:20]

    ref = reference_data[cols_to_use]
    cur = current_data[cols_to_use]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    if output_path is None:
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_PATH / "drift_report.html"

    report.save_html(str(output_path))
    logger.info("Reporte de drift guardado en: %s", output_path)

    result = report.as_dict()
    drift_metrics = result.get("metrics", [{}])[0].get("result", {})
    n_drifted = drift_metrics.get("number_of_drifted_columns", 0)
    n_total = drift_metrics.get("number_of_columns", len(cols_to_use))
    drift_share = drift_metrics.get("share_of_drifted_columns", 0.0)

    logger.info(
        "Drift detectado en %d/%d features (%.1f%%)",
        n_drifted,
        n_total,
        drift_share * 100,
    )

    return {
        "drifted_columns": n_drifted,
        "total_columns": n_total,
        "drift_share": drift_share,
        "report_path": str(output_path),
        "significant_drift": drift_share > 0.3,
    }


def check_prediction_distribution(
    probs_reference: pd.Series,
    probs_current: pd.Series,
    alert_threshold: float = 0.15,
) -> dict:
    """
    Compara la distribucion de probabilidades predichas entre referencia y produccion.
    Una desviacion mayor al umbral indica posible deterioro del modelo.

    Args:
        probs_reference: Probabilidades de reingreso en el periodo de referencia.
        probs_current: Probabilidades de reingreso en el periodo actual.
        alert_threshold: Diferencia relativa en la media que activa una alerta.

    Returns:
        Diccionario con metricas de comparacion y flag de alerta.
    """
    ref_mean = probs_reference.mean()
    cur_mean = probs_current.mean()
    ref_std = probs_reference.std()
    cur_std = probs_current.std()

    relative_diff = abs(cur_mean - ref_mean) / (ref_mean + 1e-10)
    alert = relative_diff > alert_threshold

    result = {
        "reference_mean": round(ref_mean, 4),
        "current_mean": round(cur_mean, 4),
        "reference_std": round(ref_std, 4),
        "current_std": round(cur_std, 4),
        "relative_diff": round(relative_diff, 4),
        "alert": alert,
    }

    log_fn = logger.warning if alert else logger.info
    log_fn(
        "Distribucion de predicciones: ref_mean=%.4f | cur_mean=%.4f "
        "| diff_rel=%.1f%% | alerta=%s",
        ref_mean,
        cur_mean,
        relative_diff * 100,
        alert,
    )

    return result


if __name__ == "__main__":
    import numpy as np

    # Simulacion de datos de produccion para probar el modulo
    ref_data = load_reference_data()
    sample_size = min(1000, len(ref_data))
    current_sample = ref_data.sample(sample_size, random_state=99)

    # Simular drift leve en 'age' y 'number_inpatient'
    if "age" in current_sample.columns:
        current_sample = current_sample.copy()
        current_sample["age"] = current_sample["age"] * 1.1

    result = compute_drift_report(current_sample, ref_data)
    print("\nResultado del analisis de drift:")
    for k, v in result.items():
        print(f"  {k}: {v}")
