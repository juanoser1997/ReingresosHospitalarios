"""
Descarga el dataset Diabetes 130-US Hospitals desde el UCI ML Repository.

Dataset: Diabetes 130-US Hospitals for years 1999-2008
URL:     https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
Filas:   101,766 encuentros hospitalarios
Columnas: 50 variables clinicas y administrativas
Target:  readmitted (<30 dias, >30 dias, NO)

Uso:
    uv run python src/data/download.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/diabetic_data.csv")
IDS_MAPPING_PATH = Path("data/raw/IDS_mapping.csv")
UCI_DATASET_ID = 296


def download_from_ucimlrepo(output_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Descarga el dataset desde UCI usando la libreria ucimlrepo.

    Args:
        output_path: Ruta donde guardar el CSV.

    Returns:
        DataFrame con los datos crudos.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info("Dataset ya existe en %s. Cargando desde disco.", output_path)
        return pd.read_csv(output_path)

    logger.info(
        "Descargando dataset Diabetes 130-US Hospitals desde UCI (ID: %d)...",
        UCI_DATASET_ID,
    )

    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
    except Exception as e:
        logger.warning(
            "ucimlrepo fallo (%s). Intentando descarga directa desde UCI...", e
        )
        df = _download_direct()

    df.to_csv(output_path, index=False)
    logger.info(
        "Dataset guardado en %s. Shape: %s",
        output_path,
        df.shape,
    )
    return df


def _download_direct() -> pd.DataFrame:
    """
    Descarga directa desde el repositorio UCI como fallback.
    URL alternativa con el archivo original del paper de Strack et al. (2014).
    """
    import urllib.request
    import zipfile
    import io

    url = (
        "https://archive.ics.uci.edu/static/public/296/"
        "diabetes+130-us+hospitals+for+years+1999-2008.zip"
    )
    logger.info("Descargando desde: %s", url)

    with urllib.request.urlopen(url) as response:  # noqa: S310
        zip_data = io.BytesIO(response.read())

    with zipfile.ZipFile(zip_data) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith("diabetic_data.csv")]
        if not csv_names:
            raise FileNotFoundError(
                "No se encontro diabetic_data.csv dentro del ZIP de UCI."
            )
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f)

    return df


if __name__ == "__main__":
    df = download_from_ucimlrepo()
    print(f"\nDataset descargado correctamente.")
    print(f"Shape: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"\nDistribucion del target (readmitted):")
    print(df["readmitted"].value_counts())
    pct_lt30 = (df["readmitted"] == "<30").mean() * 100
    print(f"\nPorcentaje de reingresos < 30 dias: {pct_lt30:.1f}%")
