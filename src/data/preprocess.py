"""
Preprocesamiento del dataset Diabetes 130-US Hospitals.

Pasos:
1. Eliminar columnas con alta proporcion de nulos o sin valor predictivo
2. Codificar el target como binario (reingreso < 30 dias: 1 / resto: 0)
3. Manejar valores especiales ('?' como nulo)
4. Imputar valores faltantes
5. Encoding de variables categoricas
6. Split estratificado train/test (80/20)

Referencia del dataset:
    Strack et al. (2014). Impact of HbA1c Measurement on Hospital Readmission
    Rates: Analysis of 70,000 Clinical Database Patient Records.
    BioMed Research International.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/diabetic_data.csv")
PROCESSED_DATA_PATH = Path("data/processed")

TARGET_RAW = "readmitted"
TARGET = "readmitted_lt30"

# Columnas a eliminar: >40% nulos, IDs, o sin valor predictivo clinico
COLS_TO_DROP = [
    "encounter_id",
    "patient_nbr",
    "weight",           # 97% nulos
    "payer_code",       # 40% nulos, no clinico
    "medical_specialty",# 49% nulos
    "examide",          # varianza cero
    "citoglipton",      # varianza cero
]

# Columnas de medicamentos (26 variables farmacologicas)
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Categorias de admission_type, discharge_disposition y admission_source
# que corresponden a fallecidos o de hospice: excluir del dataset
DISCHARGE_EXCLUDE = [11, 13, 14, 19, 20, 21]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Carga los datos crudos; si no existen, los descarga automaticamente."""
    if not path.exists():
        logger.warning("No se encontro %s. Descargando dataset automaticamente...", path)
        from src.data.download import download_from_ucimlrepo

        download_from_ucimlrepo(path)

    df = pd.read_csv(path, na_values="?")
    logger.info("Datos cargados: %s filas, %s columnas", *df.shape)
    return df


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea el target binario: 1 si reingreso en menos de 30 dias, 0 en caso contrario.
    Tambien elimina encuentros de pacientes fallecidos o en hospice.
    """
    df = df.copy()

    # Excluir pacientes fallecidos o transferidos a hospice
    df = df[~df["discharge_disposition_id"].isin(DISCHARGE_EXCLUDE)]

    # Crear target binario
    df[TARGET] = (df[TARGET_RAW] == "<30").astype(int)
    df = df.drop(columns=[TARGET_RAW])

    pos_rate = df[TARGET].mean() * 100
    logger.info(
        "Target creado. Clase positiva (reingreso <30 dias): %.1f%%", pos_rate
    )
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes.
    - Categoricos: rellenar con 'Unknown'
    - Numericos: rellenar con la mediana
    """
    df = df.copy()

    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    remaining = df.isnull().sum().sum()
    logger.info("Nulos restantes tras imputacion: %d", remaining)
    return df


def encode_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte la columna 'age' de intervalos de texto a valores numericos ordinales.
    Ejemplo: '[0-10)' -> 5, '[10-20)' -> 15, etc.
    """
    df = df.copy()
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    df["age"] = df["age"].map(age_map).fillna(50)
    return df


def encode_medications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las columnas de medicamentos a numericas:
    No -> 0, Steady -> 1, Up -> 2, Down -> 2
    """
    df = df.copy()
    med_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 2}
    for col in MED_COLS:
        if col in df.columns:
            df[col] = df[col].map(med_map).fillna(0).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica Label Encoding a las columnas categoricas restantes.
    Usa LabelEncoder de sklearn para mantener reproducibilidad.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    logger.info(
        "Encoding completado. Columnas codificadas: %d. Shape: %s",
        len(cat_cols),
        df.shape,
    )
    return df


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Pipeline completo de preprocesamiento.

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("Iniciando preprocesamiento...")

    # 1. Eliminar columnas sin valor
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info("Columnas eliminadas: %s", cols_to_drop)

    # 2. Crear target binario y filtrar fallecidos
    df = create_binary_target(df)

    # 3. Imputar nulos
    df = handle_missing_values(df)

    # 4. Encodings especificos
    df = encode_age(df)
    df = encode_medications(df)

    # 5. Encoding general de categoricas restantes
    df = encode_categoricals(df)

    # 6. Split estratificado (mantiene balance de clases en ambos sets)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Split estratificado. Train: %d | Test: %d | Positivos train: %.1f%%",
        len(X_train),
        len(X_test),
        y_train.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path = PROCESSED_DATA_PATH,
) -> None:
    """Guarda los datasets procesados en disco."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)
    logger.info("Datos procesados guardados en %s", output_dir)


if __name__ == "__main__":
    df = load_raw_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    save_processed_data(X_train, X_test, y_train, y_test)
    print(f"Preprocesamiento completado.")
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Tasa de reingreso en train: {y_train.mean():.3f}")
