# Ejecucion Docker 

El warning/error:


Ese script valida:

1. conexion TCP a `mlflow:5000`,
2. respuesta HTTP,
3. disponibilidad real del API de MLflow.

## Ejecutar limpio

Desde la carpeta del proyecto:

```bash
docker compose down --remove-orphans
rm -rf data/raw data/processed mlruns_docker logs
mkdir -p data/raw data/processed logs
docker compose up --build
```

## Verificar descarga de datos

En otra terminal:

```bash
ls -lah data/raw
ls -lah data/processed
```

Debes ver:

```text
data/raw/diabetic_data.csv
data/processed/X_train.csv
data/processed/X_test.csv
data/processed/y_train.csv
data/processed/y_test.csv
```

## Logs utiles

```bash
docker compose logs -f mlflow
docker compose logs -f trainer
docker compose logs -f api
```

## Nota

El primer arranque puede tardar porque descarga la base, preprocesa datos, entrena XGBoost y registra el modelo en MLflow. Los siguientes arranques deben ser mucho mas rapidos si no borras `data/` ni `mlruns_docker/`.
