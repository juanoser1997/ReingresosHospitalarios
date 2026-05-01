# Mapeo del Proyecto Final con los Modulos del Curso

Universidad de Medellin - MLOps

---

Este documento explica como cada componente del proyecto final cubre
los contenidos de los modulos del curso MLOps.

---

## Modulo 00 - Setup del Entorno

**Archivos del proyecto:**
- `pyproject.toml`: gestion de dependencias con `uv`
- `.python-version`: version de Python fijada a 3.11
- `.pre-commit-config.yaml`: hooks con ruff para calidad de codigo
- `.env.example`: variables de entorno documentadas

**Lo que demuestra:** configuracion de entorno reproducible y profesional.

---

## Modulo 01 - Introduccion a ML

**Archivos del proyecto:**
- `notebooks/01_eda.ipynb`: analisis exploratorio completo del dataset clinico
- `notebooks/02_baseline.ipynb`: modelo baseline con DummyClassifier y Regresion Logistica
- `src/data/download.py`: ingesta programatica desde UCI ML Repository
- `src/data/preprocess.py`: pipeline de limpieza y transformacion

**Lo que demuestra:** entendimiento del problema, EDA sistematico y
establecimiento de un baseline medible.

---

## Modulo 02 - Experiment Tracking con MLflow

**Archivos del proyecto:**
- `src/models/train.py`: tracking de parametros, metricas y artefactos con MLflow 3.x
- `notebooks/03_experiments.ipynb`: exploracion interactiva de experimentos
- `configs/mlflow_config.yaml`: configuracion del servidor de tracking y registry

**Lo que demuestra:**
- Logging de hiperparametros, metricas (AUC, recall, F1) y artefactos
- Comparacion de multiples runs en la UI de MLflow
- Registro y versionado en el MLflow Model Registry
- Transicion de modelos entre etapas: None -> Staging -> Production

---

## Modulo 03 - Orquestacion con Prefect

**Archivos del proyecto:**
- `flows/training_pipeline.py`: pipeline completo con tasks y flows de Prefect 3.x
- Incluye retry logic, caching, logging y artifacts de Prefect

**Lo que demuestra:**
- Diseno de tasks independientes con `@task` y `@flow`
- Retry automatico en la tarea de descarga de datos
- Optimizacion de hiperparametros con **Optuna** (20+ trials) registrados en MLflow
- Generacion de artefactos de reporte en cada ejecucion
- Scheduling: el flow puede programarse con `prefect deployment apply`

---

## Modulo 04 - Deployment de Modelos

**Archivos del proyecto:**
- `src/api/main.py`: API REST con FastAPI y lifespan para carga del modelo
- `src/api/schemas.py`: validacion de inputs con Pydantic v2
- `src/models/predict.py`: inferencia desde el MLflow Model Registry
- `Dockerfile`: imagen Docker optimizada para produccion
- `docker-compose.yml`: stack completo (API + MLflow + Prefect)
- `.github/workflows/deploy.yml`: CI/CD para build y push de imagen Docker

**Lo que demuestra:**
- Deployment como web service REST con FastAPI
- Containerizacion reproducible con Docker
- Endpoints: `/health`, `/predict`, `/predict/batch`
- Variables de entorno para configuracion sin recompilar imagen
- Pipeline de CI/CD con GitHub Actions

---

## Modulo 05 - Monitoreo y Observabilidad

**Archivos del proyecto:**
- `src/monitoring/drift.py`: deteccion de data drift con Evidently
- `docs/monitoring.md`: estrategia completa de monitoreo propuesta

**Lo que demuestra:**
- Comparacion de distribuciones de features entre referencia y produccion
- Deteccion de drift en predicciones por umbral relativo
- Propuesta de dashboards con Grafana y metricas de Prometheus
- Estrategia de reentrenamiento basada en triggers objetivos

---

## Modulo 06 - Proyecto Final Integrador

**Este repositorio completo** es el proyecto final. Integra todos los modulos
en un pipeline end-to-end sobre un caso de uso real de salud:

- Problema de negocio definido (reduccion de reingresos hospitalarios)
- Dataset publico descargado programaticamente
- Pipeline de preprocesamiento reproducible
- Experiment tracking con MLflow + Optuna (20+ trials)
- Orquestacion con Prefect
- API REST deployada en Docker
- Tests unitarios y de integracion
- CI/CD con GitHub Actions
- Documentacion completa (README, docs/, CONTRIBUTING.md)
- 12 commits progresivos documentados

---

## Criterios de Evaluacion Cubiertos

| Criterio                                    | Cobertura      | Archivo Principal              |
|---------------------------------------------|----------------|-------------------------------|
| README con instrucciones de ejecucion       | Completo       | README.md                     |
| Experiment tracking con MLflow              | Completo       | src/models/train.py           |
| Optimizacion de hiperparametros             | Completo       | flows/training_pipeline.py    |
| Pipeline de entrenamiento (Prefect)         | Completo       | flows/training_pipeline.py    |
| Deployment como API REST                    | Completo       | src/api/main.py               |
| Containerizacion con Docker                 | Completo       | Dockerfile, docker-compose.yml|
| Unit tests                                  | Completo       | tests/unit/                   |
| CI/CD con GitHub Actions                    | Completo       | .github/workflows/            |
| Monitoreo propuesto                         | Completo       | docs/monitoring.md            |
| Documentacion tecnica                       | Completo       | docs/                         |
| Best practices (ruff, pre-commit)           | Completo       | .pre-commit-config.yaml       |
