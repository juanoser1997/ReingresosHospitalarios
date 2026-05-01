# Guia de Despliegue

---

## Requisitos Previos

- Python 3.11+
- uv: `pip install uv` o `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker Desktop (para el stack completo)
- Git configurado con tu usuario de GitHub

---

## Despliegue Local Paso a Paso

### Paso 1: Clonar e instalar

```bash
git clone https://github.com/<tu-usuario>/mlops-alto-costo.git
cd mlops-alto-costo

uv sync --all-groups
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### Paso 2: Configurar pre-commit y hooks

```bash
uv run pre-commit install
cp .env.example .env
```

### Paso 3: Descargar el dataset

```bash
uv run python src/data/download.py
```

Guarda `data/raw/diabetic_data.csv` (~23 MB, 101,766 filas).

### Paso 4: Explorar los datos (opcional)

```bash
uv run jupyter notebook notebooks/01_eda.ipynb
```

### Paso 5: Iniciar MLflow

En una terminal dedicada:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./models/artifacts \
  --host 0.0.0.0 \
  --port 5000
```

UI disponible en http://localhost:5000

### Paso 6: Iniciar Prefect (opcional para orquestacion)

En otra terminal:

```bash
uv run prefect server start
```

UI disponible en http://localhost:4200

### Paso 7: Ejecutar el pipeline de entrenamiento

```bash
# Con Prefect (recomendado)
uv run python flows/training_pipeline.py

# Sin Prefect (solo entrenamiento)
uv run python src/models/train.py
```

El pipeline ejecuta 20 trials de Optuna. Cada trial aparece como un run
hijo en MLflow. Tiempo estimado: 10-20 minutos segun el hardware.

### Paso 8: Promover el mejor modelo a Production

Desde la UI de MLflow (http://localhost:5000):
1. Ir a "Models" en el menu lateral
2. Seleccionar "diabetes-readmission-xgboost"
3. Seleccionar la version con mayor ROC-AUC
4. Cambiar el stage a "Production"

O desde Python:

```python
import mlflow

client = mlflow.tracking.MlflowClient("sqlite:///mlflow.db")
client.transition_model_version_stage(
    name="diabetes-readmission-xgboost",
    version=1,
    stage="Production",
)
print("Modelo promovido a Production.")
```

### Paso 9: Levantar la API

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

Verificar: `curl http://localhost:8000/health`

Documentacion interactiva: http://localhost:8000/docs

### Paso 10: Ejecutar los tests

```bash
# Tests unitarios (rapidos, sin datos en disco)
uv run pytest tests/unit/ -v

# Tests de integracion (requieren datos procesados)
uv run pytest tests/integration/ -v -m integration

# Todos los tests con cobertura
uv run pytest tests/ -v --cov=src --cov-report=html
```

---

## Despliegue con Docker Compose

Para levantar el stack completo en contenedores:

```bash
# Construir y levantar todo
docker-compose up --build

# Solo en background
docker-compose up -d --build

# Ver logs de la API
docker-compose logs -f api

# Detener todo
docker-compose down
```

Servicios disponibles tras el despliegue:

| Servicio | URL                   |
|----------|-----------------------|
| API      | http://localhost:8000 |
| MLflow   | http://localhost:5000 |
| Prefect  | http://localhost:4200 |

**Nota:** El contenedor de la API espera a que MLflow este saludable antes
de iniciar (depends_on con condition: service_healthy).

---

## Variables de Entorno

| Variable              | Default                          | Descripcion                              |
|-----------------------|----------------------------------|------------------------------------------|
| MODEL_NAME            | diabetes-readmission-xgboost     | Nombre del modelo en MLflow Registry     |
| MODEL_STAGE           | Production                       | Etapa del modelo a cargar                |
| MLFLOW_TRACKING_URI   | sqlite:///mlflow.db              | URI del servidor de MLflow               |
| DECISION_THRESHOLD    | 0.3                              | Umbral de clasificacion (0.0 - 1.0)      |
| OPTUNA_STORAGE        | sqlite:///optuna.db              | Storage para los estudios de Optuna      |
| N_TRIALS              | 20                               | Numero de trials de Optuna               |
| MIN_AUC               | 0.68                             | ROC-AUC minimo para validar el modelo    |

---

## Configuracion de GitHub Actions

Para activar el CI/CD automatico:

1. Ir a Settings > Secrets and variables > Actions en tu repositorio de GitHub
2. Agregar los secrets:
   - `DOCKER_USERNAME`: tu usuario de Docker Hub
   - `DOCKER_PASSWORD`: tu access token de Docker Hub (no el password)

El workflow `ci.yml` corre en cada push a `main` o `develop`:
linting con ruff, tests unitarios, build de Docker.

El workflow `deploy.yml` corre solo en merge a `main`:
build y push de la imagen a Docker Hub.

---

## Estrategia de Commits Progresivos

La siguiente secuencia construye el repositorio de forma incremental,
reflejando el desarrollo real por fases del proyecto MLOps.

### Commit 1: Estructura base del proyecto

```bash
git init
git add README.md pyproject.toml .python-version .gitignore \
        .env.example CONTRIBUTING.md COURSE_OVERVIEW.md
git commit -m "feat: inicializar estructura del proyecto MLOps

- README con descripcion del problema de negocio (reingreso hospitalario)
- pyproject.toml con uv y dependencias del curso (mlflow 3.x, prefect 3.x, optuna)
- .python-version fijado a 3.11
- CONTRIBUTING.md con flujo de trabajo y convenciones
- COURSE_OVERVIEW.md mapeando el proyecto a los modulos del curso"
```

### Commit 2: Calidad de codigo y CI inicial

```bash
git add .pre-commit-config.yaml .githooks/ .github/
git commit -m "ci: agregar pre-commit con ruff y workflows de GitHub Actions

- .pre-commit-config.yaml: hooks de ruff (linting + formateo)
- .github/workflows/ci.yml: lint + tests + build Docker en cada push
- .github/workflows/deploy.yml: push de imagen Docker en merge a main
- .githooks/pre-push: validacion adicional antes de push"
```

### Commit 3: Modulo de datos

```bash
git add src/data/ configs/mlflow_config.yaml
git commit -m "feat(data): agregar descarga y preprocesamiento del dataset UCI

- src/data/download.py: descarga programatica desde UCI ML Repository (ID: 296)
  con fallback a descarga directa si ucimlrepo falla
- src/data/preprocess.py: limpieza de '?', target binario (<30 dias),
  exclusion de fallecidos, encoding de edad y medicamentos, split estratificado
- configs/mlflow_config.yaml: configuracion del servidor de tracking"
```

### Commit 4: Feature engineering clinico

```bash
git add src/features/
git commit -m "feat(features): agregar feature engineering clinico

- total_prior_visits: suma de visitas previas (predictor de reingreso)
- polypharmacy: flag de polifarmacia (>=10 medicamentos)
- lab_procedures_per_day: intensidad de uso de laboratorio
- comorbidity_score: score simplificado basado en numero de diagnosticos
- insulin_changed: cambio en dosis de insulina (Strack et al. 2014)
- a1c_tested: si se midio HbA1c durante la hospitalizacion
- had_emergency: visitas previas a urgencias"
```

### Commit 5: Entrenamiento con MLflow y Optuna

```bash
git add src/models/ configs/model_config.yaml
git commit -m "feat(models): agregar entrenamiento XGBoost con MLflow 3.x y Optuna

- src/models/train.py: estudio Optuna con 20 trials, cada trial es un
  run hijo en MLflow; registra el mejor modelo en el Model Registry
- scale_pos_weight optimizado por Optuna para manejar desbalance de clases
- metricas clinicas: ROC-AUC, recall, precision, F1
- configs/model_config.yaml: hiperparametros y espacio de busqueda de Optuna
- src/models/predict.py: inferencia desde MLflow Registry con umbral 0.3"
```

### Commit 6: Pipeline de orquestacion con Prefect

```bash
git add flows/
git commit -m "feat(flows): agregar pipeline de entrenamiento con Prefect 3.x

- flows/training_pipeline.py: orquesta 6 tasks con retry, logging y artefactos
- download_data_task: retry=2 para servicio externo UCI
- run_optuna_study_task: 20 trials con runs hijos en MLflow
- register_best_model_task: registra el modelo final en el Registry
- validate_model_task: valida ROC-AUC minimo antes de aprobar el modelo
- generate_report_task: artefacto Markdown en Prefect con el resumen"
```

### Commit 7: API REST con FastAPI y Docker

```bash
git add src/api/ Dockerfile docker-compose.yml
git commit -m "feat(api): agregar API REST con FastAPI y contenerizacion Docker

- src/api/schemas.py: PatientFeatures con validacion Pydantic v2,
  PredictionResponse con niveles de riesgo y recomendaciones clinicas
- src/api/main.py: endpoints /health, /predict, /predict/batch (max 200)
  con lifespan para carga del modelo y logica de riesgo BAJO/MEDIO/ALTO
- Dockerfile: imagen Python 3.11-slim con uv, healthcheck configurado
- docker-compose.yml: stack completo API + MLflow + Prefect"
```

### Commit 8: Monitoreo con Evidently

```bash
git add src/monitoring/ docs/monitoring.md
git commit -m "feat(monitoring): agregar deteccion de drift con Evidently

- src/monitoring/drift.py: reporte HTML de data drift comparando
  features clinicas clave entre referencia y produccion
- check_prediction_distribution: alerta si la media de probabilidades
  se desvia mas del 15% respecto al periodo de referencia
- docs/monitoring.md: estrategia completa de monitoreo propuesta"
```

### Commit 9: Suite de tests

```bash
git add tests/
git commit -m "test: agregar suite de tests unitarios e integracion

- tests/unit/test_preprocess.py: 7 tests para descarga y preprocesamiento
- tests/unit/test_features.py: 8 tests para feature engineering clinico
- tests/unit/test_api.py: 12 tests para endpoints de la API (con mocks)
- tests/integration/test_pipeline.py: 7 tests del pipeline completo
  (marcados con @pytest.mark.integration)"
```

### Commit 10: Notebooks

```bash
git add notebooks/
git commit -m "docs(notebooks): agregar notebooks de EDA, baseline y experimentos

- notebooks/01_eda.ipynb: analisis exploratorio del dataset UCI Diabetes 130-US
  con distribucion de clases, correlaciones y analisis clinico
- notebooks/02_baseline.ipynb: DummyClassifier y Regresion Logistica como baseline
- notebooks/03_experiments.ipynb: exploracion interactiva de los trials de Optuna"
```

### Commit 11: Documentacion completa

```bash
git add docs/
git commit -m "docs: agregar documentacion tecnica completa

- docs/architecture.md: diagrama de flujo, decisiones de diseno documentadas
  (eleccion del dataset, ROC-AUC vs accuracy, umbral 0.3, XGBoost con Optuna)
- docs/api.md: documentacion de todos los endpoints con ejemplos curl y Python
- docs/deployment.md: guia de despliegue local y Docker, estrategia de commits
- docs/monitoring.md: estrategia de monitoreo con Evidently y Grafana"
```

### Commit 12: Tag de version y Release

```bash
git add .
git commit -m "chore: preparar release v1.0.0

- Verificar que todos los tests pasan: uv run pytest tests/unit/
- Verificar que el linting pasa: uv run ruff check src/ tests/
- Actualizar version en pyproject.toml a 1.0.0"

git tag -a v1.0.0 -m "Release v1.0.0: proyecto MLOps completo

Pipeline end-to-end para prediccion de reingreso hospitalario:
- Dataset: UCI Diabetes 130-US Hospitals (101,766 registros)
- Modelo: XGBoost optimizado con Optuna (20 trials)
- Tracking: MLflow 3.x con Model Registry
- Orquestacion: Prefect 3.x
- API: FastAPI con Docker
- CI/CD: GitHub Actions"

git push origin main --tags
```

---

## Convenciones de Mensajes de Commit

Seguir el estandar Conventional Commits (https://www.conventionalcommits.org):

```
<tipo>(<scope>): <descripcion corta en imperativo>

<cuerpo opcional>
```

| Tipo      | Cuando usarlo                                        |
|-----------|------------------------------------------------------|
| feat      | Nueva funcionalidad                                  |
| fix       | Correccion de bug                                    |
| docs      | Cambios solo en documentacion                        |
| test      | Agregar o corregir tests                             |
| refactor  | Refactorizacion sin cambio de funcionalidad          |
| ci        | Cambios en CI/CD o configuracion de pipelines        |
| chore     | Tareas de mantenimiento, dependencias, version       |
