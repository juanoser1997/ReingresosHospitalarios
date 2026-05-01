# MLOps - Prediccion de Riesgo de Reingreso Hospitalario en Pacientes Diabeticos

Universidad de Medellin — Curso MLOps — Proyecto Final

Integrantes : 
Juan Jose Basante Navarro
Luisa Fernanda Franco Giraldo
Juan Mauricio Sanchez Restrepo

---

## Descripcion del Problema

Las enfermedades de alto costo (diabetes, insuficiencia cardiaca, enfermedad renal cronica) generan
un alto volumen de reingresos hospitalarios evitables. En Colombia, el sistema de salud penaliza
a las IPS por reingresos dentro de los 30 dias posteriores al alta, lo que genera un fuerte
incentivo para la prevencion.

Este proyecto implementa un pipeline MLOps completo para predecir si un paciente diabetico
sera reingresado al hospital en menos de 30 dias, permitiendo a los equipos clinicos
intervenir de forma preventiva antes del alta.

**Dataset:** Diabetes 130-US Hospitals (1999-2008)  
**Fuente:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008  
**Tarea:** Clasificacion binaria — reingreso en menos de 30 dias: Si / No  
**Filas:** 101,766 encuentros hospitalarios  
**Columnas:** 50 variables clinicas y administrativas  

---

## Contexto de Negocio

Una IPS en Colombia necesita reducir su tasa de reingreso en 30 dias del 11% actual al 7%.
El modelo permite:

- Identificar pacientes de alto riesgo antes del alta
- Priorizar visitas de seguimiento del equipo de enfermeria
- Optimizar la asignacion de recursos de gestion de casos cronicos
- Apoyar el cumplimiento de indicadores de calidad de la Resolucion 256 de 2016 del MinSalud

---

## Stack Tecnologico

| Componente          | Tecnologia            |
|---------------------|-----------------------|
| Lenguaje            | Python 3.11           |
| Experiment Tracking | MLflow 3.x            |
| Optimizacion        | Optuna 4.x            |
| Orquestacion        | Prefect 3.x (*)       |
| API                 | FastAPI + Uvicorn     |
| Contenerizacion     | Docker + Compose      |
| Modelo              | XGBoost               |

(*) Prefect fue incluido como parte del contenido del curso para entender los conceptos
de orquestacion de pipelines. Sin embargo, no fue utilizado en el proceso de entrenamiento
real porque la complejidad del problema no lo justificaba — el pipeline es lo suficientemente
lineal para manejarse con scripts Python orquestados directamente desde Docker Compose.

---

## Estructura del Repositorio

```
mlops-alto-costo/
├── src/
│   ├── data/
│   │   ├── download.py              # Descarga desde UCI
│   │   └── preprocess.py            # Limpieza, encoding, split
│   ├── features/
│   │   └── engineering.py           # Features clinicas derivadas
│   ├── models/
│   │   ├── train.py                 # Entrenamiento + MLflow + Optuna
│   │   └── predict.py               # Inferencia desde MLflow Registry
│   └── api/
│       ├── main.py                  # FastAPI app
│       ├── schemas.py               # Esquemas de datos
│       └── ui.py                    # Interfaz web
├── scripts/
│   ├── wait_for_mlflow.py           # Espera a que MLflow este disponible
│   ├── run_training_if_needed.py    # Orquestador del pipeline de entrenamiento
│   └── promote_model.py             # Promocion del modelo a Production
├── flows/
│   └── training_pipeline.py         # Prefect flow (referencia conceptual)
├── configs/
│   └── model_config.yaml
├── data/
│   ├── raw/                         # No versionado en git
│   └── processed/                   # No versionado en git
├── logs/
├── Dockerfile
└── docker-compose.yml
```

---

## Requisitos

- **Docker Desktop** instalado y corriendo
- **Git** para clonar el repositorio
- Conexion a internet (para descargar el dataset desde UCI al primer arranque)

No se necesita Python local ni ninguna dependencia adicional — todo corre dentro de los contenedores.

---

## Inicio Rapido

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu-usuario>/mlops-alto-costo.git
cd mlops-alto-costo
```

### 2. Levantar el stack completo

```bash
docker compose up --build
```

Esto ejecuta automaticamente en orden:

1. **MLflow** arranca como servidor de tracking y registro de modelos
2. **Trainer** descarga el dataset, preprocesa, entrena con Optuna (50 trials por defecto) y promueve el modelo a Production
3. **API** arranca una vez el modelo esta disponible

El primer arranque tarda aproximadamente 30-45 minutos dependiendo de los recursos disponibles
(la mayor parte del tiempo es el entrenamiento con Optuna).

### 3. Acceder a los servicios

| Servicio       | URL                        | Descripcion                        |
|----------------|----------------------------|------------------------------------|
| API / UI       | http://localhost:8000      | Interfaz web de prediccion         |
| Documentacion  | http://localhost:8000/docs | Swagger con todos los endpoints    |
| MLflow         | http://localhost:5000      | Experimentos y modelo registrado   |

Al abrir http://localhost:8000 se redirige automaticamente a la interfaz de prediccion.

---

## Ejecucion Limpia

Si se quiere borrar todo y comenzar desde cero:

```bash
docker compose down -v --rmi all
docker compose up --build
```

---

## Configuracion Opcional

El numero de trials de Optuna puede ajustarse como variable de entorno.
El valor por defecto es 50, que ofrece un buen balance entre tiempo y calidad del modelo:

```bash
# Entrenamiento rapido para pruebas
N_TRIALS=10 docker compose up --build

# Entrenamiento exhaustivo
N_TRIALS=100 docker compose up --build
```

---

## Dataset: Diabetes 130-US Hospitals

| Variable           | Descripcion                                        |
|--------------------|----------------------------------------------------|
| age                | Edad del paciente                                  |
| time_in_hospital   | Dias de hospitalizacion (1-14)                     |
| num_medications    | Medicamentos distintos recetados                   |
| num_lab_procedures | Examenes de laboratorio realizados                 |
| number_diagnoses   | Diagnosticos registrados en el sistema             |
| A1Cresult          | Resultado de hemoglobina glucosilada (HbA1c)       |
| insulin            | Manejo de insulina durante la hospitalizacion      |
| diabetesMed        | Se receto medicamento para diabetes                |
| readmitted         | Reingreso: menos de 30 dias, mas de 30, o ninguno  |

**Balance de clases:** aproximadamente 11% de casos positivos (reingreso en menos de 30 dias).

---

## Features de Ingenieria Clinica

Ademas de las variables originales, el modelo incluye variables derivadas basadas en
evidencia clinica:

| Feature                  | Descripcion                                                   |
|--------------------------|---------------------------------------------------------------|
| total_prior_visits       | Suma de consultas, urgencias y hospitalizaciones previas      |
| polypharmacy             | Indicador de uso de 10 o mas medicamentos distintos           |
| lab_procedures_per_day   | Ratio de examenes por dia de hospitalizacion                  |
| comorbidity_score        | Puntaje de comorbilidad segun numero de diagnosticos          |
| readmission_risk_score   | Score compuesto inspirado en el indice LACE                   |
| medication_intensity     | Medicamentos por dia de hospitalizacion                       |
| had_prior_inpatient      | Indicador de hospitalizacion previa en el ultimo ano          |
| high_complexity          | Indicador de 3 o mas factores de riesgo simultaneos           |

---

## Metricas del Modelo

La metrica de optimizacion es una combinacion ponderada de AUC y Recall:

```
score = 0.6 * AUC + 0.4 * Recall
```

Se prioriza el Recall porque en el contexto clinico es mas costoso no identificar
un paciente que va a reingresar (falso negativo) que generar una alerta innecesaria
(falso positivo).

| Metrica   | Objetivo |
|-----------|----------|
| ROC-AUC   | > 0.72   |
| Recall    | > 0.65   |
| Precision | > 0.25   |
| F1-Score  | > 0.35   |

---

## Flujo del Pipeline

```
UCI Repository
      |
      v
src/data/download.py          --> data/raw/diabetic_data.csv
      |
      v
src/data/preprocess.py        --> data/processed/
      |
      v
src/features/engineering.py   --> features clinicas derivadas
      |
      v
src/models/train.py           --> MLflow Tracking + Optuna
      |
      v
MLflow Model Registry         --> Production
      |
      v
src/api/main.py (FastAPI)     --> interfaz web + endpoints REST
      |
      v
Docker Compose                --> orquestacion de los tres servicios
```
