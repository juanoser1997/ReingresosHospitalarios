# Estrategia de Monitoreo

---

## Objetivo

Detectar de forma temprana cualquier degradacion en el rendimiento del modelo
de prediccion de reingreso hospitalario, antes de que impacte la calidad
de las decisiones clinicas.

---

## Categorias de Monitoreo

### 1. Monitoreo de Infraestructura (tiempo real)

**Metricas:**
- Latencia del endpoint `/predict` (p50, p95, p99)
- Tasa de errores HTTP (4xx y 5xx) por minuto
- Uso de CPU y memoria del contenedor Docker
- Disponibilidad del servicio (uptime > 99.5%)

**Implementacion:**

```python
# Agregar a src/api/main.py
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

Con Prometheus + Grafana:
- Prometheus hace scraping de `http://api:8000/metrics` cada 15 segundos
- Grafana visualiza los paneles y envia alertas si latencia p95 > 500ms

**Alertas inmediatas:**
- API sin responder por mas de 2 minutos
- Tasa de errores 5xx superior al 5% en una ventana de 5 minutos

---

### 2. Monitoreo de Datos de Entrada (semanal)

**Que monitorear:** cambios en la distribucion de las features clinicas
de los pacientes que reciben predicciones en produccion vs. el conjunto
de entrenamiento.

**Herramienta:** Evidently AI (`src/monitoring/drift.py`)

**Features prioritarias para monitorear:**

| Feature                | Razon de importancia                                    |
|------------------------|---------------------------------------------------------|
| age                    | Cambios demograficos en la poblacion atendida           |
| time_in_hospital       | Cambios en protocolos de hospitalizacion                |
| num_medications        | Cambios en practicas farmacologicas                     |
| number_diagnoses       | Cambios en codificacion clinica                         |
| number_inpatient       | Cambios en la complejidad de los pacientes              |
| A1Cresult              | Cambios en protocolos de medicion de HbA1c              |
| insulin                | Cambios en guias de manejo de insulina                  |

**Proceso:**
1. Cada semana se acumulan los features de todos los requests a `/predict`
2. Se corre `compute_drift_report()` comparando con `data/processed/X_train.csv`
3. Si `drift_share > 0.30` (mas del 30% de features con drift), se genera alerta
4. El equipo revisa el reporte HTML en `logs/drift_reports/drift_report.html`

**Umbral de accion:** Si el drift persiste dos semanas consecutivas, programar
reentrenamiento con datos mas recientes.

---

### 3. Monitoreo del Modelo (mensual con retardo)

**El problema del retardo:** Para saber si la prediccion fue correcta, se
necesita esperar a que pasen 30 dias desde el alta para confirmar si el
paciente reingreso.

**Proceso de evaluacion retardada:**

1. Al hacer una prediccion, guardar en base de datos:
   ```
   {encounter_id, timestamp, features_hash, predicted_probability, predicted_label}
   ```

2. A los 35 dias, cruzar con el sistema de informacion hospitalario para
   obtener el resultado real (reingreso o no)

3. Calcular las metricas del modelo en el periodo:
   - ROC-AUC mensual
   - Recall mensual
   - Comparar con las metricas del periodo de entrenamiento

**Proxy metrics (cuando no hay ground truth disponible):**

Si el sistema hospitalario no reporta reingresos de inmediato, usar:
- Distribucion de probabilidades predichas: si la media cambia mas del 15%
  respecto al periodo anterior, puede indicar degradacion del modelo
- Comparar el porcentaje de pacientes marcados como "ALTO riesgo" semana a semana

---

### 4. Monitoreo de Datos de Entrada en Tiempo Real

Validaciones implementadas en la API (Pydantic):
- Valores fuera del rango clinico documentado (ej: age > 100, time_in_hospital > 14)
- Combinaciones imposibles (ej: admission_type_id invalido)

Validaciones adicionales recomendadas en produccion:
```python
# Ejemplo: flag combinaciones clinicamente inusuales
if features.num_medications > 50 and features.time_in_hospital < 2:
    logger.warning(
        "Combinacion inusual: %d medicamentos en %d dias",
        features.num_medications,
        features.time_in_hospital,
    )
```

---

## Stack de Monitoreo Propuesto

```
FastAPI (/metrics)
    |
    v
Prometheus          <-- scraping cada 15s
    |
    v
Grafana             <-- dashboards + alertas
    |
    v
PagerDuty / Slack   <-- notificaciones al equipo

Predicciones        --> PostgreSQL (con timestamp y encounter_id)
    |
    v
Script semanal      --> compute_drift_report() con Evidently
    |
    v
Reporte HTML        --> logs/drift_reports/

Sistema Hospitalario --> Cruce mensual de predicciones vs. realidad
    |
    v
Metricas del modelo --> MLflow (nuevo run de evaluacion mensual)
```

---

## Dashboard de Grafana (Propuesto)

| Panel                           | Visualizacion       | Frecuencia  |
|---------------------------------|---------------------|-------------|
| Latencia de la API (p50/p95/p99)| Time series         | Tiempo real |
| Requests por minuto y errores   | Time series         | Tiempo real |
| Distribucion de probabilidades  | Histograma          | Diario      |
| % de pacientes ALTO riesgo      | Gauge               | Diario      |
| Drift share por feature         | Bar chart           | Semanal     |
| ROC-AUC del modelo en produccion| Time series         | Mensual     |
| Recall mensual vs. baseline     | Comparacion lineas  | Mensual     |

---

## Criterios de Reentrenamiento

Trigger automatico si se cumple alguna de estas condiciones:

| Condicion                                          | Severidad | Accion                          |
|----------------------------------------------------|-----------|----------------------------------|
| drift_share > 0.30 por 2 semanas consecutivas      | Alta      | Programar reentrenamiento        |
| ROC-AUC produccion < 0.62 (caida de 10%)          | Critica   | Reentrenamiento urgente          |
| Recall produccion < 0.50                           | Critica   | Reentrenamiento urgente          |
| Mas de 90 dias desde el ultimo entrenamiento       | Media     | Reentrenamiento preventivo       |
| Cambio en guias clinicas de manejo de diabetes     | Alta      | Reentrenamiento + validacion     |

**Proceso de reentrenamiento:**
1. El equipo aprueba el reentrenamiento
2. Se ejecuta `uv run python flows/training_pipeline.py`
3. El nuevo modelo queda en **Staging** en MLflow
4. Se valida manualmente con un conjunto de holdout reciente
5. Si las metricas mejoran, se promueve a **Production**
6. El modelo anterior pasa a **Archived**
