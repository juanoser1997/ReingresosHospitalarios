# Documentacion de la API

La API esta construida con FastAPI. La documentacion interactiva se genera
automaticamente en:

- Swagger UI: http://localhost:8000/docs
- ReDoc:      http://localhost:8000/redoc

---

## Endpoints

### GET /health

Verifica que la API esta activa y el modelo esta cargado en memoria.

**Respuesta 200:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true
}
```

Si el modelo no pudo cargarse desde MLflow:
```json
{
  "status": "degraded",
  "version": "1.0.0",
  "model_loaded": false
}
```

---

### POST /predict

Predice la probabilidad de reingreso hospitalario en menos de 30 dias para un paciente.

**Request body:**
```json
{
  "age": 72,
  "gender": "Female",
  "admission_type_id": 1,
  "discharge_disposition_id": 1,
  "admission_source_id": 7,
  "time_in_hospital": 5,
  "num_lab_procedures": 44,
  "num_procedures": 1,
  "num_medications": 14,
  "number_outpatient": 0,
  "number_emergency": 1,
  "number_inpatient": 1,
  "number_diagnoses": 9,
  "A1Cresult": ">8",
  "insulin": "Up",
  "diabetesMed": "Yes",
  "change": "Ch"
}
```

**Campos requeridos:**

| Campo                    | Tipo    | Descripcion                               | Valores validos         |
|--------------------------|---------|-------------------------------------------|-------------------------|
| age                      | integer | Edad del paciente                         | 0 - 100                 |
| admission_type_id        | integer | Tipo de admision                          | 1-8 (1=Emergencia)      |
| discharge_disposition_id | integer | Tipo de alta                              | 1-28 (1=Alta domicilio) |
| admission_source_id      | integer | Fuente de admision                        | 1-26 (7=Emergencia)     |
| time_in_hospital         | integer | Dias de hospitalizacion                   | 1 - 14                  |
| num_lab_procedures       | integer | Procedimientos de laboratorio             | 0 - 132                 |
| num_procedures           | integer | Procedimientos no laboratorio             | 0 - 6                   |
| num_medications          | integer | Medicamentos distintos administrados      | 0 - 81                  |
| number_diagnoses         | integer | Diagnosticos ingresados                   | 1 - 16                  |

**Campos opcionales (con defaults):**

| Campo           | Default   | Descripcion                               |
|-----------------|-----------|-------------------------------------------|
| gender          | Unknown   | Genero (Male, Female, Unknown)            |
| number_outpatient | 0       | Visitas ambulatorias ultimo ano           |
| number_emergency  | 0       | Visitas urgencias ultimo ano              |
| number_inpatient  | 0       | Hospitalizaciones ultimo ano              |
| A1Cresult       | None      | Resultado HbA1c (>8, >7, Norm, None)     |
| insulin         | No        | Cambio insulina (No, Steady, Up, Down)    |
| diabetesMed     | Yes       | Medicacion para diabetes (Yes, No)        |
| change          | No        | Cambio en medicacion (Ch, No)             |

**Respuesta 200:**
```json
{
  "readmission_probability": 0.4231,
  "high_risk": true,
  "risk_level": "ALTO",
  "threshold_used": 0.3,
  "model_name": "diabetes-readmission-xgboost",
  "model_stage": "Production",
  "recommendation": "INTERVENIR ANTES DEL ALTA. Solicitar evaluacion por gestion de casos cronicos. Coordinar visita domiciliaria en las primeras 48 horas."
}
```

**Niveles de riesgo:**

| Nivel | Probabilidad | Accion                                             |
|-------|--------------|-----------------------------------------------------|
| BAJO  | < 0.20       | Seguimiento ambulatorio estandar (cita en 30 dias)  |
| MEDIO | 0.20 - 0.39  | Llamada de seguimiento a las 72 horas del alta      |
| ALTO  | >= 0.40      | Intervencion antes del alta + visita domiciliaria   |

**Errores:**

| Codigo | Descripcion                                           |
|--------|-------------------------------------------------------|
| 422    | Validacion fallida (campo fuera de rango o invalido)  |
| 503    | Modelo no disponible (MLflow no configurado)          |
| 500    | Error interno al generar la prediccion                |

---

### POST /predict/batch

Genera predicciones para multiples pacientes en una sola llamada.
Limite: 200 pacientes por request.

**Request body:**
```json
[
  {
    "patient_id": "P001",
    "features": { ... }
  },
  {
    "patient_id": "P002",
    "features": { ... }
  }
]
```

El campo `patient_id` es opcional. `features` tiene el mismo schema que `/predict`.

**Respuesta 200:**
```json
{
  "predictions": [
    {
      "readmission_probability": 0.4231,
      "high_risk": true,
      "risk_level": "ALTO",
      ...
    },
    {
      "readmission_probability": 0.1102,
      "high_risk": false,
      "risk_level": "BAJO",
      ...
    }
  ],
  "total": 2,
  "high_risk_count": 1
}
```

---

## Ejemplos de Uso

### curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 72,
    "gender": "Female",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 5,
    "num_lab_procedures": 44,
    "num_procedures": 1,
    "num_medications": 14,
    "number_emergency": 1,
    "number_inpatient": 1,
    "number_diagnoses": 9,
    "A1Cresult": ">8",
    "insulin": "Up",
    "diabetesMed": "Yes",
    "change": "Ch"
  }'
```

### Python con httpx

```python
import httpx

payload = {
    "age": 72,
    "gender": "Female",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 5,
    "num_lab_procedures": 44,
    "num_procedures": 1,
    "num_medications": 14,
    "number_emergency": 1,
    "number_inpatient": 1,
    "number_diagnoses": 9,
    "A1Cresult": ">8",
    "insulin": "Up",
    "diabetesMed": "Yes",
    "change": "Ch",
}

response = httpx.post("http://localhost:8000/predict", json=payload)
result = response.json()

print(f"Probabilidad de reingreso: {result['readmission_probability']:.1%}")
print(f"Nivel de riesgo: {result['risk_level']}")
print(f"Recomendacion: {result['recommendation']}")
```

### Batch desde un DataFrame de pandas

```python
import httpx
import pandas as pd

df = pd.read_csv("nuevos_pacientes.csv")

batch_payload = [
    {"patient_id": str(row["encounter_id"]), "features": row.to_dict()}
    for _, row in df.iterrows()
]

response = httpx.post(
    "http://localhost:8000/predict/batch",
    json=batch_payload,
    timeout=30.0,
)
result = response.json()

print(f"Total evaluados: {result['total']}")
print(f"Alto riesgo: {result['high_risk_count']} ({result['high_risk_count']/result['total']:.1%})")
```
