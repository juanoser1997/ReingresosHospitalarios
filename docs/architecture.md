# Arquitectura del Proyecto

## Diagrama de Flujo MLOps

```
[UCI ML Repository]
        |
        v
[src/data/download.py]
        |  data/raw/diabetic_data.csv (101,766 filas)
        v
[src/features/engineering.py]
        |  +7 features clinicas derivadas
        v
[src/data/preprocess.py]
        |  encoding, imputacion, split estratificado 80/20
        v
[src/models/train.py] <-----> [MLflow Server :5000]
        |  Optuna: 20 trials XGBoost        |
        |  Cada trial = run hijo MLflow      |
        v                                    |
[MLflow Model Registry] <-------------------+
        |  Staging -> Production (manual)
        v
[src/api/main.py - FastAPI :8000]
        |  /health  /predict  /predict/batch
        v
[Docker Container]
        |
        v
[Sistema de informacion hospitalario / EHR]
        |
        v
[src/monitoring/drift.py]
        |  Comparacion semanal con Evidently
        v
[Alerta -> Reentrenamiento via Prefect :4200]
```

---

## Decisiones de Diseno

### Eleccion del Dataset

El dataset **Diabetes 130-US Hospitals** (UCI, 1999-2008) fue elegido por:

- Relevancia clinica directa: prediccion de reingreso en 30 dias, indicador
  regulado en Colombia (Resolucion 256 de 2016 del Ministerio de Salud)
- Tamano adecuado para MLOps (101,766 registros) sin requerir infraestructura costosa
- Desbalance de clases real (~11% positivos) que obliga a tomar decisiones
  tecnicas informadas (scale_pos_weight, umbral ajustado, metricas apropiadas)
- Variables heterogeneas: demograficas, clinicas, farmacologicas y administrativas
  que justifican un pipeline de preprocesamiento no trivial
- Dataset original de un paper academico con outliers documentados
  (Strack et al., 2014)

### Por que ROC-AUC como Metrica Principal

En un contexto clinico con clases desbalanceadas:
- La accuracy seria engannosa (un modelo que siempre predice 0 tendria 89% de accuracy)
- ROC-AUC evalua la discriminacion del modelo sin depender del umbral
- Permite ajustar el threshold segun las prioridades clinicas (costo de falsos negativos
  vs. carga operativa por falsos positivos)

El umbral por defecto es **0.3** (no 0.5) para priorizar el recall:
un paciente de alto riesgo no identificado tiene consecuencias mas graves
que una intervencion innecesaria.

### Por que XGBoost con Optuna

- XGBoost supera consistentemente a Random Forest en datasets tabulares clinicos
  con variables categoricas de alta cardinalidad
- Optuna con 20 trials permite una busqueda eficiente con Tree-structured
  Parzen Estimator (TPE), superior al grid search
- scale_pos_weight es el hiperparametro mas critico para el desbalance; Optuna
  lo busca en el rango [5.0, 15.0] segun la proporcion real de clases
- early_stopping_rounds previene overfitting sin necesidad de fijar n_estimators
  manualmente

### Por que Prefect 3.x

- API de Python puro con decoradores @task y @flow sin YAML de DAGs
- Retry automatico en la tarea de descarga de datos (servicio externo UCI)
- Caching de resultados evita re-ejecutar pasos costosos si no cambian los inputs
- Artefactos de Markdown se generan automaticamente al final del pipeline
- Compatible con scheduling tipo cron para reentrenamiento periodico

### Estrategia de Imputacion

- **?** en el dataset original se trata como nulo (pandas na_values="?")
- Variables categoricas con nulos: imputar con "Unknown" en lugar de la moda,
  para que el modelo aprenda a tratar la ausencia de informacion como categoria
  propia (comun en datos clinicos administrativos)
- weight (97% nulos) y payer_code (40% nulos): se eliminan completamente

### Por que Umbral 0.3 en Produccion

El umbral de decision afecta directamente la operacion del equipo de salud:
- Con threshold=0.5: se identifican ~40% de los reingresos reales
- Con threshold=0.3: se identifican ~65% de los reingresos (objetivo de recall)
- El costo adicional: ~15% mas de pacientes marcados como alto riesgo
- Este intercambio es aceptable si el costo de la intervencion es bajo
  (llamada de seguimiento) vs. el costo del reingreso (hospitalizacion completa)

### Limitaciones Conocidas

- El modelo fue entrenado con datos de hospitales de EE.UU. entre 1999-2008.
  La generalizacion a Colombia requiere validacion con datos locales.
- Los codigos ICD-9 de diagnostico (diag_1, diag_2, diag_3) fueron encodificados
  con LabelEncoder. En produccion real se recomienda usar embeddings o
  jerarquia CIE-10.
- El pipeline no implementa re-entrenamiento automatico. El equipo debe
  revisar el reporte de drift semanal y decidir manualmente.
