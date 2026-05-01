"""
Schemas Pydantic v2 para la API de prediccion de reingreso hospitalario.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """
    Features clinicas y administrativas de un encuentro hospitalario.
    Basadas en el dataset Diabetes 130-US Hospitals (UCI, 2014).
    """

    # Demograficas
    age: int = Field(..., ge=0, le=100, description="Edad del paciente en anos")
    gender: Literal["Male", "Female", "Unknown"] = Field("Unknown", description="Genero")

    # Administrativas
    admission_type_id: int = Field(
        ..., ge=1, le=8, description="Tipo de admision (1=Emergencia, 2=Urgente, 3=Electiva)"
    )
    discharge_disposition_id: int = Field(
        ..., ge=1, le=28, description="Tipo de alta (1=Alta a domicilio)"
    )
    admission_source_id: int = Field(
        ..., ge=1, le=26, description="Fuente de admision (7=Emergencia)"
    )

    # Clinicas de la hospitalizacion actual
    time_in_hospital: int = Field(
        ..., ge=1, le=14, description="Dias de hospitalizacion"
    )
    num_lab_procedures: int = Field(
        ..., ge=0, le=132, description="Numero de procedimientos de laboratorio"
    )
    num_procedures: int = Field(
        ..., ge=0, le=6, description="Numero de procedimientos no laboratorio"
    )
    num_medications: int = Field(
        ..., ge=0, le=81, description="Numero de medicamentos distintos"
    )
    number_outpatient: int = Field(
        0, ge=0, description="Visitas ambulatorias en el ultimo ano"
    )
    number_emergency: int = Field(
        0, ge=0, description="Visitas a urgencias en el ultimo ano"
    )
    number_inpatient: int = Field(
        0, ge=0, description="Hospitalizaciones en el ultimo ano"
    )
    number_diagnoses: int = Field(
        ..., ge=1, le=16, description="Numero de diagnosticos ingresados"
    )

    # Control glucemico
    A1Cresult: Literal[">8", ">7", "Norm", "None"] = Field(
        "None", description="Resultado de HbA1c"
    )
    insulin: Literal["No", "Steady", "Up", "Down"] = Field(
        "No", description="Cambio en dosis de insulina"
    )
    diabetesMed: Literal["Yes", "No"] = Field(
        "Yes", description="Se prescribio medicamento para diabetes"
    )
    change: Literal["Ch", "No"] = Field(
        "No", description="Cambio en medicacion diabetica"
    )

    class Config:
        json_schema_extra = {
            "example": {
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
                "change": "Ch",
            }
        }


class PredictionResponse(BaseModel):
    """Respuesta de la API con la prediccion de reingreso."""

    readmission_probability: float = Field(
        ..., description="Probabilidad de reingreso en menos de 30 dias (0-1)"
    )
    high_risk: bool = Field(
        ..., description="True si el paciente es de alto riesgo segun el umbral"
    )
    risk_level: Literal["BAJO", "MEDIO", "ALTO"] = Field(
        ..., description="Nivel de riesgo clinico"
    )
    threshold_used: float = Field(..., description="Umbral de decision aplicado")
    model_name: str
    model_stage: str
    recommendation: str = Field(
        ..., description="Recomendacion clinica basada en el nivel de riesgo"
    )


class BatchPredictionItem(BaseModel):
    """Item individual dentro de una prediccion en batch."""
    patient_id: Optional[str] = Field(None, description="Identificador del paciente")
    features: PatientFeatures


class BatchPredictionResponse(BaseModel):
    """Respuesta para predicciones en batch."""
    predictions: list[PredictionResponse]
    total: int
    high_risk_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
