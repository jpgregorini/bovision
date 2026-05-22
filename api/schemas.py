"""Pydantic schemas for API request/response validation."""
from datetime import datetime
from pydantic import BaseModel


class AnimalCreate(BaseModel):
    rfid: str
    raca: str | None = None
    sexo: str | None = None


class AnimalResponse(BaseModel):
    id: int
    rfid: str
    raca: str | None
    sexo: str | None
    created_at: datetime
    model_config = {"from_attributes": True}


class WeighingCreate(BaseModel):
    animal_id: int | None = None
    weight_kg: float
    confidence: float | None = None
    comprimento_m: float | None = None
    altura_m: float | None = None
    largura_m: float | None = None
    area_m2: float | None = None
    perimetro_m: float | None = None
    evidence_path: str | None = None


class WeighingResponse(BaseModel):
    id: int
    animal_id: int | None
    weight_kg: float
    confidence: float | None
    comprimento_m: float | None
    altura_m: float | None
    largura_m: float | None
    area_m2: float | None
    perimetro_m: float | None
    evidence_path: str | None
    created_at: datetime
    model_config = {"from_attributes": True}


class SummaryResponse(BaseModel):
    total_weighings: int
    average_weight_kg: float | None
    last_weighing_at: datetime | None
    total_animals: int
    active_alerts: int


class AlertResponse(BaseModel):
    animal_id: int
    rfid: str
    current_weight_kg: float
    previous_weight_kg: float
    change_percent: float
    last_weighing_at: datetime


class WeightHistoryPoint(BaseModel):
    date: str
    average_weight_kg: float
    count: int
