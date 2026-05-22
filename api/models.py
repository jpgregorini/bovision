"""SQLAlchemy ORM models."""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from api.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class Animal(Base):
    __tablename__ = "animals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    rfid = Column(String, unique=True, nullable=False, index=True)
    raca = Column(String, nullable=True)
    sexo = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    weighings = relationship("Weighing", back_populates="animal")


class Weighing(Base):
    __tablename__ = "weighings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    animal_id = Column(Integer, ForeignKey("animals.id"), nullable=True)
    weight_kg = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    comprimento_m = Column(Float, nullable=True)
    altura_m = Column(Float, nullable=True)
    largura_m = Column(Float, nullable=True)
    area_m2 = Column(Float, nullable=True)
    perimetro_m = Column(Float, nullable=True)
    evidence_path = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    animal = relationship("Animal", back_populates="weighings")


class Farm(Base):
    __tablename__ = "farms"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    location = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
