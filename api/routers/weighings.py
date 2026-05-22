"""Weighings router — CRUD endpoints for cattle weighing records."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from api.database import get_db
from api.models import Weighing
from api.schemas import WeighingCreate, WeighingResponse

router = APIRouter(tags=["weighings"])


@router.post("/weighings", response_model=WeighingResponse, status_code=201)
def create_weighing(data: WeighingCreate, db: Session = Depends(get_db)):
    weighing = Weighing(**data.model_dump())
    db.add(weighing)
    db.commit()
    db.refresh(weighing)
    return weighing


@router.get("/weighings", response_model=list[WeighingResponse])
def list_weighings(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    return db.query(Weighing).order_by(Weighing.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/weighings/{weighing_id}", response_model=WeighingResponse)
def get_weighing(weighing_id: int, db: Session = Depends(get_db)):
    weighing = db.query(Weighing).filter(Weighing.id == weighing_id).first()
    if not weighing:
        raise HTTPException(status_code=404, detail="Weighing not found")
    return weighing
