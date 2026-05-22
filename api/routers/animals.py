"""Animals router — CRUD endpoints for cattle records."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from api.database import get_db
from api.models import Animal
from api.schemas import AnimalCreate, AnimalResponse, WeighingResponse

router = APIRouter(tags=["animals"])


class AnimalDetailResponse(AnimalResponse):
    weighings: list[WeighingResponse] = []


@router.post("/animals", response_model=AnimalResponse, status_code=201)
def create_animal(data: AnimalCreate, db: Session = Depends(get_db)):
    animal = Animal(**data.model_dump())
    db.add(animal)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail=f"Animal with RFID '{data.rfid}' already exists")
    db.refresh(animal)
    return animal


@router.get("/animals", response_model=list[AnimalResponse])
def list_animals(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    return db.query(Animal).order_by(Animal.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/animals/{animal_id}", response_model=AnimalDetailResponse)
def get_animal(animal_id: int, db: Session = Depends(get_db)):
    animal = db.query(Animal).filter(Animal.id == animal_id).first()
    if not animal:
        raise HTTPException(status_code=404, detail="Animal not found")
    return animal
