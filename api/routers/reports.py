"""Reports router — summary, alerts, and weight history endpoints."""
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session
from api.database import get_db
from api.models import Animal, Weighing
from api.schemas import SummaryResponse, AlertResponse, WeightHistoryPoint

router = APIRouter(tags=["reports"])


@router.get("/reports/summary", response_model=SummaryResponse)
def get_summary(db: Session = Depends(get_db)):
    total_weighings = db.query(func.count(Weighing.id)).scalar() or 0
    avg_weight = db.query(func.avg(Weighing.weight_kg)).scalar()
    last_weighing = db.query(Weighing).order_by(Weighing.created_at.desc()).first()
    total_animals = db.query(func.count(Animal.id)).scalar() or 0
    alerts = _get_alerts(db)
    return SummaryResponse(
        total_weighings=total_weighings,
        average_weight_kg=round(avg_weight, 1) if avg_weight else None,
        last_weighing_at=last_weighing.created_at if last_weighing else None,
        total_animals=total_animals,
        active_alerts=len(alerts),
    )


@router.get("/reports/alerts", response_model=list[AlertResponse])
def get_alerts(db: Session = Depends(get_db)):
    return _get_alerts(db)


def _get_alerts(db: Session) -> list[AlertResponse]:
    alerts = []
    animals = db.query(Animal).all()
    for animal in animals:
        weighings = db.query(Weighing).filter(Weighing.animal_id == animal.id).order_by(Weighing.created_at.desc()).limit(2).all()
        if len(weighings) < 2:
            continue
        current = weighings[0]
        previous = weighings[1]
        if previous.weight_kg == 0:
            continue
        change_pct = ((current.weight_kg - previous.weight_kg) / previous.weight_kg) * 100
        if change_pct < -5.0:
            alerts.append(AlertResponse(
                animal_id=animal.id, rfid=animal.rfid,
                current_weight_kg=current.weight_kg, previous_weight_kg=previous.weight_kg,
                change_percent=round(change_pct, 1), last_weighing_at=current.created_at,
            ))
    return alerts


@router.get("/reports/weight-history", response_model=list[WeightHistoryPoint])
def get_weight_history(days: int = 30, db: Session = Depends(get_db)):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    weighings = db.query(Weighing).filter(Weighing.created_at >= cutoff).order_by(Weighing.created_at).all()
    daily: dict[str, list[float]] = {}
    for w in weighings:
        date_str = w.created_at.strftime("%Y-%m-%d") if w.created_at else "unknown"
        if date_str not in daily:
            daily[date_str] = []
        daily[date_str].append(w.weight_kg)
    return [
        WeightHistoryPoint(date=date, average_weight_kg=round(sum(weights) / len(weights), 1), count=len(weights))
        for date, weights in sorted(daily.items())
    ]
