import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.database import Base
from api.models import Animal, Weighing, Farm


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_create_animal(db_session):
    animal = Animal(rfid="BOI_001", raca="nelore", sexo="M")
    db_session.add(animal)
    db_session.commit()
    result = db_session.query(Animal).first()
    assert result.rfid == "BOI_001"
    assert result.raca == "nelore"
    assert result.created_at is not None


def test_create_weighing_without_animal(db_session):
    weighing = Weighing(weight_kg=487.4, confidence=0.92, comprimento_m=1.43, altura_m=1.31, largura_m=0.49, area_m2=1.92, perimetro_m=1.84, evidence_path="data/evidence/123456.jpg")
    db_session.add(weighing)
    db_session.commit()
    result = db_session.query(Weighing).first()
    assert result.weight_kg == 487.4
    assert result.animal_id is None


def test_create_weighing_with_animal(db_session):
    animal = Animal(rfid="BOI_002", raca="angus", sexo="F")
    db_session.add(animal)
    db_session.commit()
    weighing = Weighing(animal_id=animal.id, weight_kg=391.0, confidence=0.89, comprimento_m=1.28, altura_m=1.19, largura_m=0.41, area_m2=1.61, perimetro_m=1.71)
    db_session.add(weighing)
    db_session.commit()
    result = db_session.query(Weighing).first()
    assert result.animal_id == animal.id
    assert result.animal.rfid == "BOI_002"


def test_create_farm(db_session):
    farm = Farm(name="Fazenda Esperanca", location="Goias, Brasil")
    db_session.add(farm)
    db_session.commit()
    result = db_session.query(Farm).first()
    assert result.name == "Fazenda Esperanca"


def test_rfid_unique_constraint(db_session):
    a1 = Animal(rfid="BOI_DUP", raca="nelore", sexo="M")
    a2 = Animal(rfid="BOI_DUP", raca="angus", sexo="F")
    db_session.add(a1)
    db_session.commit()
    db_session.add(a2)
    with pytest.raises(Exception):
        db_session.commit()
