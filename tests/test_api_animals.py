import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from api.database import Base, get_db
from api.main import app


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_create_animal(client):
    response = client.post("/api/animals", json={"rfid": "BOI_001", "raca": "nelore", "sexo": "M"})
    assert response.status_code == 201
    assert response.json()["rfid"] == "BOI_001"


def test_list_animals(client):
    client.post("/api/animals", json={"rfid": "BOI_001"})
    client.post("/api/animals", json={"rfid": "BOI_002"})
    response = client.get("/api/animals")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_animal_with_weighings(client):
    animal_resp = client.post("/api/animals", json={"rfid": "BOI_010", "raca": "angus"})
    animal_id = animal_resp.json()["id"]
    client.post("/api/weighings", json={"animal_id": animal_id, "weight_kg": 400.0})
    client.post("/api/weighings", json={"animal_id": animal_id, "weight_kg": 420.0})
    response = client.get(f"/api/animals/{animal_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["rfid"] == "BOI_010"
    assert len(data["weighings"]) == 2


def test_get_animal_not_found(client):
    response = client.get("/api/animals/999")
    assert response.status_code == 404


def test_duplicate_rfid(client):
    client.post("/api/animals", json={"rfid": "BOI_DUP"})
    response = client.post("/api/animals", json={"rfid": "BOI_DUP"})
    assert response.status_code == 409
