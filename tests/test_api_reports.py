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


def test_summary_empty(client):
    response = client.get("/api/reports/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_weighings"] == 0
    assert data["average_weight_kg"] is None


def test_summary_with_data(client):
    client.post("/api/weighings", json={"weight_kg": 400.0})
    client.post("/api/weighings", json={"weight_kg": 500.0})
    response = client.get("/api/reports/summary")
    data = response.json()
    assert data["total_weighings"] == 2
    assert data["average_weight_kg"] == 450.0


def test_alerts_no_alerts(client):
    response = client.get("/api/reports/alerts")
    assert response.status_code == 200
    assert response.json() == []


def test_alerts_weight_loss(client):
    animal_resp = client.post("/api/animals", json={"rfid": "BOI_ALERT"})
    animal_id = animal_resp.json()["id"]
    client.post("/api/weighings", json={"animal_id": animal_id, "weight_kg": 500.0})
    client.post("/api/weighings", json={"animal_id": animal_id, "weight_kg": 460.0})
    response = client.get("/api/reports/alerts")
    data = response.json()
    assert len(data) >= 1
    assert data[0]["rfid"] == "BOI_ALERT"
    assert data[0]["change_percent"] < -5.0


def test_weight_history(client):
    client.post("/api/weighings", json={"weight_kg": 400.0})
    client.post("/api/weighings", json={"weight_kg": 500.0})
    response = client.get("/api/reports/weight-history")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert "date" in data[0]
    assert "average_weight_kg" in data[0]
