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


def test_create_weighing(client):
    response = client.post("/api/weighings", json={"weight_kg": 487.4, "confidence": 0.92, "comprimento_m": 1.43, "altura_m": 1.31, "largura_m": 0.49, "area_m2": 1.92, "perimetro_m": 1.84})
    assert response.status_code == 201
    data = response.json()
    assert data["weight_kg"] == 487.4
    assert data["id"] is not None


def test_list_weighings(client):
    client.post("/api/weighings", json={"weight_kg": 400.0})
    client.post("/api/weighings", json={"weight_kg": 500.0})
    response = client.get("/api/weighings")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_weighing_by_id(client):
    create_resp = client.post("/api/weighings", json={"weight_kg": 450.0})
    weighing_id = create_resp.json()["id"]
    response = client.get(f"/api/weighings/{weighing_id}")
    assert response.status_code == 200
    assert response.json()["weight_kg"] == 450.0


def test_get_weighing_not_found(client):
    response = client.get("/api/weighings/999")
    assert response.status_code == 404


def test_list_weighings_with_limit(client):
    for i in range(5):
        client.post("/api/weighings", json={"weight_kg": 400 + i * 10})
    response = client.get("/api/weighings?limit=3")
    assert response.status_code == 200
    assert len(response.json()) == 3
