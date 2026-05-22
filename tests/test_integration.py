"""
End-to-end integration test.

Simulates the full pipeline flow:
1. API is running (test client)
2. Pipeline processes a frame (simulated via direct POST)
3. Weighing appears in API
4. Reports reflect the new data

Fixture 'client' from tests/conftest.py.
"""


def test_full_flow(client, tmp_path):
    """Test the complete pipeline -> API -> reports flow."""

    # Step 1: Verify API is running
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

    # Step 2: Verify empty state
    summary = client.get("/api/reports/summary").json()
    assert summary["total_weighings"] == 0

    # Step 3: Simulate pipeline sending a weighing
    weighing_data = {
        "weight_kg": 487.4,
        "confidence": 0.92,
        "comprimento_m": 1.43,
        "altura_m": 1.31,
        "largura_m": 0.49,
        "area_m2": 1.92,
        "perimetro_m": 1.84,
        "evidence_path": str(tmp_path / "evidence" / "123456.jpg"),
    }
    response = client.post("/api/weighings", json=weighing_data)
    assert response.status_code == 201

    # Step 4: Verify weighing appears in list
    weighings = client.get("/api/weighings").json()
    assert len(weighings) == 1
    assert weighings[0]["weight_kg"] == 487.4

    # Step 5: Verify summary updated
    summary = client.get("/api/reports/summary").json()
    assert summary["total_weighings"] == 1
    assert summary["average_weight_kg"] == 487.4

    # Step 6: Create animal and weighings with weight loss for alert
    animal = client.post("/api/animals", json={"rfid": "BOI_E2E"}).json()
    client.post("/api/weighings", json={
        "animal_id": animal["id"], "weight_kg": 500.0
    })
    client.post("/api/weighings", json={
        "animal_id": animal["id"], "weight_kg": 450.0  # -10%
    })

    # Step 7: Verify alert generated
    alerts = client.get("/api/reports/alerts").json()
    assert len(alerts) == 1
    assert alerts[0]["rfid"] == "BOI_E2E"

    # Step 8: Verify weight history has data
    history = client.get("/api/reports/weight-history").json()
    assert len(history) >= 1

    print("Integration test PASSED")
