"""
Seed the database with sample data for manual testing.

Usage:
    python scripts/seed_data.py [--api-url http://localhost:8000]
"""

import argparse
import random
import requests

BREEDS = ["nelore", "angus", "hereford", "brahman", "guzerá"]


def seed(api_url: str):
    print(f"Seeding {api_url}...\n")

    # Create animals
    animals = []
    for i in range(10):
        data = {
            "rfid": f"BOI_{i+1:03d}",
            "raca": random.choice(BREEDS),
            "sexo": random.choice(["M", "F"]),
        }
        resp = requests.post(f"{api_url}/api/animals", json=data)
        if resp.status_code == 201:
            animals.append(resp.json())
            print(f"  Animal: {data['rfid']} ({data['raca']})")
        else:
            print(f"  Skip {data['rfid']}: {resp.status_code}")

    # Create weighings (3 per animal, simulating weight gain/loss)
    for animal in animals:
        base_weight = random.uniform(350, 550)
        for j in range(3):
            change = random.uniform(-20, 30)
            weight = base_weight + change * j
            data = {
                "animal_id": animal["id"],
                "weight_kg": round(weight, 1),
                "confidence": round(random.uniform(0.8, 0.98), 2),
                "comprimento_m": round(random.uniform(1.2, 1.6), 2),
                "altura_m": round(random.uniform(1.1, 1.4), 2),
                "largura_m": round(random.uniform(0.35, 0.55), 2),
                "area_m2": round(random.uniform(1.4, 2.2), 2),
                "perimetro_m": round(random.uniform(1.5, 2.0), 2),
            }
            resp = requests.post(f"{api_url}/api/weighings", json=data)
            if resp.status_code == 201:
                print(f"  Weighing: {animal['rfid']} = {data['weight_kg']} kg")

    # Create one animal with clear weight loss (for alert testing)
    alert_animal = requests.post(f"{api_url}/api/animals", json={
        "rfid": "BOI_ALERT", "raca": "nelore", "sexo": "M"
    }).json()
    requests.post(f"{api_url}/api/weighings", json={
        "animal_id": alert_animal["id"], "weight_kg": 500.0, "confidence": 0.95
    })
    requests.post(f"{api_url}/api/weighings", json={
        "animal_id": alert_animal["id"], "weight_kg": 440.0, "confidence": 0.93
    })
    print(f"\n  Alert animal: BOI_ALERT (500 -> 440 kg, -12%)")

    print(f"\nDone! {len(animals) + 1} animals, {len(animals) * 3 + 2} weighings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()
    seed(args.api_url)
