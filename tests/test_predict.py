import json
import pickle
import numpy as np
import pytest
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.predict import WeightPredictor, WeightEstimate
from src.measure import Measurements


@pytest.fixture
def trained_model(tmp_path):
    np.random.seed(42)
    X = np.random.rand(100, 3) * 2 + 0.5
    y = X[:, 0] * 200 + X[:, 1] * 150 + X[:, 2] * 50 + np.random.randn(100) * 5
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBRegressor(n_estimators=20, random_state=42)
    model.fit(X_scaled, y)
    model_path = tmp_path / "weight_model.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    metadata_path = tmp_path / "metadata.json"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(metadata_path, "w") as f:
        json.dump({"feature_names": ["comprimento_m", "altura_m", "perimetro_m"]}, f)
    return model_path, scaler_path, metadata_path


def test_predict_returns_weight_estimate(trained_model):
    model_path, scaler_path, metadata_path = trained_model
    predictor = WeightPredictor(
        model_path=str(model_path), scaler_path=str(scaler_path), metadata_path=str(metadata_path),
    )
    measurements = Measurements(comprimento_m=1.4, altura_m=1.3, largura_m=0.5, area_m2=1.9, perimetro_m=1.8)
    result = predictor.predict(measurements)
    assert isinstance(result, WeightEstimate)
    assert result.weight_kg > 0
    assert 0.0 <= result.confidence <= 1.0


def test_predict_higher_measurements_give_higher_weight(trained_model):
    model_path, scaler_path, metadata_path = trained_model
    predictor = WeightPredictor(
        model_path=str(model_path), scaler_path=str(scaler_path), metadata_path=str(metadata_path),
    )
    small = Measurements(comprimento_m=1.0, altura_m=0.9, largura_m=0.3, area_m2=1.0, perimetro_m=1.2)
    large = Measurements(comprimento_m=1.8, altura_m=1.5, largura_m=0.6, area_m2=2.5, perimetro_m=2.2)
    small_result = predictor.predict(small)
    large_result = predictor.predict(large)
    assert large_result.weight_kg > small_result.weight_kg


def test_weight_estimate_dataclass():
    est = WeightEstimate(weight_kg=487.4, confidence=0.91)
    assert est.weight_kg == 487.4
    assert est.confidence == 0.91
