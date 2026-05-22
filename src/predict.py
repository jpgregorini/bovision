"""
Weight prediction module.
Loads a trained XGBoost model and scaler, receives morphology measurements,
and estimates the cattle's weight in kilograms.
"""
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from src.measure import Measurements


@dataclass
class WeightEstimate:
    weight_kg: float
    confidence: float


class WeightPredictor:
    def __init__(self, model_path=None, scaler_path=None, metadata_path=None):
        if model_path is None:
            model_path = os.getenv("MODEL_REGRESSION_PATH", "models/regression/weight_model.pkl")
        if scaler_path is None:
            scaler_path = os.getenv("SCALER_PATH", "models/regression/scaler.pkl")
        if metadata_path is None:
            metadata_path = str(Path(model_path).parent / "metadata.json")
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)
        with open(metadata_path) as f:
            self._metadata = json.load(f)
        self._feature_names = self._metadata["feature_names"]

    def predict(self, measurements: Measurements) -> WeightEstimate:
        m_dict = {
            "comprimento_m": measurements.comprimento_m,
            "altura_m": measurements.altura_m,
            "largura_m": measurements.largura_m,
            "area_m2": measurements.area_m2,
            "perimetro_m": measurements.perimetro_m,
        }
        features = np.array([[m_dict[f] for f in self._feature_names]])
        features_scaled = self._scaler.transform(features)
        weight = float(self._model.predict(features_scaled)[0])
        confidence = self._estimate_confidence(features_scaled)
        return WeightEstimate(weight_kg=round(max(weight, 0), 1), confidence=round(confidence, 2))

    def _estimate_confidence(self, features_scaled: np.ndarray) -> float:
        pred = float(self._model.predict(features_scaled)[0])
        if 100 <= pred <= 800:
            base_confidence = 0.85
        elif 50 <= pred <= 1000:
            base_confidence = 0.65
        else:
            base_confidence = 0.40
        return min(base_confidence, 0.99)
