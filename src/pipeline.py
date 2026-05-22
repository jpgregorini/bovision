"""
Pipeline orchestrator.
Runs the full weighing cycle: capture -> detect -> measure -> predict -> send.
"""
import os
import time
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import requests
from src.capture import FrameCapture
from src.detect import BovineDetector, Detection
from src.measure import MorphologyMeasurer, Measurements
from src.predict import WeightPredictor, WeightEstimate


@dataclass
class PipelineConfig:
    api_url: str = "http://localhost:8000"
    evidence_dir: str = "data/evidence"
    confidence_threshold: float = 0.5
    confirmation_frames: int = 3
    cooldown_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            api_url=os.getenv("API_URL", "http://localhost:8000"),
            evidence_dir=os.getenv("EVIDENCE_DIR", "data/evidence"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
            confirmation_frames=int(os.getenv("CONFIRMATION_FRAMES", "3")),
            cooldown_seconds=float(os.getenv("COOLDOWN_SECONDS", "30")),
        )


class Pipeline:
    def __init__(self, config=None, capture=None, detector=None, measurer=None, predictor=None):
        self.config = config or PipelineConfig.from_env()
        self.capture = capture or FrameCapture()
        self.detector = detector or BovineDetector()
        self.measurer = measurer or MorphologyMeasurer()
        self.predictor = predictor or WeightPredictor()
        Path(self.config.evidence_dir).mkdir(parents=True, exist_ok=True)

    def process_one(self) -> dict | None:
        confirmed_detections = []
        for _ in range(self.config.confirmation_frames):
            rgb, depth = self.capture.get_frame()
            detection = self.detector.detect(rgb)
            if detection is None:
                return None
            confirmed_detections.append((rgb, depth, detection))
        rgb, depth, detection = confirmed_detections[-1]
        measurements = self.measurer.measure(detection.mask, depth)
        estimate = self.predictor.predict(measurements)
        evidence_path = self._save_evidence(rgb, detection)
        result = {
            "weight_kg": estimate.weight_kg,
            "confidence": estimate.confidence,
            "comprimento_m": measurements.comprimento_m,
            "altura_m": measurements.altura_m,
            "largura_m": measurements.largura_m,
            "area_m2": measurements.area_m2,
            "perimetro_m": measurements.perimetro_m,
            "evidence_path": evidence_path,
        }
        self._send_to_api(result)
        return result

    def _save_evidence(self, rgb, detection):
        evidence = rgb.copy()
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 255, 0), 2)
        contours, _ = cv2.findContours(detection.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(evidence, contours, -1, (0, 255, 0), 2)
        timestamp = int(time.time())
        filename = f"{timestamp}.jpg"
        filepath = str(Path(self.config.evidence_dir) / filename)
        cv2.imwrite(filepath, evidence)
        return filepath

    def _send_to_api(self, data):
        url = f"{self.config.api_url}/api/weighings"
        try:
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 201:
                print(f"  Sent to API: {data['weight_kg']:.1f} kg")
            else:
                print(f"  API error: {response.status_code} {response.text}")
        except requests.RequestException as e:
            print(f"  API connection failed: {e}")

    def run(self):
        print("=== BoviSight Pipeline Running ===")
        print(f"  API: {self.config.api_url}")
        print(f"  Confirmation frames: {self.config.confirmation_frames}")
        print(f"  Cooldown: {self.config.cooldown_seconds}s")
        print("  Press Ctrl+C to stop.\n")
        while True:
            try:
                result = self.process_one()
                if result:
                    print(f"[WEIGHING] {result['weight_kg']:.1f} kg (confidence: {result['confidence']:.0%})")
                    time.sleep(self.config.cooldown_seconds)
                else:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nPipeline stopped.")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(1)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    Pipeline().run()
