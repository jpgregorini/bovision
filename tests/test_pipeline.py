import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from src.pipeline import Pipeline, PipelineConfig
from src.detect import Detection
from src.measure import Measurements
from src.predict import WeightEstimate


@pytest.fixture
def config(tmp_path):
    return PipelineConfig(
        api_url="http://localhost:8000",
        evidence_dir=str(tmp_path / "evidence"),
        confidence_threshold=0.5,
        confirmation_frames=2,
        cooldown_seconds=0,
    )


@pytest.fixture
def mock_components():
    capture = MagicMock()
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 3.0, dtype=np.float32)
    capture.get_frame.return_value = (rgb, depth)
    type(capture).has_frames = PropertyMock(return_value=True)

    detector = MagicMock()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:400, 150:500] = 1
    detector.detect.return_value = Detection(mask=mask, box=(150, 100, 500, 400), confidence=0.92)

    measurer = MagicMock()
    measurer.measure.return_value = Measurements(comprimento_m=1.43, altura_m=1.31, largura_m=0.49, area_m2=1.92, perimetro_m=1.84)

    predictor = MagicMock()
    predictor.predict.return_value = WeightEstimate(weight_kg=487.4, confidence=0.91)

    return capture, detector, measurer, predictor


def test_pipeline_single_weighing(config, mock_components):
    capture, detector, measurer, predictor = mock_components
    pipeline = Pipeline(config=config, capture=capture, detector=detector, measurer=measurer, predictor=predictor)
    with patch("src.pipeline.requests") as mock_requests:
        mock_requests.post.return_value = MagicMock(status_code=201)
        result = pipeline.process_one()
    assert result is not None
    assert result["weight_kg"] == 487.4
    assert result["confidence"] == 0.91
    mock_requests.post.assert_called_once()


def test_pipeline_no_detection(config, mock_components):
    capture, detector, measurer, predictor = mock_components
    detector.detect.return_value = None
    pipeline = Pipeline(config=config, capture=capture, detector=detector, measurer=measurer, predictor=predictor)
    result = pipeline.process_one()
    assert result is None


def test_pipeline_saves_evidence(config, mock_components, tmp_path):
    capture, detector, measurer, predictor = mock_components
    config.evidence_dir = str(tmp_path / "evidence")
    pipeline = Pipeline(config=config, capture=capture, detector=detector, measurer=measurer, predictor=predictor)
    with patch("src.pipeline.requests") as mock_requests:
        mock_requests.post.return_value = MagicMock(status_code=201)
        pipeline.process_one()
    evidence_files = list(Path(config.evidence_dir).glob("*.jpg"))
    assert len(evidence_files) == 1


def test_pipeline_confirmation_frames(config, mock_components):
    capture, detector, measurer, predictor = mock_components
    config.confirmation_frames = 3
    detector.detect.side_effect = [
        Detection(mask=np.ones((480, 640), dtype=np.uint8), box=(0, 0, 640, 480), confidence=0.9),
        Detection(mask=np.ones((480, 640), dtype=np.uint8), box=(0, 0, 640, 480), confidence=0.85),
        None,
    ]
    pipeline = Pipeline(config=config, capture=capture, detector=detector, measurer=measurer, predictor=predictor)
    result = pipeline.process_one()
    assert result is None
