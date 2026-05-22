import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.detect import BovineDetector, Detection


@pytest.fixture
def mock_yolo_result():
    result = MagicMock()
    mask_data = np.zeros((480, 640), dtype=np.uint8)
    mask_data[100:400, 150:500] = 1
    mask_tensor = MagicMock()
    mask_tensor.cpu.return_value.numpy.return_value = np.array([mask_data])
    result.masks = MagicMock()
    result.masks.data = mask_tensor
    box = MagicMock()
    box.xyxy = MagicMock()
    box.xyxy.cpu.return_value.numpy.return_value = np.array([[150, 100, 500, 400]])
    box.conf = MagicMock()
    box.conf.cpu.return_value.numpy.return_value = np.array([0.92])
    result.boxes = box
    return result


@pytest.fixture
def mock_yolo_no_detection():
    result = MagicMock()
    result.masks = None
    result.boxes = MagicMock()
    result.boxes.xyxy = MagicMock()
    result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([]).reshape(0, 4)
    result.boxes.conf = MagicMock()
    result.boxes.conf.cpu.return_value.numpy.return_value = np.array([])
    return result


@patch("src.detect.YOLO")
def test_detect_returns_detection(mock_yolo_cls, mock_yolo_result):
    mock_model = MagicMock()
    mock_model.return_value = [mock_yolo_result]
    mock_yolo_cls.return_value = mock_model
    detector = BovineDetector(model_path="fake.pt", confidence_threshold=0.5)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detection = detector.detect(frame)
    assert detection is not None
    assert isinstance(detection, Detection)
    assert detection.confidence == pytest.approx(0.92, abs=0.01)
    assert detection.mask.shape == (480, 640)
    assert detection.box == (150, 100, 500, 400)


@patch("src.detect.YOLO")
def test_detect_returns_none_below_threshold(mock_yolo_cls, mock_yolo_result):
    mock_yolo_result.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.3])
    mock_model = MagicMock()
    mock_model.return_value = [mock_yolo_result]
    mock_yolo_cls.return_value = mock_model
    detector = BovineDetector(model_path="fake.pt", confidence_threshold=0.5)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detection = detector.detect(frame)
    assert detection is None


@patch("src.detect.YOLO")
def test_detect_returns_none_when_no_detection(mock_yolo_cls, mock_yolo_no_detection):
    mock_model = MagicMock()
    mock_model.return_value = [mock_yolo_no_detection]
    mock_yolo_cls.return_value = mock_model
    detector = BovineDetector(model_path="fake.pt")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detection = detector.detect(frame)
    assert detection is None


def test_detection_dataclass():
    mask = np.zeros((100, 100), dtype=np.uint8)
    det = Detection(mask=mask, box=(10, 20, 90, 80), confidence=0.95)
    assert det.confidence == 0.95
    assert det.box == (10, 20, 90, 80)
