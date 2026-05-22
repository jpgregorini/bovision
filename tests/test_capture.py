import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from src.capture import FrameCapture


@pytest.fixture
def sample_images_dir(tmp_path):
    """Create a temp directory with fake images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(3):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(img_dir / f"test_{i}.jpg"), img)
    return img_dir


def test_get_frame_returns_rgb_and_depth(sample_images_dir):
    capture = FrameCapture(images_dir=str(sample_images_dir))
    rgb, depth = capture.get_frame()
    assert isinstance(rgb, np.ndarray)
    assert isinstance(depth, np.ndarray)
    assert rgb.ndim == 3
    assert rgb.shape[2] == 3
    assert depth.ndim == 2
    assert rgb.shape[:2] == depth.shape


def test_get_frame_cycles_through_images(sample_images_dir):
    capture = FrameCapture(images_dir=str(sample_images_dir))
    frames = [capture.get_frame() for _ in range(5)]
    assert len(frames) == 5


def test_get_frame_depth_is_simulated(sample_images_dir):
    capture = FrameCapture(images_dir=str(sample_images_dir), mock_depth_m=3.0)
    _, depth = capture.get_frame()
    assert np.allclose(depth, 3.0, atol=0.1)


def test_empty_directory_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        FrameCapture(images_dir=str(empty_dir))


def test_has_frames_property(sample_images_dir):
    capture = FrameCapture(images_dir=str(sample_images_dir))
    assert capture.has_frames is True
