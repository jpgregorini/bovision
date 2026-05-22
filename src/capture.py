"""
Frame capture module.
In dev mode: reads images from a directory and generates synthetic depth maps.
In prod mode (future): connects to Intel RealSense cameras.
"""
import os
from pathlib import Path
import cv2
import numpy as np


class FrameCapture:
    """Captures RGB + depth frames. Uses mock images in dev mode."""

    def __init__(self, images_dir: str | None = None, mock_depth_m: float = 3.0):
        env = os.getenv("BOVISION_ENV", "dev")
        if env == "dev":
            self._init_mock(images_dir, mock_depth_m)
        else:
            self._init_realsense()

    def _init_mock(self, images_dir: str | None, mock_depth_m: float):
        if images_dir is None:
            images_dir = os.getenv("SAMPLE_IMAGES_DIR", "data/sample_images")
        self._images_dir = Path(images_dir)
        self._mock_depth_m = mock_depth_m
        self._mode = "mock"
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self._image_paths = sorted([
            p for p in self._images_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ])
        if len(self._image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self._images_dir}. "
                "Add .jpg/.png images or set SAMPLE_IMAGES_DIR."
            )
        self._index = 0

    def _init_realsense(self):
        raise NotImplementedError(
            "RealSense capture requires BOVISION_ENV=prod and pyrealsense2. "
            "Set BOVISION_ENV=dev for mock mode."
        )

    @property
    def has_frames(self) -> bool:
        return len(self._image_paths) > 0

    def get_frame(self) -> tuple[np.ndarray, np.ndarray]:
        if self._mode == "mock":
            return self._get_mock_frame()
        raise NotImplementedError("RealSense capture not implemented in MVP")

    def _get_mock_frame(self) -> tuple[np.ndarray, np.ndarray]:
        img_path = self._image_paths[self._index % len(self._image_paths)]
        self._index += 1
        rgb = cv2.imread(str(img_path))
        if rgb is None:
            raise IOError(f"Failed to read image: {img_path}")
        h, w = rgb.shape[:2]
        depth = np.full((h, w), self._mock_depth_m, dtype=np.float32)
        return rgb, depth
