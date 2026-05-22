"""
Morphology measurement module.
Extracts physical measurements (in meters) from a binary mask and depth map.
"""
from dataclasses import dataclass
import cv2
import numpy as np

DEFAULT_FOCAL_LENGTH_PX = 380.0


@dataclass
class Measurements:
    comprimento_m: float
    altura_m: float
    largura_m: float
    area_m2: float
    perimetro_m: float


class MorphologyMeasurer:
    def __init__(self, focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX):
        self._focal_px = focal_length_px

    def measure(self, mask: np.ndarray, depth: np.ndarray) -> Measurements:
        if mask.sum() == 0:
            raise ValueError("Cannot measure from an empty mask.")
        animal_depths = depth[mask > 0]
        mean_depth = float(np.median(animal_depths))
        px_to_m = mean_depth / self._focal_px
        coords = np.where(mask > 0)
        y_min, y_max = int(coords[0].min()), int(coords[0].max())
        x_min, x_max = int(coords[1].min()), int(coords[1].max())
        width_px = x_max - x_min
        height_px = y_max - y_min
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter_px = sum(cv2.arcLength(c, closed=True) for c in contours)
        area_px = float(mask.sum())
        comprimento_m = float(width_px * px_to_m)
        altura_m = float(height_px * px_to_m)
        largura_m = float(comprimento_m * 0.35)
        area_m2 = float(area_px * (px_to_m ** 2))
        perimetro_m = float(perimeter_px * px_to_m)
        return Measurements(
            comprimento_m=round(comprimento_m, 3),
            altura_m=round(altura_m, 3),
            largura_m=round(largura_m, 3),
            area_m2=round(area_m2, 3),
            perimetro_m=round(perimetro_m, 3),
        )
