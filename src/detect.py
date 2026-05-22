"""
Bovine detection and segmentation using YOLOv8.
Loads a trained YOLOv8-seg model and returns binary masks,
bounding boxes, and confidence scores for detected cattle.
"""
import os
from dataclasses import dataclass
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    mask: np.ndarray  # Binary mask, same size as input frame
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


class BovineDetector:
    def __init__(self, model_path: str | None = None, confidence_threshold: float | None = None):
        if model_path is None:
            model_path = os.getenv("MODEL_DETECTION_PATH", "models/detection/best.pt")
        if confidence_threshold is None:
            confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        self._model = YOLO(model_path)
        self._threshold = confidence_threshold

    def detect(self, rgb_frame: np.ndarray) -> Detection | None:
        results = self._model(rgb_frame, verbose=False)
        if not results:
            return None
        result = results[0]
        if result.masks is None:
            return None
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        if len(confs) == 0:
            return None
        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])
        if best_conf < self._threshold:
            return None
        masks = result.masks.data.cpu().numpy()
        mask = masks[best_idx]
        h, w = rgb_frame.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8)
        box = boxes[best_idx]
        box_tuple = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        return Detection(mask=mask, box=box_tuple, confidence=best_conf)
