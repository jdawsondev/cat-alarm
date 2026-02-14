from __future__ import annotations

import logging

import numpy as np
from ultralytics import YOLO

from .classifier import Classifier, ClassificationResult

log = logging.getLogger(__name__)

# COCO class IDs for animals
COCO_ANIMALS: dict[int, str] = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}


class YoloClassifier(Classifier):
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        self._model = YOLO(model_path)
        self._confidence = confidence

    def classify(self, frame: np.ndarray) -> ClassificationResult | None:
        try:
            results = self._model(frame, verbose=False)
        except Exception:
            log.exception("YOLO inference failed")
            return None

        best_animal: str | None = None
        best_conf: float = 0.0

        for result in results:
            num_boxes = len(result.boxes)
            log.debug("YOLO returned %d boxes", num_boxes)
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = result.names.get(cls_id, "unknown")
                log.debug(
                    "  box: class=%d (%s) conf=%.3f animal=%s",
                    cls_id, cls_name, conf, cls_id in COCO_ANIMALS,
                )
                if cls_id in COCO_ANIMALS and conf > best_conf:
                    best_conf = conf
                    best_animal = COCO_ANIMALS[cls_id]

        if best_animal is None:
            return ClassificationResult(
                animal_detected=False,
                animal_type="none",
                confidence="low",
                description="No animal detected by YOLO",
            )

        if best_conf >= 0.8:
            confidence = "high"
        elif best_conf >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        return ClassificationResult(
            animal_detected=True,
            animal_type=best_animal,
            confidence=confidence,
            description=f"YOLO detected {best_animal} ({best_conf:.0%} confidence)",
        )
