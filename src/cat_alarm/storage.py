from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class DetectionStorage:
    def __init__(self, output_dir: str = "./detections"):
        self._output_dir = Path(output_dir)

    def save(self, frame: np.ndarray, animal_type: str) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{animal_type}_{timestamp}.jpg"
        path = self._output_dir / filename
        cv2.imwrite(str(path), frame)
        log.info("Saved detection image: %s", path)
        return path
