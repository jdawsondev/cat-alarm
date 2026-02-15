from __future__ import annotations

import logging

import cv2
import numpy as np

from .base import CameraSource

log = logging.getLogger(__name__)


class WebcamSource(CameraSource):
    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open webcam at index {self._device_index}"
            )
        log.info("Webcam opened (device %d)", self._device_index)

    def read_frame(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Webcam closed")
