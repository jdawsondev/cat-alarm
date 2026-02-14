from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    def __init__(
        self,
        blur_kernel_size: int = 21,
        threshold: int = 25,
        min_contour_area: int = 500,
    ):
        self._blur_k = blur_kernel_size
        self._threshold = threshold
        self._min_area = min_contour_area
        self._prev_gray: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self._blur_k, self._blur_k), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        delta = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        thresh = cv2.threshold(delta, self._threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return any(cv2.contourArea(c) >= self._min_area for c in contours)
