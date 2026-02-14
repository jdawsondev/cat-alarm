from __future__ import annotations

import logging
import time

from .cameras.base import CameraSource
from .config import AppConfig
from .detection.classifier import AnimalClassifier
from .detection.motion import MotionDetector
from .notifications.base import Notifier
from .storage import DetectionStorage

log = logging.getLogger(__name__)


class DetectionPipeline:
    def __init__(
        self,
        config: AppConfig,
        camera: CameraSource,
        classifier: AnimalClassifier,
        notifier: Notifier,
    ):
        self._config = config
        self._camera = camera
        self._classifier = classifier
        self._notifier = notifier
        self._motion = MotionDetector(
            blur_kernel_size=config.motion.blur_kernel_size,
            threshold=config.motion.threshold,
            min_contour_area=config.motion.min_contour_area,
            consecutive_frames=config.motion.consecutive_frames,
        )
        self._storage = DetectionStorage(config.storage.output_dir)
        self._target_animals = {
            a.lower() for a in config.classification.target_animals
        }
        self._cooldowns: dict[str, float] = {}
        self._cooldown_sec = config.notification.cooldown_seconds
        self._classify_cooldown_sec = config.classification.cooldown_seconds
        self._last_classification: float = 0
        self._frame_interval = 1.0 / config.camera.fps

    def run(self) -> None:
        log.info("Starting detection pipeline (targets: %s)", self._target_animals)
        with self._camera:
            while True:
                start = time.monotonic()
                frame = self._camera.read_frame()
                if frame is None:
                    log.warning("No frame received, retrying...")
                    time.sleep(0.5)
                    continue

                if self._motion.detect(frame):
                    now = time.monotonic()
                    if (now - self._last_classification) < self._classify_cooldown_sec:
                        log.debug("Motion detected but classification on cooldown, skipping")
                        elapsed = time.monotonic() - start
                        sleep_time = self._frame_interval - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        continue

                    log.debug("Motion detected, classifying frame...")
                    self._last_classification = now
                    result = self._classifier.classify(frame)

                    if result and result.animal_detected:
                        animal = result.animal_type.lower()
                        log.info(
                            "Animal classified: %s (confidence: %s) — %s",
                            animal,
                            result.confidence,
                            result.description,
                        )

                        image_path = self._storage.save(frame, animal)

                        if animal in self._target_animals and self._cooldown_ok(animal):
                            self._notifier.notify(
                                animal, image_path, result.description
                            )
                            self._cooldowns[animal] = time.monotonic()
                        elif animal not in self._target_animals:
                            log.info("Ignoring non-target animal: %s", animal)
                        else:
                            log.info("Cooldown active for %s, skipping notification", animal)

                elapsed = time.monotonic() - start
                sleep_time = self._frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _cooldown_ok(self, animal_type: str) -> bool:
        last = self._cooldowns.get(animal_type)
        if last is None:
            return True
        return (time.monotonic() - last) >= self._cooldown_sec
