from __future__ import annotations

import logging
import sys

from .cameras.webcam import WebcamSource
from .config import load_config
from .detection.classifier import AnimalClassifier, Classifier
from .detection.yolo import YoloClassifier
from .notifications.desktop import DesktopNotifier
from .pipeline import DetectionPipeline


def _build_classifier(config) -> Classifier:
    backend = config.classification.backend.lower()
    if backend == "yolo":
        return YoloClassifier(model_path=config.classification.yolo_model)
    if backend == "claude":
        if not config.anthropic_api_key:
            logging.getLogger(__name__).error(
                "ANTHROPIC_API_KEY not set. Add it to .env or environment."
            )
            sys.exit(1)
        return AnimalClassifier(
            api_key=config.anthropic_api_key,
            model=config.classification.model,
        )
    logging.getLogger(__name__).error("Unknown classification backend: %s", backend)
    sys.exit(1)


def main() -> None:
    config = load_config()

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    camera = WebcamSource(device_index=config.camera.device_index)
    classifier = _build_classifier(config)
    notifier = DesktopNotifier(
        title_template=config.notification.title_template,
    )

    pipeline = DetectionPipeline(
        config=config,
        camera=camera,
        classifier=classifier,
        notifier=notifier,
    )

    try:
        log.info("Cat Alarm starting — press Ctrl+C to stop")
        pipeline.run()
    except KeyboardInterrupt:
        log.info("Shutting down.")


if __name__ == "__main__":
    main()
