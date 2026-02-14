from __future__ import annotations

import logging
import sys

from .cameras.webcam import WebcamSource
from .config import load_config
from .detection.classifier import AnimalClassifier
from .notifications.desktop import DesktopNotifier
from .pipeline import DetectionPipeline


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    config = load_config()

    if not config.anthropic_api_key:
        log.error("ANTHROPIC_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    camera = WebcamSource(device_index=config.camera.device_index)
    classifier = AnimalClassifier(
        api_key=config.anthropic_api_key,
        model=config.classification.model,
    )
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
