from __future__ import annotations

import logging
from pathlib import Path

from plyer import notification

from .base import Notifier

log = logging.getLogger(__name__)


class DesktopNotifier(Notifier):
    def __init__(self, title_template: str = "Cat Alarm: {animal_type} detected!"):
        self._title_template = title_template

    def notify(self, animal_type: str, image_path: Path, description: str) -> None:
        title = self._title_template.format(animal_type=animal_type)
        message = f"{description}\nImage saved: {image_path}"

        try:
            notification.notify(
                title=title,
                message=message,
                app_name="Cat Alarm",
                timeout=10,
            )
            log.info("Desktop notification sent: %s", title)
        except Exception as e:
            log.error("Failed to send desktop notification: %s", e)
