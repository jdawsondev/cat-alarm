from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Notifier(ABC):
    @abstractmethod
    def notify(self, animal_type: str, image_path: Path, description: str) -> None: ...
