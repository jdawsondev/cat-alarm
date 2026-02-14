from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class CameraConfig:
    type: str = "webcam"
    device_index: int = 0
    fps: int = 2


@dataclass
class MotionConfig:
    blur_kernel_size: int = 21
    threshold: int = 25
    min_contour_area: int = 500
    consecutive_frames: int = 3


@dataclass
class ClassificationConfig:
    model: str = "claude-haiku-4-5-20251001"
    cooldown_seconds: int = 30
    target_animals: list[str] = field(default_factory=lambda: ["cat", "raccoon"])


@dataclass
class NotificationConfig:
    type: str = "desktop"
    cooldown_seconds: int = 300
    title_template: str = "Cat Alarm: {animal_type} detected!"


@dataclass
class StorageConfig:
    output_dir: str = "./detections"


@dataclass
class AppConfig:
    camera: CameraConfig
    motion: MotionConfig
    classification: ClassificationConfig
    notification: NotificationConfig
    storage: StorageConfig
    anthropic_api_key: str = ""


def load_config(config_path: str = "config.yaml", env_path: str = ".env") -> AppConfig:
    load_dotenv(env_path)

    yaml_data: dict = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            yaml_data = yaml.safe_load(f) or {}

    return AppConfig(
        camera=CameraConfig(**yaml_data.get("camera", {})),
        motion=MotionConfig(**yaml_data.get("motion", {})),
        classification=ClassificationConfig(**yaml_data.get("classification", {})),
        notification=NotificationConfig(**yaml_data.get("notification", {})),
        storage=StorageConfig(**yaml_data.get("storage", {})),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )
