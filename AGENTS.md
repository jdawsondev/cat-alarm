# CLAUDE.md

## Project Overview

cat-alarm is a doorstep animal detection system. It captures webcam frames, detects motion, classifies animals, and sends desktop notifications.

## Architecture

Pipeline flow: Camera -> Motion Detection -> Classification -> Notification

- `src/cat_alarm/pipeline.py` — main detection loop, orchestrates all components
- `src/cat_alarm/detection/classifier.py` — `Classifier` ABC, `ClassificationResult` dataclass, and `AnimalClassifier` (Claude API backend)
- `src/cat_alarm/detection/yolo.py` — `YoloClassifier` (local YOLOv8 backend, uses COCO classes)
- `src/cat_alarm/detection/motion.py` — motion detection via frame differencing
- `src/cat_alarm/cameras/base.py` — `CameraSource` ABC
- `src/cat_alarm/cameras/webcam.py` — OpenCV webcam implementation
- `src/cat_alarm/notifications/base.py` — `Notifier` ABC
- `src/cat_alarm/notifications/desktop.py` — desktop notification via plyer
- `src/cat_alarm/config.py` — dataclass-based config, loaded from `config.yaml` and `.env`
- `src/cat_alarm/storage.py` — saves detection images to disk

## Configuration

All config is in `config.yaml`. API key goes in `.env` as `ANTHROPIC_API_KEY`.

Key settings:
- `classification.backend` — `"yolo"` (local, free) or `"claude"` (API, costs money)
- `classification.yolo_model` — YOLO weights file (default `yolov8n.pt`, auto-downloads)
- `log_level` — top-level setting, e.g. `DEBUG` or `INFO`

## Development

- Python src layout: code lives under `src/cat_alarm/`
- Virtual environment: `.venv/`
- Install: `pip install -e .`
- Run: `cat-alarm` (entry point defined in pyproject.toml)
- No test suite yet

## YOLO vs Claude backend

- YOLO: free, local, fast. Detects COCO animals (cat, dog, horse, bird, etc.). Cannot detect raccoons.
- Claude: uses Claude Vision API. Can detect any animal including raccoons. Requires API key and costs money.

## Conventions

- New classifier backends should extend `Classifier` ABC from `detection/classifier.py` and return `ClassificationResult`
- New camera sources extend `CameraSource` ABC from `cameras/base.py`
- New notifiers extend `Notifier` ABC from `notifications/base.py`
