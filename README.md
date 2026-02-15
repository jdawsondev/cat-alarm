# Cat Alarm

Doorstep animal detection system that uses a webcam, motion sensing, and AI classification to detect animals and send desktop notifications.

## How It Works

1. Captures frames from a webcam
2. Detects motion via frame differencing
3. Classifies detected motion using either YOLOv8 (local) or Claude Vision (API)
4. Sends a desktop notification when a target animal is detected
5. Saves detection images to disk

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -e .
```

For the Claude backend, create a `.env` file:

```
ANTHROPIC_API_KEY=your-key-here
```

## Configuration

Edit `config.yaml` to configure the system. Key settings:

| Setting | Description |
|---|---|
| `classification.backend` | `"yolo"` (local, free) or `"claude"` (API) |
| `classification.target_animals` | List of animals to alert on (e.g. `cat`, `raccoon`) |
| `classification.cooldown_seconds` | Seconds between classifications |
| `notification.cooldown_seconds` | Seconds between notifications for the same animal |
| `log_level` | `DEBUG`, `INFO`, `WARNING`, etc. |

### Classification Backends

- **YOLOv8** (`backend: yolo`): Runs locally, no API key needed. Detects COCO dataset animals (cat, dog, bird, horse, etc.). Cannot detect raccoons. Model weights download automatically on first run.
- **Claude Vision** (`backend: claude`): Uses the Anthropic API. Can identify any animal including raccoons. Requires `ANTHROPIC_API_KEY`.

## Usage

```bash
cat-alarm
```

Press Ctrl+C to stop. Detection images are saved to `./detections/`.
