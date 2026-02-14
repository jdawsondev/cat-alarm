from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass

import anthropic
import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    animal_detected: bool
    animal_type: str
    confidence: str
    description: str


class AnimalClassifier:
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def classify(self, frame: np.ndarray) -> ClassificationResult | None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            log.error("Failed to encode frame as JPEG")
            return None

        image_b64 = base64.standard_b64encode(buf.tobytes()).decode()

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this doorstep camera image. Is there an animal visible? "
                                    "Respond with ONLY valid JSON (no markdown):\n"
                                    '{"animal_detected": true/false, "animal_type": "cat"/"raccoon"/"dog"/etc or "none", '
                                    '"confidence": "high"/"medium"/"low", '
                                    '"description": "brief description of what you see"}'
                                ),
                            },
                        ],
                    }
                ],
            )
        except anthropic.APIError as e:
            log.error("Claude API error: %s", e)
            return None

        text = response.content[0].text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.warning("Failed to parse classifier response: %s", text)
            return None

        return ClassificationResult(
            animal_detected=data.get("animal_detected", False),
            animal_type=data.get("animal_type", "unknown"),
            confidence=data.get("confidence", "low"),
            description=data.get("description", ""),
        )
