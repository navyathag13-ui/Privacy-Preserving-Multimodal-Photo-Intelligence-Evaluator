"""Image captioning evaluation task."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from PIL import Image

from backend.models.base import BaseModelAdapter

logger = logging.getLogger(__name__)

# Simple heuristic hallucination indicators — real eval would use a reference VLM judge
_HALLUCINATION_PATTERNS = [
    r"\bno \w+ (?:visible|present|shown)\b",
    r"\bI cannot\b",
    r"\bI'm not sure\b",
    r"\bunknown\b",
    r"\bunable to (?:see|determine)\b",
]
_HALL_RE = [re.compile(p, re.I) for p in _HALLUCINATION_PATTERNS]


@dataclass
class CaptionResult:
    model_name: str
    caption: str
    latency_ms: float
    hallucination_flag: bool
    word_count: int


def evaluate_captioning(
    model: BaseModelAdapter,
    image: Image.Image,
    image_id: str,
) -> CaptionResult:
    """Run captioning for one image and return structured metrics."""
    if not model.supports_captioning:
        raise ValueError(f"Model {model.name} does not support captioning")

    start = time.perf_counter()
    caption = model.caption(image)
    latency_ms = (time.perf_counter() - start) * 1000

    hallucination_flag = any(pattern.search(caption) for pattern in _HALL_RE)
    word_count = len(caption.split())

    logger.debug(
        "[captioning] %s | image=%s | %.1fms | words=%d",
        model.name,
        image_id,
        latency_ms,
        word_count,
    )

    return CaptionResult(
        model_name=model.name,
        caption=caption,
        latency_ms=latency_ms,
        hallucination_flag=hallucination_flag,
        word_count=word_count,
    )
