"""Visual Question Answering evaluation task."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List

from PIL import Image

from backend.models.base import BaseModelAdapter

logger = logging.getLogger(__name__)

# Default question bank used when no custom questions are provided
DEFAULT_QUESTIONS = [
    "What objects are visible in this image?",
    "Is this scene indoors or outdoors?",
    "What is the main subject of this image?",
    "What text is visible in the image?",
    "Describe the lighting conditions.",
]


@dataclass
class VQAAnswer:
    question: str
    answer: str
    latency_ms: float


@dataclass
class VQAResult:
    model_name: str
    image_id: str
    answers: List[VQAAnswer]
    mean_latency_ms: float


def evaluate_vqa(
    model: BaseModelAdapter,
    image: Image.Image,
    image_id: str,
    questions: List[str] | None = None,
) -> VQAResult:
    """Run VQA for one image across all questions."""
    if not model.supports_vqa:
        raise ValueError(f"Model {model.name} does not support VQA")

    qs = questions or DEFAULT_QUESTIONS
    answers = []

    for question in qs:
        start = time.perf_counter()
        answer = model.answer(image, question)
        latency_ms = (time.perf_counter() - start) * 1000
        answers.append(VQAAnswer(question=question, answer=answer, latency_ms=latency_ms))
        logger.debug("[vqa] %s | %s | %.1fms", model.name, question[:40], latency_ms)

    mean_latency = sum(a.latency_ms for a in answers) / len(answers) if answers else 0.0
    return VQAResult(
        model_name=model.name,
        image_id=image_id,
        answers=answers,
        mean_latency_ms=mean_latency,
    )
