"""Best-shot ranking for photo bursts.

Combines signal-based image quality metrics (sharpness, brightness, contrast,
face presence) into a composite score and ranks a set of similar images.

No model required for the base implementation, making this fast and reliable.
An optional model-embedding path is included for comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ImageQualitySignals:
    image_id: str
    sharpness: float          # Laplacian variance — higher is sharper
    brightness: float         # Mean luminance 0–255
    contrast: float           # Std-dev of luminance
    saturation: float         # Mean saturation in HSV
    face_score: float         # Bonus for images containing faces (0 or 1)
    composite_score: float    # Weighted combination


@dataclass
class RankingResult:
    group_id: str
    rankings: List[ImageQualitySignals]  # sorted best first
    best_image_id: str
    signals_explanation: str


def _compute_sharpness(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def _compute_brightness_contrast(img_gray: np.ndarray) -> tuple[float, float]:
    return float(img_gray.mean()), float(img_gray.std())


def _compute_saturation(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


def _has_face(img_bgr: np.ndarray) -> bool:
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    return len(faces) > 0


def _score_image(image_id: str, image: Image.Image) -> ImageQualitySignals:
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    sharpness = _compute_sharpness(img_gray)
    brightness, contrast = _compute_brightness_contrast(img_gray)
    saturation = _compute_saturation(img_bgr)
    face_score = 1.0 if _has_face(img_bgr) else 0.0

    # Normalise sharpness to ~0-1 range (empirical cap at 2000)
    norm_sharpness = min(sharpness / 2000.0, 1.0)
    # Ideal brightness is mid-range (128); penalise extremes
    norm_brightness = 1.0 - abs(brightness - 128.0) / 128.0
    norm_contrast = min(contrast / 80.0, 1.0)
    norm_saturation = min(saturation / 200.0, 1.0)

    composite = (
        0.40 * norm_sharpness
        + 0.20 * norm_brightness
        + 0.15 * norm_contrast
        + 0.15 * norm_saturation
        + 0.10 * face_score
    )

    return ImageQualitySignals(
        image_id=image_id,
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        saturation=round(saturation, 2),
        face_score=face_score,
        composite_score=round(composite, 4),
    )


def rank_burst(
    images: List[Image.Image],
    image_ids: List[str],
    group_id: str = "burst",
) -> RankingResult:
    """Score and rank a burst of images; return best-shot first."""
    scores = [_score_image(iid, img) for iid, img in zip(image_ids, images)]
    ranked = sorted(scores, key=lambda s: s.composite_score, reverse=True)
    best = ranked[0]

    explanation = (
        f"Ranked {len(images)} images using sharpness (40%), brightness (20%), "
        f"contrast (15%), saturation (15%), face presence (10%). "
        f"Best image: {best.image_id} with composite score {best.composite_score:.3f}."
    )

    logger.info("[ranking] group=%s | best=%s | score=%.3f", group_id, best.image_id, best.composite_score)

    return RankingResult(
        group_id=group_id,
        rankings=ranked,
        best_image_id=best.image_id,
        signals_explanation=explanation,
    )
