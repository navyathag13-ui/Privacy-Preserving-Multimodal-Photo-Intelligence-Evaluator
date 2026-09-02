"""Unified privacy-preserving preprocessing pipeline.

Usage
-----
    from backend.preprocessing.pipeline import PrivacyPipeline
    from PIL import Image

    pipeline = PrivacyPipeline(face_masking=True, text_masking=True, strip_metadata=True)
    processed, report = pipeline.process(Image.open("photo.jpg"))
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import piexif
from PIL import Image

from backend.config import get_settings
from backend.preprocessing.face_masking import blur_faces
from backend.preprocessing.text_masking import mask_text_regions

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class PrivacyReport:
    faces_detected: int = 0
    text_regions_detected: int = 0
    metadata_stripped: bool = False
    operations_applied: list[str] = field(default_factory=list)


class PrivacyPipeline:
    """Applies configurable privacy transformations to a PIL image."""

    def __init__(
        self,
        face_masking: bool | None = None,
        text_masking: bool | None = None,
        strip_metadata: bool = True,
        face_blur_strength: int | None = None,
    ):
        self.face_masking = (
            settings.enable_face_masking if face_masking is None else face_masking
        )
        self.text_masking = (
            settings.enable_text_masking if text_masking is None else text_masking
        )
        self.strip_metadata = strip_metadata
        self.face_blur_strength = face_blur_strength or settings.face_blur_strength

    def process(self, image: Image.Image) -> tuple[Image.Image, PrivacyReport]:
        """Apply privacy transforms and return (processed_image, report)."""
        report = PrivacyReport()
        img = image.copy()

        if self.strip_metadata:
            img = _strip_exif(img)
            report.metadata_stripped = True
            report.operations_applied.append("strip_exif")

        if self.face_masking:
            try:
                img, face_count = blur_faces(img, blur_strength=self.face_blur_strength)
                report.faces_detected = face_count
                if face_count > 0:
                    report.operations_applied.append(f"blur_faces({face_count})")
            except Exception as exc:
                logger.warning("Face masking failed: %s", exc)

        if self.text_masking:
            try:
                img, text_count = mask_text_regions(img)
                report.text_regions_detected = text_count
                if text_count > 0:
                    report.operations_applied.append(f"mask_text({text_count})")
            except Exception as exc:
                logger.warning("Text masking failed: %s", exc)

        return img, report

    def process_path(self, path: str | Path) -> tuple[Image.Image, PrivacyReport]:
        image = Image.open(path).convert("RGB")
        return self.process(image)


def _strip_exif(image: Image.Image) -> Image.Image:
    """Return a copy of the image with EXIF metadata removed."""
    try:
        buf = io.BytesIO()
        # Save without exif
        image.save(buf, format="JPEG", exif=b"")
        buf.seek(0)
        return Image.open(buf).copy()
    except Exception:
        # If format doesn't support exif, just return a copy
        return image.copy()
