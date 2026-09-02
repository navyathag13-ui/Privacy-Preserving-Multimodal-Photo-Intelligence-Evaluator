"""Unit tests for the privacy preprocessing pipeline."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.preprocessing.pipeline import PrivacyPipeline, _strip_exif


def _make_rgb_image(w: int = 100, h: int = 100) -> Image.Image:
    """Create a simple RGB test image."""
    data = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(data, mode="RGB")


class TestStripExif:
    def test_strip_exif_returns_image(self):
        img = _make_rgb_image()
        result = _strip_exif(img)
        assert isinstance(result, Image.Image)

    def test_strip_exif_preserves_dimensions(self):
        img = _make_rgb_image(200, 150)
        result = _strip_exif(img)
        assert result.size == (200, 150)


class TestPrivacyPipeline:
    def test_pipeline_returns_image_and_report(self):
        pipeline = PrivacyPipeline(face_masking=False, text_masking=False)
        img = _make_rgb_image()
        processed, report = pipeline.process(img)
        assert isinstance(processed, Image.Image)
        assert report.metadata_stripped is True

    def test_pipeline_with_all_disabled(self):
        pipeline = PrivacyPipeline(face_masking=False, text_masking=False, strip_metadata=False)
        img = _make_rgb_image()
        processed, report = pipeline.process(img)
        assert report.metadata_stripped is False
        assert report.faces_detected == 0
        assert report.text_regions_detected == 0

    def test_pipeline_preserves_image_size(self):
        pipeline = PrivacyPipeline(face_masking=False, text_masking=False)
        img = _make_rgb_image(300, 200)
        processed, _ = pipeline.process(img)
        assert processed.size == (300, 200)

    def test_report_operations_list(self):
        pipeline = PrivacyPipeline(face_masking=False, text_masking=False, strip_metadata=True)
        img = _make_rgb_image()
        _, report = pipeline.process(img)
        assert "strip_exif" in report.operations_applied
