"""Unit tests for evaluation modules (no model loading required)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.evaluation.captioning import CaptionResult, evaluate_captioning
from backend.evaluation.clustering import cluster_images
from backend.evaluation.metrics import compute_privacy_delta, compute_summary
from backend.evaluation.ranking import rank_burst
from backend.evaluation.search import ImageIndex
from backend.evaluation.vqa import VQAResult, evaluate_vqa


def _dummy_image(w: int = 64, h: int = 64) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


# ────────────────────────────── Captioning ──────────────────────────────────

class TestCaptioning:
    def test_returns_caption_result(self):
        model = MagicMock()
        model.name = "mock"
        model.supports_captioning = True
        model.caption.return_value = "a dog in a field"
        img = _dummy_image()
        result = evaluate_captioning(model, img, "img001")
        assert isinstance(result, CaptionResult)
        assert result.caption == "a dog in a field"
        assert result.word_count == 5

    def test_hallucination_flag_triggered(self):
        model = MagicMock()
        model.name = "mock"
        model.supports_captioning = True
        model.caption.return_value = "I cannot determine what is in this image"
        result = evaluate_captioning(model, _dummy_image(), "img001")
        assert result.hallucination_flag is True

    def test_no_hallucination_for_normal_caption(self):
        model = MagicMock()
        model.name = "mock"
        model.supports_captioning = True
        model.caption.return_value = "A beautiful sunset over the ocean"
        result = evaluate_captioning(model, _dummy_image(), "img001")
        assert result.hallucination_flag is False

    def test_raises_for_unsupported_model(self):
        model = MagicMock()
        model.name = "clip"
        model.supports_captioning = False
        with pytest.raises(ValueError):
            evaluate_captioning(model, _dummy_image(), "img001")


# ──────────────────────────────── VQA ───────────────────────────────────────

class TestVQA:
    def test_returns_vqa_result(self):
        model = MagicMock()
        model.name = "mock"
        model.supports_vqa = True
        model.answer.return_value = "outdoors"
        result = evaluate_vqa(model, _dummy_image(), "img001", questions=["Is this indoors?"])
        assert isinstance(result, VQAResult)
        assert len(result.answers) == 1
        assert result.answers[0].answer == "outdoors"


# ─────────────────────────────── Ranking ────────────────────────────────────

class TestRanking:
    def test_rank_burst_returns_all_images(self):
        images = [_dummy_image() for _ in range(3)]
        ids = ["img001", "img002", "img003"]
        result = rank_burst(images, ids)
        assert len(result.rankings) == 3
        assert result.best_image_id in ids

    def test_rank_burst_single_image(self):
        result = rank_burst([_dummy_image()], ["img001"])
        assert result.best_image_id == "img001"


# ─────────────────────────────── Clustering ─────────────────────────────────

class TestClustering:
    def test_identical_images_cluster_together(self):
        model = MagicMock()
        model.name = "clip"
        model.supports_embeddings = True
        # Same embedding for first two, different for third
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [1.0, 0.0, 0.0]
        emb_c = [0.0, 1.0, 0.0]
        model.embed_image.side_effect = [emb_a, emb_b, emb_c]

        images = [_dummy_image() for _ in range(3)]
        ids = ["img001", "img002", "img003"]
        result = cluster_images(model, images, ids, threshold=0.05)
        # img001 and img002 should be in the same cluster
        assert result.cluster_labels[0] == result.cluster_labels[1]
        assert result.cluster_labels[0] != result.cluster_labels[2]


# ─────────────────────────────── Search ─────────────────────────────────────

class TestSearch:
    def _build_index(self):
        model = MagicMock()
        model.name = "clip"
        model.supports_embeddings = True
        embs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        model.embed_image.side_effect = embs
        model.embed_text.return_value = [1, 0, 0]  # matches first image

        images = [_dummy_image() for _ in range(3)]
        ids = ["img001", "img002", "img003"]
        index = ImageIndex(model)
        index.build(images, ids)
        return index, model

    def test_search_returns_results(self):
        index, _ = self._build_index()
        result = index.search("dog", top_k=2)
        assert len(result.results) == 2
        assert result.results[0].rank == 1

    def test_top_result_is_most_similar(self):
        index, _ = self._build_index()
        result = index.search("query", top_k=3)
        # img001's embedding [1,0,0] matches query [1,0,0] exactly
        assert result.results[0].image_id == "img001"


# ─────────────────────────────── Metrics ─────────────────────────────────────

class TestMetrics:
    def _make_result(self, model, score, latency, hallucination=False):
        r = MagicMock()
        r.model_name = model
        r.score = score
        r.latency_ms = latency
        r.hallucination_flag = hallucination
        r.error_tag = None
        return r

    def test_compute_summary_returns_per_model_stats(self):
        results = [
            self._make_result("blip", 0.8, 200),
            self._make_result("blip", 0.6, 180, hallucination=True),
            self._make_result("clip", 0.9, 50),
        ]
        summary = compute_summary(results)
        assert "blip" in summary
        assert "clip" in summary
        assert summary["blip"]["hallucination_count"] == 1

    def test_privacy_delta_computes_correctly(self):
        orig = [self._make_result("blip", 0.8, 200), self._make_result("blip", 0.9, 210)]
        masked = [self._make_result("blip", 0.7, 190), self._make_result("blip", 0.75, 200)]
        delta = compute_privacy_delta(orig, masked)
        assert delta["original_avg_score"] == pytest.approx(0.85, abs=0.01)
        assert delta["masked_avg_score"] == pytest.approx(0.725, abs=0.01)
        assert delta["privacy_degradation_delta"] < 0  # quality dropped
