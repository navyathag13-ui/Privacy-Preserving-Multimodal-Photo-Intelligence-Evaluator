"""Aggregate metrics computed from a list of EvaluationResult ORM records."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from backend.storage.schema import EvaluationResult


def compute_summary(results: list[EvaluationResult]) -> dict[str, Any]:
    """Return per-model summary statistics from a list of result records."""
    by_model: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in results:
        by_model[r.model_name].append(r)

    summary = {}
    for model_name, model_results in by_model.items():
        latencies = [r.latency_ms for r in model_results if r.latency_ms is not None]
        scores = [r.score for r in model_results if r.score is not None]
        hall_count = sum(1 for r in model_results if r.hallucination_flag)
        total = len(model_results)
        errors = sum(1 for r in model_results if r.error_tag)

        summary[model_name] = {
            "total_evaluations": total,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) > 1 else None,
            "avg_score": round(sum(scores) / len(scores), 4) if scores else None,
            "hallucination_count": hall_count,
            "hallucination_rate": round(hall_count / total, 4) if total else 0,
            "error_count": errors,
            "success_rate": round((total - errors) / total, 4) if total else 0,
        }
    return summary


def compute_privacy_delta(
    results_original: list[EvaluationResult],
    results_masked: list[EvaluationResult],
) -> dict[str, Any]:
    """Compare metrics between original and privacy-masked evaluation runs."""

    def avg_score(rs: list[EvaluationResult]) -> float | None:
        scores = [r.score for r in rs if r.score is not None]
        return round(sum(scores) / len(scores), 4) if scores else None

    orig_score = avg_score(results_original)
    mask_score = avg_score(results_masked)
    delta = None
    if orig_score is not None and mask_score is not None:
        delta = round(mask_score - orig_score, 4)

    return {
        "original_avg_score": orig_score,
        "masked_avg_score": mask_score,
        "privacy_degradation_delta": delta,
        "original_hallucination_rate": (
            round(sum(r.hallucination_flag for r in results_original) / len(results_original), 4)
            if results_original
            else None
        ),
        "masked_hallucination_rate": (
            round(sum(r.hallucination_flag for r in results_masked) / len(results_masked), 4)
            if results_masked
            else None
        ),
    }
