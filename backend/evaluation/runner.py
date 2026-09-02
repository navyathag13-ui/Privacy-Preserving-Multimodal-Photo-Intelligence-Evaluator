"""Orchestration layer: run a full evaluation experiment across models and tasks."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.evaluation.captioning import evaluate_captioning
from backend.evaluation.clustering import cluster_images
from backend.evaluation.ranking import rank_burst
from backend.evaluation.search import ImageIndex
from backend.evaluation.vqa import evaluate_vqa
from backend.models.registry import get_model
from backend.preprocessing.pipeline import PrivacyPipeline
from backend.storage import manager as store
from backend.storage.schema import ImageRecord

logger = logging.getLogger(__name__)
settings = get_settings()


async def run_experiment(
    db: AsyncSession,
    image_records: list[ImageRecord],
    task_types: list[str],
    model_names: list[str],
    privacy_mode: bool = False,
    experiment_id: str | None = None,
    vqa_questions: list[str] | None = None,
) -> dict[str, Any]:
    """Run a full benchmark experiment.

    Parameters
    ----------
    task_types:
        Subset of ["captioning", "vqa", "clustering", "ranking", "search"]
    model_names:
        List of registered model names to benchmark
    privacy_mode:
        Apply privacy pipeline before evaluation
    """
    experiment_id = experiment_id or str(uuid.uuid4())[:8]
    pipeline = PrivacyPipeline() if privacy_mode else None

    run_ids: dict[str, str] = {}
    results_summary: dict[str, Any] = {"experiment_id": experiment_id, "tasks": {}}

    # Load and optionally pre-process images
    images: list[Image.Image] = []
    image_ids: list[str] = []

    for rec in image_records:
        img = Image.open(rec.filepath).convert("RGB")
        if pipeline:
            img, _ = pipeline.process(img)
        images.append(img)
        image_ids.append(rec.id)

    # ---------------------------------------------------------------
    # Iterate over requested tasks and models
    # ---------------------------------------------------------------
    for task in task_types:
        results_summary["tasks"][task] = {}

        # Ranking is signal-based (no model needed) — run once outside model loop
        if task == "ranking":
            run = await store.create_run(
                db,
                experiment_id=experiment_id,
                task_type=task,
                model_name="signal-based",
                privacy_mode=privacy_mode,
            )
            run_ids[f"{task}_signal-based"] = run.id
            rr = rank_burst(images, image_ids)
            rd = await store.save_result(db, {
                "run_id": run.id,
                "image_id": rr.best_image_id,
                "task_type": "ranking",
                "model_name": "signal-based",
                "response": rr.signals_explanation,
                "privacy_mode": privacy_mode,
                "extra_metadata": {
                    "rankings": [
                        {
                            "image_id": s.image_id,
                            "composite_score": s.composite_score,
                            "sharpness": s.sharpness,
                            "brightness": s.brightness,
                        }
                        for s in rr.rankings
                    ]
                },
            })
            await store.complete_run(db, run.id)
            results_summary["tasks"][task]["signal-based"] = {
                "run_id": run.id,
                "result_count": 1,
            }
            continue

        for model_name in model_names:
            try:
                model = get_model(model_name)
            except ValueError as exc:
                logger.warning("Skipping unknown model %s: %s", model_name, exc)
                continue

            run = await store.create_run(
                db,
                experiment_id=experiment_id,
                task_type=task,
                model_name=model_name,
                privacy_mode=privacy_mode,
            )
            run_ids[f"{task}_{model_name}"] = run.id

            task_results = []

            if task == "captioning" and model.supports_captioning:
                for img, img_id in zip(images, image_ids):
                    cr = evaluate_captioning(model, img, img_id)
                    rd = await store.save_result(db, {
                        "run_id": run.id,
                        "image_id": img_id,
                        "task_type": "captioning",
                        "model_name": model_name,
                        "response": cr.caption,
                        "latency_ms": cr.latency_ms,
                        "privacy_mode": privacy_mode,
                        "hallucination_flag": cr.hallucination_flag,
                        "extra_metadata": {"word_count": cr.word_count},
                    })
                    task_results.append(rd)

            elif task == "vqa" and model.supports_vqa:
                for img, img_id in zip(images, image_ids):
                    vr = evaluate_vqa(model, img, img_id, questions=vqa_questions)
                    for ans in vr.answers:
                        rd = await store.save_result(db, {
                            "run_id": run.id,
                            "image_id": img_id,
                            "task_type": "vqa",
                            "model_name": model_name,
                            "prompt": ans.question,
                            "response": ans.answer,
                            "latency_ms": ans.latency_ms,
                            "privacy_mode": privacy_mode,
                        })
                        task_results.append(rd)

            elif task == "clustering" and model.supports_embeddings:
                cr = cluster_images(model, images, image_ids)
                rd = await store.save_result(db, {
                    "run_id": run.id,
                    "image_id": image_ids[0] if image_ids else "n/a",
                    "task_type": "clustering",
                    "model_name": model_name,
                    "response": f"{cr.num_clusters} clusters; {len(cr.duplicate_pairs)} dup pairs",
                    "privacy_mode": privacy_mode,
                    "extra_metadata": {
                        "num_clusters": cr.num_clusters,
                        "cluster_labels": cr.cluster_labels,
                        "duplicate_pairs": cr.duplicate_pairs,
                        "image_ids": cr.image_ids,
                    },
                })
                task_results.append(rd)

            elif task == "search" and model.supports_embeddings:
                index = ImageIndex(model)
                index.build(images, image_ids)
                await store.save_result(db, {
                    "run_id": run.id,
                    "image_id": image_ids[0] if image_ids else "n/a",
                    "task_type": "search",
                    "model_name": model_name,
                    "response": f"Index built with {len(images)} images",
                    "privacy_mode": privacy_mode,
                })
                # Store embeddings for later querying
                for img_id, img in zip(image_ids, images):
                    emb = model.embed_image(img)
                    await store.save_embedding(db, img_id, model_name, emb, privacy_mode)

            await store.complete_run(db, run.id)
            results_summary["tasks"][task][model_name] = {
                "run_id": run.id,
                "result_count": len(task_results),
            }

    return results_summary
