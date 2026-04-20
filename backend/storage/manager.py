"""High-level storage operations used by evaluation components."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.storage.schema import (
    EmbeddingRecord,
    EvaluationResult,
    EvaluationRun,
    ImageRecord,
)

settings = get_settings()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

async def upsert_image(db: AsyncSession, image_data: dict[str, Any]) -> ImageRecord:
    """Insert or update an image record."""
    stmt = select(ImageRecord).where(ImageRecord.filepath == image_data["filepath"])
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        for k, v in image_data.items():
            setattr(existing, k, v)
        record = existing
    else:
        record = ImageRecord(**image_data)
        db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def list_images(db: AsyncSession) -> list[ImageRecord]:
    result = await db.execute(select(ImageRecord))
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Evaluation run helpers
# ---------------------------------------------------------------------------

async def create_run(db: AsyncSession, **kwargs) -> EvaluationRun:
    run = EvaluationRun(**kwargs)
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run


async def complete_run(db: AsyncSession, run_id: str) -> None:
    await db.execute(
        update(EvaluationRun)
        .where(EvaluationRun.id == run_id)
        .values(status="completed", completed_at=datetime.utcnow())
    )
    await db.commit()


async def get_run(db: AsyncSession, run_id: str) -> EvaluationRun | None:
    return (
        await db.execute(select(EvaluationRun).where(EvaluationRun.id == run_id))
    ).scalar_one_or_none()


async def list_runs(db: AsyncSession) -> list[EvaluationRun]:
    result = await db.execute(
        select(EvaluationRun).order_by(EvaluationRun.created_at.desc())
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Evaluation result helpers
# ---------------------------------------------------------------------------

async def save_result(db: AsyncSession, result_data: dict[str, Any]) -> EvaluationResult:
    result = EvaluationResult(**result_data)
    db.add(result)
    await db.commit()
    await db.refresh(result)
    return result


async def list_results(
    db: AsyncSession,
    run_id: str | None = None,
    task_type: str | None = None,
    model_name: str | None = None,
) -> list[EvaluationResult]:
    stmt = select(EvaluationResult)
    if run_id:
        stmt = stmt.where(EvaluationResult.run_id == run_id)
    if task_type:
        stmt = stmt.where(EvaluationResult.task_type == task_type)
    if model_name:
        stmt = stmt.where(EvaluationResult.model_name == model_name)
    return list((await db.execute(stmt)).scalars().all())


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

async def save_embedding(
    db: AsyncSession,
    image_id: str,
    model_name: str,
    embedding: list[float],
    privacy_mode: bool = False,
) -> EmbeddingRecord:
    record = EmbeddingRecord(
        image_id=image_id,
        model_name=model_name,
        embedding=embedding,
        privacy_mode=privacy_mode,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def get_embeddings(
    db: AsyncSession,
    model_name: str,
    privacy_mode: bool = False,
) -> list[EmbeddingRecord]:
    stmt = select(EmbeddingRecord).where(
        EmbeddingRecord.model_name == model_name,
        EmbeddingRecord.privacy_mode == privacy_mode,
    )
    return list((await db.execute(stmt)).scalars().all())


# ---------------------------------------------------------------------------
# Parquet export
# ---------------------------------------------------------------------------

def export_results_to_parquet(results: list[EvaluationResult], path: str | None = None) -> str:
    """Export results list to a Parquet file and return the file path."""
    out_dir = Path(settings.results_parquet_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = str(out_dir / f"results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet")

    rows = []
    for r in results:
        rows.append({
            "id": r.id,
            "run_id": r.run_id,
            "image_id": r.image_id,
            "task_type": r.task_type,
            "model_name": r.model_name,
            "prompt": r.prompt,
            "response": r.response,
            "score": r.score,
            "latency_ms": r.latency_ms,
            "privacy_mode": r.privacy_mode,
            "hallucination_flag": r.hallucination_flag,
            "error_tag": r.error_tag,
            "created_at": str(r.created_at),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


def export_results_to_csv(results: list[EvaluationResult], path: str | None = None) -> str:
    """Export results list to CSV and return the file path."""
    out_dir = Path(settings.results_parquet_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = str(out_dir / f"results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")

    rows = []
    for r in results:
        rows.append({
            "id": r.id,
            "run_id": r.run_id,
            "image_id": r.image_id,
            "task_type": r.task_type,
            "model_name": r.model_name,
            "response": r.response,
            "score": r.score,
            "latency_ms": r.latency_ms,
            "privacy_mode": r.privacy_mode,
            "hallucination_flag": r.hallucination_flag,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path
