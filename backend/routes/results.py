"""Results retrieval and export routes."""

from __future__ import annotations

import io
import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.evaluation.metrics import compute_privacy_delta, compute_summary
from backend.storage import manager as store
from backend.storage.schema import EvaluationResult

router = APIRouter(prefix="/results", tags=["results"])


def _result_to_dict(r: EvaluationResult) -> dict:
    return {
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
        "extra_metadata": r.extra_metadata,
        "created_at": str(r.created_at),
    }


@router.get("/runs")
async def list_runs(db: AsyncSession = Depends(get_db)):
    runs = await store.list_runs(db)
    return [
        {
            "id": r.id,
            "experiment_id": r.experiment_id,
            "task_type": r.task_type,
            "model_name": r.model_name,
            "privacy_mode": r.privacy_mode,
            "status": r.status,
            "created_at": str(r.created_at),
            "completed_at": str(r.completed_at) if r.completed_at else None,
        }
        for r in runs
    ]


@router.get("/run/{run_id}")
async def get_run_results(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    run = await store.get_run(db, run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    results = await store.list_results(db, run_id=run_id)
    return {
        "run_id": run_id,
        "task_type": run.task_type,
        "model_name": run.model_name,
        "privacy_mode": run.privacy_mode,
        "status": run.status,
        "results": [_result_to_dict(r) for r in results],
    }


@router.get("/summary")
async def results_summary(
    task_type: str | None = None,
    model_name: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    results = await store.list_results(db, task_type=task_type, model_name=model_name)
    return compute_summary(results)


@router.get("/privacy_delta")
async def privacy_delta(
    run_id_original: str,
    run_id_masked: str,
    db: AsyncSession = Depends(get_db),
):
    """Compare quality metrics between an original and a privacy-masked run."""
    orig = await store.list_results(db, run_id=run_id_original)
    masked = await store.list_results(db, run_id=run_id_masked)
    if not orig:
        raise HTTPException(404, "Original run results not found")
    if not masked:
        raise HTTPException(404, "Masked run results not found")
    return compute_privacy_delta(orig, masked)


@router.get("/export/csv")
async def export_csv(
    run_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Download results as CSV."""
    results = await store.list_results(db, run_id=run_id)
    if not results:
        raise HTTPException(404, "No results to export")
    path = store.export_results_to_csv(results)
    with open(path, "rb") as f:
        data = f.read()
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{os.path.basename(path)}"'},
    )


@router.get("/export/parquet")
async def export_parquet(
    run_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Download results as Parquet."""
    results = await store.list_results(db, run_id=run_id)
    if not results:
        raise HTTPException(404, "No results to export")
    path = store.export_results_to_parquet(results)
    with open(path, "rb") as f:
        data = f.read()
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{os.path.basename(path)}"'},
    )
