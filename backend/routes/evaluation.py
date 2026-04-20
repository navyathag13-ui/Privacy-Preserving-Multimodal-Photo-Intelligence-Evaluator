"""Evaluation trigger routes."""

from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.evaluation.runner import run_experiment
from backend.evaluation.search import ImageIndex
from backend.models.registry import available_models, get_model
from backend.storage import manager as store
from backend.storage.schema import EmbeddingRecord, ImageRecord

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

SUPPORTED_TASKS = {"captioning", "vqa", "clustering", "ranking", "search"}


class RunRequest(BaseModel):
    task_types: List[str]
    model_names: List[str]
    privacy_mode: bool = False
    image_ids: List[str] | None = None  # None = all images
    group_id: str | None = None
    vqa_questions: List[str] | None = None
    experiment_id: str | None = None


class SearchRequest(BaseModel):
    query: str
    model_name: str = "clip-vit-base-patch32"
    top_k: int = 5
    privacy_mode: bool = False


@router.get("/models")
async def list_models():
    """Return available model adapters and their capabilities."""
    return available_models()


@router.post("/run")
async def trigger_evaluation(
    req: RunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Trigger a benchmark evaluation run (async background task)."""
    invalid_tasks = set(req.task_types) - SUPPORTED_TASKS
    if invalid_tasks:
        raise HTTPException(400, detail=f"Unknown tasks: {invalid_tasks}")

    # Fetch image records
    stmt = select(ImageRecord)
    if req.image_ids:
        stmt = stmt.where(ImageRecord.id.in_(req.image_ids))
    elif req.group_id:
        stmt = stmt.where(ImageRecord.group_id == req.group_id)

    image_records = list((await db.execute(stmt)).scalars().all())
    if not image_records:
        raise HTTPException(400, detail="No matching images found")

    experiment_id = req.experiment_id or str(uuid.uuid4())[:8]

    # Run in background so request returns immediately
    background_tasks.add_task(
        run_experiment,
        db,
        image_records,
        req.task_types,
        req.model_names,
        privacy_mode=req.privacy_mode,
        experiment_id=experiment_id,
        vqa_questions=req.vqa_questions,
    )

    return {
        "status": "queued",
        "experiment_id": experiment_id,
        "image_count": len(image_records),
        "tasks": req.task_types,
        "models": req.model_names,
        "privacy_mode": req.privacy_mode,
    }


@router.post("/search")
async def semantic_search(req: SearchRequest, db: AsyncSession = Depends(get_db)):
    """Run a text-to-image semantic search against stored embeddings."""
    stmt = select(EmbeddingRecord).where(
        EmbeddingRecord.model_name == req.model_name,
        EmbeddingRecord.privacy_mode == req.privacy_mode,
    )
    emb_records = list((await db.execute(stmt)).scalars().all())
    if not emb_records:
        raise HTTPException(400, detail="No embeddings found. Run a 'search' evaluation first.")

    model = get_model(req.model_name)
    # Rebuild index from stored embeddings
    from PIL import Image as PILImage

    import numpy as np

    # We build index from stored embedding vectors directly
    ids = [r.image_id for r in emb_records]
    embs_list = [r.embedding for r in emb_records]

    class _FakeModel:
        name = req.model_name
        supports_embeddings = True

        def embed_text(self, text):
            return model.embed_text(text)

    index = ImageIndex(_FakeModel())
    index._image_ids = ids
    index._embeddings = np.array(embs_list, dtype="float32")
    try:
        import faiss

        dim = index._embeddings.shape[1]
        index._index = faiss.IndexFlatIP(dim)
        index._index.add(index._embeddings)
        index._use_faiss = True
    except ImportError:
        pass

    result = index.search(req.query, top_k=req.top_k)
    return {
        "query": result.query,
        "model": result.model_name,
        "results": [
            {"rank": r.rank, "image_id": r.image_id, "score": round(r.score, 4)}
            for r in result.results
        ],
    }
