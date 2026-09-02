"""Image ingestion and management routes."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Annotated, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.database import get_db
from backend.storage import manager as store
from backend.storage.schema import ImageRecord

router = APIRouter(prefix="/images", tags=["images"])
settings = get_settings()


def _image_to_dict(rec: ImageRecord) -> dict:
    return {
        "id": rec.id,
        "filename": rec.filename,
        "filepath": rec.filepath,
        "width": rec.width,
        "height": rec.height,
        "file_size_bytes": rec.file_size_bytes,
        "group_id": rec.group_id,
        "tags": rec.tags,
        "created_at": str(rec.created_at),
    }


@router.get("/")
async def list_images(db: AsyncSession = Depends(get_db)):
    """Return all ingested images."""
    images = await store.list_images(db)
    return [_image_to_dict(r) for r in images]


@router.post("/upload")
async def upload_images(
    files: List[UploadFile] = File(...),
    group_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Upload one or more image files."""
    upload_dir = Path(settings.images_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not an image")

        dest = upload_dir / f"{uuid.uuid4()}_{f.filename}"
        with dest.open("wb") as buf:
            shutil.copyfileobj(f.file, buf)

        try:
            with Image.open(dest) as img:
                w, h = img.size
        except Exception:
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Could not read {f.filename}")

        rec = await store.upsert_image(db, {
            "filename": f.filename,
            "filepath": str(dest),
            "width": w,
            "height": h,
            "file_size_bytes": dest.stat().st_size,
            "group_id": group_id,
        })
        records.append(_image_to_dict(rec))

    return {"uploaded": len(records), "images": records}


@router.post("/ingest_folder")
async def ingest_folder(
    folder_path: str,
    group_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Ingest all images from a local folder path."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail="Folder not found")

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    paths = [p for p in folder.iterdir() if p.suffix.lower() in extensions]

    if not paths:
        return {"ingested": 0, "images": []}

    records = []
    for p in paths:
        try:
            with Image.open(p) as img:
                w, h = img.size
        except Exception:
            continue

        rec = await store.upsert_image(db, {
            "filename": p.name,
            "filepath": str(p),
            "width": w,
            "height": h,
            "file_size_bytes": p.stat().st_size,
            "group_id": group_id,
        })
        records.append(_image_to_dict(rec))

    return {"ingested": len(records), "images": records}


@router.get("/{image_id}")
async def get_image(image_id: str, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select

    stmt = select(ImageRecord).where(ImageRecord.id == image_id)
    rec = (await db.execute(stmt)).scalar_one_or_none()
    if not rec:
        raise HTTPException(status_code=404, detail="Image not found")
    return _image_to_dict(rec)
