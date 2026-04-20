#!/usr/bin/env python3
"""CLI script to ingest images from a local folder into the evaluator database.

Usage:
    python scripts/ingest_images.py --folder ./data/images --group nature
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.database import AsyncSessionLocal, init_db
from backend.storage import manager as store

settings = get_settings()
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


async def ingest(folder: Path, group_id: str | None) -> None:
    await init_db()

    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
    if not paths:
        print(f"No supported images found in {folder}")
        return

    print(f"Found {len(paths)} images in {folder}")

    async with AsyncSessionLocal() as db:
        for p in paths:
            try:
                with Image.open(p) as img:
                    w, h = img.size
            except Exception as exc:
                print(f"  SKIP {p.name}: {exc}")
                continue

            rec = await store.upsert_image(db, {
                "filename": p.name,
                "filepath": str(p.resolve()),
                "width": w,
                "height": h,
                "file_size_bytes": p.stat().st_size,
                "group_id": group_id,
            })
            print(f"  ✓ {p.name} → {rec.id[:8]}…")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Ingest images into the evaluator")
    parser.add_argument("--folder", required=True, help="Path to folder containing images")
    parser.add_argument("--group", default=None, help="Optional group/burst ID")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    asyncio.run(ingest(folder, args.group))


if __name__ == "__main__":
    main()
