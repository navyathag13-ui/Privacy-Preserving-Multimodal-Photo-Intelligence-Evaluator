#!/usr/bin/env python3
"""CLI script to run a full benchmark experiment.

Usage:
    python scripts/run_benchmark.py \
        --tasks captioning vqa clustering \
        --models blip-base clip-vit-base-patch32 \
        --privacy
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import get_settings
from backend.database import AsyncSessionLocal, init_db
from backend.evaluation.runner import run_experiment
from backend.storage import manager as store

settings = get_settings()


async def run(
    tasks: list[str],
    models: list[str],
    privacy: bool,
    group_id: str | None,
    experiment_id: str | None,
) -> None:
    await init_db()

    async with AsyncSessionLocal() as db:
        all_images = await store.list_images(db)
        if group_id:
            all_images = [img for img in all_images if img.group_id == group_id]

        if not all_images:
            print("No images in database. Run ingest_images.py first.")
            return

        print(f"Running benchmark on {len(all_images)} images")
        print(f"  Tasks  : {tasks}")
        print(f"  Models : {models}")
        print(f"  Privacy: {privacy}")
        print()

        summary = await run_experiment(
            db,
            all_images,
            tasks,
            models,
            privacy_mode=privacy,
            experiment_id=experiment_id,
        )

        print("\n=== Experiment Summary ===")
        print(json.dumps(summary, indent=2))
        print("\nDone. View results in the dashboard: streamlit run dashboard/app.py")


def main():
    parser = argparse.ArgumentParser(description="Run a benchmark experiment")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["captioning"],
        choices=["captioning", "vqa", "clustering", "ranking", "search"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["blip-base"],
    )
    parser.add_argument("--privacy", action="store_true", help="Enable privacy preprocessing")
    parser.add_argument("--group", default=None, help="Filter images by group ID")
    parser.add_argument("--experiment-id", default=None, help="Custom experiment ID")

    args = parser.parse_args()
    asyncio.run(run(args.tasks, args.models, args.privacy, args.group, args.experiment_id))


if __name__ == "__main__":
    main()
