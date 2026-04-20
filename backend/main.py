"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.database import init_db
from backend.routes.evaluation import router as eval_router
from backend.routes.images import router as images_router
from backend.routes.results import router as results_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise database tables and data directories."""
    logger.info("Starting up …")
    Path(settings.images_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.metadata_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.results_parquet_dir).mkdir(parents=True, exist_ok=True)
    await init_db()
    logger.info("Database ready.")
    yield
    logger.info("Shutting down …")


app = FastAPI(
    title="Privacy-Preserving Multimodal Photo Intelligence Evaluator",
    description=(
        "Benchmark open-source vision-language models on captioning, VQA, "
        "clustering, ranking, and semantic search — with privacy-preserving "
        "face and text masking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(images_router)
app.include_router(eval_router)
app.include_router(results_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": app.version}


@app.get("/")
async def root():
    return {
        "message": "Photo Intelligence Evaluator API",
        "docs": "/docs",
        "health": "/health",
    }
