# Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    User / Recruiter                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │  browser / terminal
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Streamlit Dashboard  (dashboard/app.py)            │
│  • Dataset Browser  • Run Evaluation  • Results Dashboard       │
│  • Model Comparison • Privacy Comparison • Semantic Search      │
│  • Failure Gallery                                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTP / REST
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI Backend  (backend/main.py)            │
│                                                                  │
│  Routes                                                          │
│  ├── /images          – ingest, upload, list                    │
│  ├── /evaluation      – trigger runs, semantic search           │
│  └── /results         – query results, export CSV/Parquet       │
│                                                                  │
│  Core Modules                                                    │
│  ├── Preprocessing Pipeline                                      │
│  │   ├── Face Detection + Gaussian Blur (OpenCV DNN / Haar)     │
│  │   ├── Text Region Detection + Masking (EAST)                 │
│  │   └── EXIF Metadata Stripping                                │
│  │                                                               │
│  ├── Model Registry                                              │
│  │   ├── BLIPAdapter  (captioning + VQA)                        │
│  │   ├── CLIPAdapter  (image + text embeddings)                 │
│  │   └── [extensible: LLaVA, InstructBLIP, SigLIP …]           │
│  │                                                               │
│  ├── Evaluation Runner                                           │
│  │   ├── Captioning Task                                         │
│  │   ├── VQA Task                                               │
│  │   ├── Clustering Task  (cosine distance + greedy grouping)   │
│  │   ├── Best-Shot Ranking (sharpness/brightness/saturation)    │
│  │   └── Semantic Search  (FAISS / numpy cosine)                │
│  │                                                               │
│  └── Metrics Computation                                         │
│      ├── Per-model latency (avg, p95)                            │
│      ├── Hallucination rate                                      │
│      ├── Retrieval Precision@K                                   │
│      └── Privacy Degradation Delta                              │
│                                                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Layer                               │
│  ├── SQLite (SQLAlchemy async)                                   │
│  │   ├── images            – indexed image metadata             │
│  │   ├── evaluation_runs   – experiment metadata                │
│  │   ├── evaluation_results – per-image per-model outputs        │
│  │   └── embeddings        – stored vector embeddings           │
│  │                                                               │
│  └── Parquet / CSV export  (data/results/)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Local Folder / Upload
        │
        ▼
  Image Ingestion
  (PIL + metadata)
        │
        ▼
  Privacy Pipeline ──────────────────────┐
  (optional)                             │ privacy_mode=True
  └── strip EXIF                         │
  └── blur faces (OpenCV DNN)            │
  └── mask text regions (EAST)           │
        │                                │
        ▼                                ▼
  Model Adapters              Privacy-processed images
  ├── BLIP → captions, VQA answers
  └── CLIP → image embeddings
        │
        ▼
  Evaluation Tasks
  ├── captioning  → caption text + hallucination flag
  ├── vqa         → Q&A pairs
  ├── clustering  → cluster labels + duplicate pairs
  ├── ranking     → ordered best-shot list
  └── search      → FAISS / numpy ANN index
        │
        ▼
  Results Storage (SQLite)
  + Metrics Computation
        │
        ▼
  Dashboard / API
```

## Key Design Decisions

### 1. Adapter pattern for models
Each model is an isolated adapter class with a `load()` method.  The
registry holds singleton instances so weights are loaded once per
process.  Adding a new model requires only creating a new adapter
file and registering it.

### 2. Privacy as a first-class toggle
Every evaluation result records `privacy_mode: bool`.  The
`/results/privacy_delta` endpoint computes quality differences between
matched original and masked runs, surfacing the accuracy-privacy
tradeoff quantitatively.

### 3. Async-first backend
All database operations are async (SQLAlchemy + aiosqlite) to avoid
blocking the event loop during slow model inference.  Long-running
evaluations are dispatched as FastAPI `BackgroundTasks`.

### 4. No required GPU
CPU inference works out of the box for BLIP-base and CLIP-ViT-B/32.
The `DEVICE=cuda` env var enables GPU acceleration without code changes.

### 5. Extensibility hooks
- New tasks: add a module in `backend/evaluation/` and wire into `runner.py`
- New models: create `backend/models/<name>_adapter.py` and register
- New privacy ops: add to `backend/preprocessing/pipeline.py`
