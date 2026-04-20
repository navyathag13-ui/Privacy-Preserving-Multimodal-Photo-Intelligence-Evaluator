# Privacy-Preserving Multimodal Photo Intelligence Evaluator

A **local-first multimodal ML evaluation platform** for benchmarking open-source vision-language models on real-world photo understanding tasks, with built-in privacy-preserving preprocessing.

## What it does

- Benchmarks multiple VLMs (BLIP, CLIP) on **captioning**, **VQA**, **semantic search**, **near-duplicate clustering**, and **best-shot ranking**
- Optionally applies **face blurring**, **text masking**, and **EXIF stripping** before evaluation
- Quantifies the **accuracy-privacy tradeoff** across models and tasks
- Stores all evaluation results in SQLite with Parquet/CSV export
- Surfaces results in an interactive Streamlit dashboard with model leaderboards, failure galleries, and privacy delta views

---

## Quick Start

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env to set DEVICE=cuda if you have a GPU
```

### 3. Generate sample images (or use your own)

```bash
python scripts/generate_sample_images.py
```

### 4. Ingest images

```bash
# Ingest the sample images
python scripts/ingest_images.py --folder ./data/images --group sample

# Or via the API after starting the backend
```

### 5. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

API docs available at [http://localhost:8000/docs](http://localhost:8000/docs)

### 6. Start the dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard available at [http://localhost:8501](http://localhost:8501)

### 7. Run a benchmark

```bash
# CLI
python scripts/run_benchmark.py \
    --tasks captioning vqa clustering search \
    --models blip-base clip-vit-base-patch32 \
    --privacy

# Or use the dashboard's "Run Evaluation" page
```

---

## Project Structure

```
.
├── backend/                    # FastAPI backend
│   ├── main.py                 # App entry point + lifespan
│   ├── config.py               # Settings (pydantic-settings)
│   ├── database.py             # SQLAlchemy async engine
│   ├── models/                 # Model adapters
│   │   ├── base.py             # Abstract adapter interface
│   │   ├── blip_adapter.py     # BLIP captioning + VQA
│   │   ├── clip_adapter.py     # CLIP embeddings
│   │   └── registry.py        # Singleton model registry
│   ├── evaluation/             # Benchmark tasks
│   │   ├── captioning.py       # Image captioning + hallucination flag
│   │   ├── vqa.py              # Visual question answering
│   │   ├── clustering.py       # Near-duplicate clustering
│   │   ├── ranking.py          # Best-shot signal-based ranking
│   │   ├── search.py           # Semantic search (FAISS/numpy)
│   │   ├── metrics.py          # Aggregate metrics + privacy delta
│   │   └── runner.py           # Experiment orchestration
│   ├── preprocessing/          # Privacy pipeline
│   │   ├── face_masking.py     # Face detection + Gaussian blur
│   │   ├── text_masking.py     # EAST text detection + masking
│   │   └── pipeline.py        # Unified PrivacyPipeline class
│   ├── routes/                 # API route modules
│   │   ├── images.py           # Upload / ingest / list
│   │   ├── evaluation.py       # Run evaluation, semantic search
│   │   └── results.py         # Query results, export
│   └── storage/                # ORM + manager helpers
│       ├── schema.py           # SQLAlchemy ORM models
│       └── manager.py         # CRUD helpers + Parquet export
├── dashboard/
│   └── app.py                  # Streamlit multi-page dashboard
├── data/
│   ├── images/                 # Image files (gitignored)
│   └── metadata/               # Ground-truth labels
│       ├── images.csv
│       ├── questions.json
│       ├── retrieval_labels.json
│       └── best_shot_labels.json
├── scripts/
│   ├── ingest_images.py        # CLI image ingestion
│   ├── run_benchmark.py        # CLI benchmark runner
│   └── generate_sample_images.py  # Synthetic test images
├── tests/
│   ├── test_preprocessing.py   # Privacy pipeline unit tests
│   ├── test_evaluation.py      # Evaluation module unit tests
│   └── test_api.py             # FastAPI integration tests
├── docs/
│   └── architecture.md        # Architecture diagram + design decisions
├── requirements.txt
└── .env.example
```

---

## Models Supported

| Model | Task | HuggingFace ID |
|-------|------|---------------|
| BLIP-base | Captioning, VQA | `Salesforce/blip-image-captioning-base` + `Salesforce/blip-vqa-base` |
| CLIP ViT-B/32 | Embeddings, Search, Clustering | `openai/clip-vit-base-patch32` |

### Adding a new model

1. Create `backend/models/your_model_adapter.py` extending `BaseModelAdapter`
2. Set `name`, `supports_captioning`, `supports_vqa`, `supports_embeddings`
3. Implement `load()` and the relevant capability methods
4. Import and register in `backend/models/registry.py`

---

## Evaluation Tasks

| Task | Description | Models Used |
|------|-------------|-------------|
| **Captioning** | Generate image captions; flag hallucinations | BLIP |
| **VQA** | Answer predefined questions per image | BLIP |
| **Clustering** | Group near-duplicate images by embedding similarity | CLIP |
| **Best-shot Ranking** | Score and rank burst images by sharpness, brightness, contrast | Signal-based (OpenCV) |
| **Semantic Search** | Text-to-image retrieval via cosine similarity | CLIP |

---

## Privacy Features

| Feature | Implementation | Default |
|---------|---------------|---------|
| Face detection + blur | OpenCV DNN (SSD) / Haar fallback | Enabled |
| Text region masking | EAST text detector | Enabled |
| EXIF metadata stripping | Pillow / piexif | Enabled |

The dashboard's **Privacy Comparison** page computes the quality delta between original and masked runs.

---

## API Reference

After starting the backend, visit [http://localhost:8000/docs](http://localhost:8000/docs) for the full OpenAPI spec.

Key endpoints:

```
POST /images/upload              – upload image files
POST /images/ingest_folder       – ingest from local folder path
GET  /images/                    – list all images

POST /evaluation/run             – trigger a benchmark experiment
POST /evaluation/search          – semantic text-to-image search
GET  /evaluation/models          – list available models

GET  /results/runs               – list all evaluation runs
GET  /results/run/{id}           – get results for a run
GET  /results/summary            – aggregate metrics per model
GET  /results/privacy_delta      – compare original vs masked quality
GET  /results/export/csv         – download results as CSV
GET  /results/export/parquet     – download results as Parquet
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Resume Bullets

> Built a multimodal evaluation platform for photo understanding tasks including captioning, visual question answering, semantic retrieval, near-duplicate clustering, and best-shot ranking using BLIP and CLIP.

> Benchmarked open-source vision-language models on task quality, latency (avg/p95), and hallucination risk using a structured async evaluation pipeline backed by SQLite with Parquet/CSV export.

> Added a privacy-preserving preprocessing layer with face detection and Gaussian blurring (OpenCV DNN), OCR text-region masking (EAST), and EXIF stripping; measured quality degradation between original and privacy-filtered evaluation runs.

> Surfaced results through an interactive Streamlit dashboard with model leaderboards, failure-case galleries, semantic search, and side-by-side privacy comparison tables.

> Designed a modular, adapter-based model registry (FastAPI + SQLAlchemy async + HuggingFace Transformers) enabling new VLMs to be plugged in by adding a single adapter file.

---

## Configuration

All settings are controlled via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `MODEL_CACHE_DIR` | `./model_cache` | Where HF weights are cached |
| `DATABASE_URL` | SQLite | Async SQLAlchemy URL |
| `ENABLE_FACE_MASKING` | `true` | Enable face blurring |
| `ENABLE_TEXT_MASKING` | `true` | Enable text masking |
| `FACE_BLUR_STRENGTH` | `25` | Gaussian blur kernel size |

---

## Hardware Requirements

| Setup | Expectation |
|-------|-------------|
| CPU only | Works; captioning ~2–5s/image, CLIP ~0.3s/image |
| GPU (CUDA) | ~10–20× faster; set `DEVICE=cuda` |
| RAM | 4 GB minimum; 8 GB recommended for both models loaded |

---

## License

MIT
