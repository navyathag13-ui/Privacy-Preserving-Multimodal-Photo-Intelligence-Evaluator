"""Streamlit evaluation dashboard.

Run:
    streamlit run dashboard/app.py

The dashboard talks to the FastAPI backend via HTTP.  Start the backend first:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Photo Intelligence Evaluator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────── helpers ──────

def api(method: str, path: str, **kwargs):
    """Thin wrapper around requests for the backend API."""
    url = f"{API_BASE}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Start it with: `uvicorn backend.main:app --reload`")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def safe_api(method: str, path: str, **kwargs):
    return api(method, path, **kwargs)


# ────────────────────────────────────────────────────────────── sidebar ──────

with st.sidebar:
    st.title("🔍 Photo Intelligence Evaluator")
    st.caption("Multimodal ML Benchmark Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📁 Dataset Browser",
            "🚀 Run Evaluation",
            "📊 Results Dashboard",
            "🏆 Model Comparison",
            "🔒 Privacy Comparison",
            "🔎 Semantic Search",
            "💥 Failure Gallery",
        ],
    )
    st.divider()
    health = safe_api("get", "/health")
    if health:
        st.success(f"Backend online  v{health.get('version', '?')}")
    else:
        st.error("Backend offline")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dataset Browser
# ══════════════════════════════════════════════════════════════════════════════

if page == "📁 Dataset Browser":
    st.header("Dataset Browser")

    col_upload, col_ingest = st.columns(2)

    with col_upload:
        st.subheader("Upload Images")
        group_id_up = st.text_input("Group ID (optional)", key="grp_up")
        uploaded = st.file_uploader(
            "Choose images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
        )
        if st.button("Upload", disabled=not uploaded):
            files = [("files", (f.name, f.read(), f.type)) for f in uploaded]
            params = {"group_id": group_id_up} if group_id_up else {}
            result = api("post", "/images/upload", files=files, params=params)
            if result:
                st.success(f"Uploaded {result['uploaded']} image(s)")

    with col_ingest:
        st.subheader("Ingest Local Folder")
        folder_path = st.text_input("Folder path", placeholder="/path/to/photos")
        group_id_ingest = st.text_input("Group ID (optional)", key="grp_ingest")
        if st.button("Ingest Folder", disabled=not folder_path):
            result = api(
                "post",
                "/images/ingest_folder",
                params={"folder_path": folder_path, "group_id": group_id_ingest or None},
            )
            if result:
                st.success(f"Ingested {result['ingested']} image(s)")

    st.divider()
    st.subheader("Image Library")
    images_data = safe_api("get", "/images/")
    if images_data:
        df = pd.DataFrame(images_data)
        if not df.empty:
            st.dataframe(
                df[["id", "filename", "width", "height", "file_size_bytes", "group_id", "created_at"]],
                use_container_width=True,
            )
            st.caption(f"{len(df)} images in library")

            # Preview grid
            st.subheader("Preview")
            cols = st.columns(4)
            for idx, row in df.iterrows():
                try:
                    img = Image.open(row["filepath"]).convert("RGB")
                    img.thumbnail((200, 200))
                    cols[idx % 4].image(img, caption=row["filename"][:20], use_column_width=True)
                except Exception:
                    pass
        else:
            st.info("No images ingested yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Run Evaluation
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🚀 Run Evaluation":
    st.header("Run Evaluation")

    models_data = safe_api("get", "/evaluation/models") or []
    model_names = [m["name"] for m in models_data]

    TASKS = ["captioning", "vqa", "clustering", "ranking", "search"]

    col1, col2 = st.columns(2)
    with col1:
        selected_tasks = st.multiselect("Tasks", TASKS, default=["captioning"])
        selected_models = st.multiselect("Models", model_names, default=model_names[:1] if model_names else [])
        privacy_mode = st.toggle("Privacy mode (mask faces + text)", value=False)
    with col2:
        experiment_id = st.text_input("Experiment ID (optional)")
        group_filter = st.text_input("Filter by Group ID (optional)")
        vqa_qs_text = st.text_area(
            "Custom VQA questions (one per line, optional)",
            placeholder="What objects are visible?\nIs this indoors or outdoors?",
        )

    vqa_questions = [q.strip() for q in vqa_qs_text.splitlines() if q.strip()] or None

    if st.button("▶ Run Evaluation", type="primary", disabled=not selected_tasks or not selected_models):
        payload = {
            "task_types": selected_tasks,
            "model_names": selected_models,
            "privacy_mode": privacy_mode,
            "group_id": group_filter or None,
            "vqa_questions": vqa_questions,
            "experiment_id": experiment_id or None,
        }
        result = api("post", "/evaluation/run", json=payload)
        if result:
            st.success(
                f"Evaluation queued — experiment `{result['experiment_id']}` "
                f"| {result['image_count']} images | tasks: {result['tasks']}"
            )
            st.json(result)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Results Dashboard
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Results Dashboard":
    st.header("Results Dashboard")

    runs_data = safe_api("get", "/results/runs") or []

    if not runs_data:
        st.info("No evaluation runs yet. Go to **Run Evaluation** to start.")
    else:
        runs_df = pd.DataFrame(runs_data)
        st.dataframe(runs_df, use_container_width=True)

        selected_run_id = st.selectbox(
            "Select a run to inspect",
            runs_df["id"].tolist(),
            format_func=lambda rid: next(
                (f"{r['task_type']} | {r['model_name']} | {r['created_at'][:19]}" for r in runs_data if r["id"] == rid),
                rid,
            ),
        )

        if selected_run_id:
            run_detail = safe_api("get", f"/results/run/{selected_run_id}")
            if run_detail and run_detail.get("results"):
                st.subheader(f"Results for run `{selected_run_id[:8]}…`")
                detail_df = pd.DataFrame(run_detail["results"])
                cols_show = [c for c in ["image_id", "prompt", "response", "score", "latency_ms", "hallucination_flag"] if c in detail_df.columns]
                st.dataframe(detail_df[cols_show], use_container_width=True)

                if "latency_ms" in detail_df.columns:
                    fig = px.histogram(
                        detail_df, x="latency_ms", nbins=20,
                        title="Latency Distribution (ms)",
                        color_discrete_sequence=["#636EFA"],
                    )
                    st.plotly_chart(fig, use_container_width=True)

            col_csv, col_parq = st.columns(2)
            with col_csv:
                if st.button("Export CSV"):
                    resp = requests.get(f"{API_BASE}/results/export/csv", params={"run_id": selected_run_id})
                    if resp.ok:
                        st.download_button("Download CSV", resp.content, file_name="results.csv", mime="text/csv")
            with col_parq:
                if st.button("Export Parquet"):
                    resp = requests.get(f"{API_BASE}/results/export/parquet", params={"run_id": selected_run_id})
                    if resp.ok:
                        st.download_button("Download Parquet", resp.content, file_name="results.parquet")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Comparison
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🏆 Model Comparison":
    st.header("Model Comparison Leaderboard")

    summary = safe_api("get", "/results/summary")
    if not summary:
        st.info("No results yet.")
    else:
        rows = []
        for model_name, stats in summary.items():
            rows.append({"model": model_name, **stats})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        if "avg_latency_ms" in df.columns:
            fig1 = px.bar(
                df.dropna(subset=["avg_latency_ms"]),
                x="model", y="avg_latency_ms",
                title="Average Latency (ms) per Model",
                color="model",
            )
            st.plotly_chart(fig1, use_container_width=True)

        if "hallucination_rate" in df.columns:
            fig2 = px.bar(
                df.dropna(subset=["hallucination_rate"]),
                x="model", y="hallucination_rate",
                title="Hallucination Rate per Model",
                color="model",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Privacy Comparison
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔒 Privacy Comparison":
    st.header("Privacy Comparison")
    st.caption("Compare model quality on original vs. privacy-masked images.")

    runs_data = safe_api("get", "/results/runs") or []
    run_opts = {
        f"{r['task_type']} | {r['model_name']} | privacy={r['privacy_mode']} | {r['created_at'][:19]}": r["id"]
        for r in runs_data
    }

    col1, col2 = st.columns(2)
    with col1:
        orig_label = st.selectbox("Original run", list(run_opts.keys()))
    with col2:
        masked_label = st.selectbox("Masked run", list(run_opts.keys()))

    if st.button("Compare") and orig_label and masked_label:
        orig_id = run_opts[orig_label]
        masked_id = run_opts[masked_label]
        delta = safe_api(
            "get",
            "/results/privacy_delta",
            params={"run_id_original": orig_id, "run_id_masked": masked_id},
        )
        if delta:
            st.subheader("Privacy Degradation Delta")
            cols = st.columns(3)
            cols[0].metric("Original Avg Score", delta.get("original_avg_score", "N/A"))
            cols[1].metric("Masked Avg Score", delta.get("masked_avg_score", "N/A"))
            deg = delta.get("privacy_degradation_delta")
            cols[2].metric(
                "Score Delta (masked − original)",
                f"{deg:+.4f}" if deg is not None else "N/A",
                delta=deg,
                delta_color="inverse",
            )
            st.divider()
            cols2 = st.columns(2)
            cols2[0].metric("Original Hallucination Rate", delta.get("original_hallucination_rate", "N/A"))
            cols2[1].metric("Masked Hallucination Rate", delta.get("masked_hallucination_rate", "N/A"))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Semantic Search
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔎 Semantic Search":
    st.header("Semantic Photo Search")

    models_data = safe_api("get", "/evaluation/models") or []
    embed_models = [m["name"] for m in models_data if m.get("supports_embeddings")]

    query = st.text_input("Search query", placeholder="dog in a park")
    model_choice = st.selectbox("Embedding model", embed_models)
    top_k = st.slider("Top K results", 1, 20, 5)
    privacy_mode = st.toggle("Privacy mode index", False)

    if st.button("Search", disabled=not query):
        result = api("post", "/evaluation/search", json={
            "query": query,
            "model_name": model_choice,
            "top_k": top_k,
            "privacy_mode": privacy_mode,
        })
        if result:
            st.subheader(f"Top {top_k} results for: \"{query}\"")
            for r in result.get("results", []):
                st.write(f"**Rank {r['rank']}** — image `{r['image_id'][:8]}…` — score `{r['score']}`")

            images_data = safe_api("get", "/images/") or []
            id_to_path = {img["id"]: img["filepath"] for img in images_data}
            cols = st.columns(min(top_k, 4))
            for idx, r in enumerate(result.get("results", [])):
                path = id_to_path.get(r["image_id"])
                if path:
                    try:
                        img = Image.open(path).convert("RGB")
                        img.thumbnail((200, 200))
                        cols[idx % 4].image(img, caption=f"Score: {r['score']:.3f}")
                    except Exception:
                        pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Failure Gallery
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💥 Failure Gallery":
    st.header("Failure Case Gallery")
    st.caption("Results with hallucination flags or error tags.")

    runs_data = safe_api("get", "/results/runs") or []
    all_results = []
    for run in runs_data:
        run_detail = safe_api("get", f"/results/run/{run['id']}")
        if run_detail:
            for r in run_detail.get("results", []):
                if r.get("hallucination_flag") or r.get("error_tag"):
                    r["run_model"] = run["model_name"]
                    all_results.append(r)

    if not all_results:
        st.info("No failures detected yet — great job!")
    else:
        st.warning(f"{len(all_results)} failure cases found.")
        failure_df = pd.DataFrame(all_results)
        cols_show = [c for c in ["image_id", "task_type", "run_model", "response", "hallucination_flag", "error_tag"] if c in failure_df.columns]
        st.dataframe(failure_df[cols_show], use_container_width=True)

        images_data = safe_api("get", "/images/") or []
        id_to_path = {img["id"]: img["filepath"] for img in images_data}

        st.subheader("Image previews")
        cols = st.columns(4)
        for idx, r in enumerate(all_results[:12]):
            path = id_to_path.get(r.get("image_id", ""))
            if path:
                try:
                    img = Image.open(path).convert("RGB")
                    img.thumbnail((200, 200))
                    label = "🛑 Hall." if r.get("hallucination_flag") else f"⚠ {r.get('error_tag', '')}"
                    cols[idx % 4].image(img, caption=label)
                except Exception:
                    pass
