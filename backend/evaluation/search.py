"""Semantic photo search using CLIP-style image-text embeddings.

Supports both in-memory FAISS index (fast) and a simple numpy fallback
for environments where FAISS is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image

from backend.models.base import BaseModelAdapter

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    image_id: str
    score: float  # cosine similarity 0–1
    rank: int


@dataclass
class SearchQueryResult:
    query: str
    model_name: str
    results: List[SearchResult]
    top_k: int


class ImageIndex:
    """In-memory image embedding index with text query support."""

    def __init__(self, model: BaseModelAdapter):
        if not model.supports_embeddings:
            raise ValueError(f"Model {model.name} does not support embeddings")
        self.model = model
        self._image_ids: List[str] = []
        self._embeddings: np.ndarray | None = None
        self._use_faiss = False
        self._index = None

    def build(self, images: List[Image.Image], image_ids: List[str]) -> None:
        """Encode all images and build the search index."""
        logger.info("Building image index with %d images …", len(images))
        embs = [np.array(self.model.embed_image(img)) for img in images]
        self._image_ids = image_ids
        self._embeddings = np.stack(embs).astype("float32")

        try:
            import faiss  # type: ignore

            dim = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)  # inner product on L2-normed = cosine sim
            self._index.add(self._embeddings)
            self._use_faiss = True
            logger.info("FAISS index built (dim=%d, n=%d).", dim, len(images))
        except ImportError:
            logger.info("FAISS not available; using numpy cosine search.")

    def search(self, query: str, top_k: int = 5) -> SearchQueryResult:
        """Search the index with a text query."""
        if self._embeddings is None:
            raise RuntimeError("Index not built. Call build() first.")

        query_emb = np.array(self.model.embed_text(query)).astype("float32")

        if self._use_faiss and self._index is not None:
            scores, indices = self._index.search(query_emb.reshape(1, -1), min(top_k, len(self._image_ids)))
            results = [
                SearchResult(
                    image_id=self._image_ids[int(idx)],
                    score=float(scores[0][rank]),
                    rank=rank + 1,
                )
                for rank, idx in enumerate(indices[0])
                if idx >= 0
            ]
        else:
            sims = self._embeddings @ query_emb
            order = np.argsort(sims)[::-1][:top_k]
            results = [
                SearchResult(
                    image_id=self._image_ids[int(i)],
                    score=float(sims[i]),
                    rank=rank + 1,
                )
                for rank, i in enumerate(order)
            ]

        logger.info("[search] query='%s' | top_k=%d | best=%.3f", query, top_k, results[0].score if results else 0)
        return SearchQueryResult(
            query=query,
            model_name=self.model.name,
            results=results,
            top_k=top_k,
        )

    def precision_at_k(
        self,
        query: str,
        relevant_ids: List[str],
        k: int = 5,
    ) -> float:
        """Compute Precision@K for a known-relevant set."""
        result = self.search(query, top_k=k)
        retrieved = {r.image_id for r in result.results}
        relevant = set(relevant_ids)
        return len(retrieved & relevant) / k if k > 0 else 0.0
