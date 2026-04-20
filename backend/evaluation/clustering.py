"""Duplicate / near-duplicate clustering using image embeddings.

Uses hierarchical agglomerative clustering on cosine distances so no k
parameter is required.  A distance threshold of 0.2 (on L2-normalised
embeddings) works well for near-duplicate detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from PIL import Image

from backend.models.base import BaseModelAdapter

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    model_name: str
    image_ids: List[str]
    cluster_labels: List[int]
    num_clusters: int
    duplicate_pairs: List[tuple[str, str]]  # pairs with similarity > threshold


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def cluster_images(
    model: BaseModelAdapter,
    images: List[Image.Image],
    image_ids: List[str],
    threshold: float = 0.15,
) -> ClusterResult:
    """Cluster images by embedding similarity.

    Parameters
    ----------
    threshold:
        Cosine distance below which two images are considered near-duplicates.
    """
    if not model.supports_embeddings:
        raise ValueError(f"Model {model.name} does not support embeddings")

    embeddings = [np.array(model.embed_image(img)) for img in images]
    n = len(embeddings)

    # Build distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _cosine_distance(embeddings[i], embeddings[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Simple greedy clustering with threshold
    labels = [-1] * n
    cluster_id = 0
    duplicate_pairs = []

    for i in range(n):
        if labels[i] == -1:
            labels[i] = cluster_id
            for j in range(i + 1, n):
                if dist_matrix[i, j] < threshold:
                    labels[j] = cluster_id
                    duplicate_pairs.append((image_ids[i], image_ids[j]))
            cluster_id += 1

    num_clusters = len(set(labels))
    logger.info(
        "[clustering] %s | %d images → %d clusters | %d duplicate pairs",
        model.name,
        n,
        num_clusters,
        len(duplicate_pairs),
    )

    return ClusterResult(
        model_name=model.name,
        image_ids=image_ids,
        cluster_labels=labels,
        num_clusters=num_clusters,
        duplicate_pairs=duplicate_pairs,
    )
