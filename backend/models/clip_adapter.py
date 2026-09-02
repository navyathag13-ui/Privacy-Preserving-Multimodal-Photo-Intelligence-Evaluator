"""CLIP adapter — supports image and text embeddings for retrieval and clustering.

Model: openai/clip-vit-base-patch32  (small, fast, well-known baseline)

Swapping to a larger variant like ViT-L/14 or SigLIP requires only changing
CLIP_MODEL_ID and adjusting embedding_dim.
"""

from __future__ import annotations

import logging
from typing import List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from backend.models.base import BaseModelAdapter

logger = logging.getLogger(__name__)

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512


class CLIPAdapter(BaseModelAdapter):
    """CLIP image and text embedding adapter."""

    name = "clip-vit-base-patch32"
    supports_captioning = False
    supports_vqa = False
    supports_embeddings = True

    def load(self) -> None:
        logger.info("Loading CLIP model (%s) …", CLIP_MODEL_ID)
        self._processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_ID, cache_dir=self.cache_dir
        )
        self._model = CLIPModel.from_pretrained(
            CLIP_MODEL_ID, cache_dir=self.cache_dir
        ).to(self.device)
        self._model.eval()
        logger.info("CLIP model loaded.")

    def embed_image(self, image: Image.Image) -> List[float]:
        """Return L2-normalised image embedding."""
        self.ensure_loaded()
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()

    def embed_text(self, text: str) -> List[float]:
        """Return L2-normalised text embedding."""
        self.ensure_loaded()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()

    def similarity(self, image: Image.Image, text: str) -> float:
        """Cosine similarity between an image and a text query (range ≈ 0–1)."""
        img_emb = torch.tensor(self.embed_image(image))
        txt_emb = torch.tensor(self.embed_text(text))
        return float(torch.dot(img_emb, txt_emb).item())
