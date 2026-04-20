"""Abstract base class for all model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image


class BaseModelAdapter(ABC):
    """Common interface every model adapter must implement."""

    name: str = "base"
    supports_captioning: bool = False
    supports_vqa: bool = False
    supports_embeddings: bool = False

    def __init__(self, device: str = "cpu", cache_dir: str | None = None):
        self.device = device
        self.cache_dir = cache_dir
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    # ------------------------------------------------------------------
    # Optional capabilities — subclasses override what they support
    # ------------------------------------------------------------------

    def caption(self, image: Image.Image) -> str:
        raise NotImplementedError(f"{self.name} does not support captioning")

    def answer(self, image: Image.Image, question: str) -> str:
        raise NotImplementedError(f"{self.name} does not support VQA")

    def embed_image(self, image: Image.Image) -> list[float]:
        raise NotImplementedError(f"{self.name} does not support image embedding")

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError(f"{self.name} does not support text embedding")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def load_image(path: str | Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} device={self.device}>"
