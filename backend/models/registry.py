"""Central model registry.

Usage
-----
    from backend.models.registry import get_model

    blip = get_model("blip-base", device="cpu")
    clip = get_model("clip-vit-base-patch32", device="cpu")
"""

from __future__ import annotations

import logging
from typing import Type

from backend.config import get_settings
from backend.models.base import BaseModelAdapter
from backend.models.blip_adapter import BLIPAdapter
from backend.models.clip_adapter import CLIPAdapter

logger = logging.getLogger(__name__)
settings = get_settings()

# Map of model_name -> adapter class
_REGISTRY: dict[str, Type[BaseModelAdapter]] = {
    BLIPAdapter.name: BLIPAdapter,
    CLIPAdapter.name: CLIPAdapter,
}

# Singleton cache so we don't reload weights on every request
_INSTANCES: dict[str, BaseModelAdapter] = {}


def register_model(adapter_cls: Type[BaseModelAdapter]) -> None:
    """Register a custom adapter at runtime."""
    _REGISTRY[adapter_cls.name] = adapter_cls
    logger.info("Registered model adapter: %s", adapter_cls.name)


def get_model(name: str, device: str | None = None) -> BaseModelAdapter:
    """Return a (cached) loaded model adapter instance."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    if name not in _INSTANCES:
        device = device or settings.device
        adapter = _REGISTRY[name](
            device=device,
            cache_dir=str(settings.model_cache_path),
        )
        _INSTANCES[name] = adapter
    return _INSTANCES[name]


def available_models() -> list[dict]:
    """Return metadata for all registered models."""
    return [
        {
            "name": name,
            "supports_captioning": cls.supports_captioning,
            "supports_vqa": cls.supports_vqa,
            "supports_embeddings": cls.supports_embeddings,
        }
        for name, cls in _REGISTRY.items()
    ]
