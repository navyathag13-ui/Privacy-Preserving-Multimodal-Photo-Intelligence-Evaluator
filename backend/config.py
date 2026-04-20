"""Application configuration loaded from environment variables."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "dev-secret-key"

    # Model
    device: str = "cpu"
    model_cache_dir: str = "./model_cache"

    # Storage
    database_url: str = "sqlite+aiosqlite:///./data/evaluations.db"
    results_parquet_dir: str = "./data/results"

    # Privacy defaults
    enable_face_masking: bool = True
    enable_text_masking: bool = True
    face_blur_strength: int = 25

    # API
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_size_mb: int = 50

    # Data paths
    data_dir: str = "./data"
    images_dir: str = "./data/images"
    metadata_dir: str = "./data/metadata"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def images_path(self) -> Path:
        return Path(self.images_dir)

    @property
    def metadata_path(self) -> Path:
        return Path(self.metadata_dir)

    @property
    def model_cache_path(self) -> Path:
        p = Path(self.model_cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache()
def get_settings() -> Settings:
    return Settings()
