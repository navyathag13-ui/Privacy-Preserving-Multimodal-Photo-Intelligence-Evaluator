"""Integration tests for FastAPI routes using async test client."""

import sys
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.main import app


@pytest.mark.asyncio
class TestHealthRoute:
    async def test_health_returns_ok(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
class TestImagesRoute:
    async def test_list_images_empty(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/images/")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


@pytest.mark.asyncio
class TestEvaluationModelsRoute:
    async def test_list_models_returns_registry(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/evaluation/models")
        assert resp.status_code == 200
        models = resp.json()
        assert isinstance(models, list)
        names = [m["name"] for m in models]
        assert "blip-base" in names
        assert "clip-vit-base-patch32" in names


@pytest.mark.asyncio
class TestResultsRoute:
    async def test_list_runs_empty(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/results/runs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_summary_empty(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/results/summary")
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)
