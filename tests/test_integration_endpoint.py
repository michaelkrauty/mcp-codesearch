"""Integration fixtures must never bypass explicitly selected test endpoints."""

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
from vector_core.settings import settings


async def test_fixture_probes_and_client_use_configured_endpoint(monkeypatch):
    endpoint = "http://isolated-test.invalid:6333"
    monkeypatch.setattr(settings, "qdrant_url", endpoint)
    get = Mock(return_value=SimpleNamespace(status_code=503))
    monkeypatch.setattr(httpx, "get", get)
    namespace = runpy.run_path(str(Path(__file__).parent / "integration" / "conftest.py"))
    assert get.call_count == 2
    assert all(call.args[0] == f"{endpoint}/collections" for call in get.call_args_list)

    fixture = namespace["qdrant_storage"].__wrapped__
    storage = SimpleNamespace(close=AsyncMock())

    def create_storage(**kwargs):
        assert not kwargs, "The fixture must not override the configured endpoint"
        return storage

    monkeypatch.setitem(fixture.__globals__, "QdrantStorage", create_storage)
    generator = fixture()
    assert await anext(generator) is storage
    await generator.aclose()
