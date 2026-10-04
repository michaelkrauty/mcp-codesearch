"""Test bootstrap isolates persistence and defaults to unavailable service endpoints."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("explicit_endpoints", [False, True])
def test_bootstrap_isolates_cache_and_service_defaults(tmp_path, explicit_endpoints):
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    sentinel = ambient / "sentinel"
    sentinel.write_text("untouched")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("VECTOR_")}
    environment.update(
        VECTOR_CACHE_DIR=str(ambient),
        VECTOR_SHARED_DATA_DIR=str(ambient),
        NOTES_DIR=str(ambient),
        VECTOR_EMBEDDING_DIM="128",
    )
    endpoint = "http://explicit.invalid" if explicit_endpoints else "http://127.0.0.1:1"
    if explicit_endpoints:
        environment.update(VECTOR_EMBEDDING_URL=endpoint, VECTOR_QDRANT_URL=endpoint)
    script = """
import asyncio
import json
import os
from unittest.mock import AsyncMock
from tests.conftest import TEST_DATA_DIR
from mcp_codesearch.settings import settings
from vector_core import EmbeddingClient

async def write_cache():
    async with EmbeddingClient(
        model="isolated-cache-proof", dim=2, profile="raw", cache_namespace="proof"
    ) as client:
        client.resolve_identity = AsyncMock(return_value=client.configured_identity())
        client._embed_prepared_batch = AsyncMock(return_value=[[1.0, 0.0]])
        await client.embed_all(["synthetic input"])
        assert client._cache_path.is_file()
        assert client._cache_path.is_relative_to(TEST_DATA_DIR)

asyncio.run(write_cache())
assert settings.cache_dir == TEST_DATA_DIR / "cache"
assert settings.shared_data_dir == TEST_DATA_DIR / "shared"
assert os.environ["NOTES_DIR"] == str(TEST_DATA_DIR / "notes")
print(json.dumps({"root": str(TEST_DATA_DIR), "embedding": settings.embedding_url,
                 "qdrant": settings.qdrant_url}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    proof = json.loads(result.stdout)
    assert proof["embedding"] == proof["qdrant"] == endpoint
    assert not Path(proof["root"]).exists()
    assert list(ambient.iterdir()) == [sentinel]
    assert sentinel.read_text() == "untouched"
