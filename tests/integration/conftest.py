"""Fixtures for integration tests."""

from uuid import uuid4

import pytest
from vector_core.embeddings.sparse import SparseVector

from mcp_codesearch.storage.qdrant import (
    ChunkPoint,
    FilePoint,
    QdrantStorage,
)


def qdrant_available() -> bool:
    """Check if Qdrant is running."""
    import httpx
    try:
        response = httpx.get("http://localhost:6333/collections", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not qdrant_available(),
    reason="Qdrant not available at localhost:6333"
)


def qdrant_and_embeddings_available() -> bool:
    """Whether the full stack can actually serve these tests.

    The embedding half is checked by embedding something, not by pinging
    /v1/models: a service can list models while rejecting the configured one,
    and only a real response reveals the vector width. That width has to match
    what settings expect, because a mismatch fails every upsert for reasons
    that say nothing about the code under test.
    """
    import httpx

    from mcp_codesearch.settings import settings

    try:
        if (
            httpx.get(f"{settings.qdrant_url}/collections", timeout=2.0).status_code
            != 200
        ):
            return False

        response = httpx.post(
            f"{settings.embedding_url.rstrip('/')}/v1/embeddings",
            json={"model": settings.embedding_model, "input": "probe"},
            timeout=10.0,
        )
        if response.status_code != 200:
            return False
        return (
            len(response.json()["data"][0]["embedding"]) == settings.embedding_dim
        )
    except Exception:
        return False


# Tests that index or search for real need both services. Without them the
# work fails at the first embedding call, which says nothing about the code
# under test, so they are skipped rather than failed.
requires_full_stack = pytest.mark.skipif(
    not qdrant_and_embeddings_available(),
    reason="Qdrant and/or embedding service not available"
)


@pytest.fixture
def test_collection_name():
    """Generate unique test collection name."""
    return f"codesearch_test_{uuid4().hex[:12]}"


@pytest.fixture
async def qdrant_storage():
    """Create QdrantStorage instance for testing."""
    storage = QdrantStorage(url="http://localhost:6333")
    yield storage
    await storage.close()


@pytest.fixture
async def test_collection(qdrant_storage, test_collection_name):
    """Create a test collection and clean up after."""
    await qdrant_storage.create_collection(test_collection_name)
    yield test_collection_name
    try:
        await qdrant_storage.delete_collection(test_collection_name)
    except Exception:
        pass


@pytest.fixture
def sample_dense_vector():
    """Sample dense vector for testing (dimension from settings)."""
    from mcp_codesearch.settings import settings
    return [0.1] * settings.embedding_dim


@pytest.fixture
def sample_sparse_vector():
    """Sample sparse vector for testing."""
    return SparseVector(
        indices=[0, 5, 10, 15, 100],
        values=[0.5, 0.3, 0.2, 0.15, 0.1],
    )


@pytest.fixture
def sample_file_point():
    """Sample file point for testing."""
    return FilePoint(
        path="src/main.py",
        abs_path="/project/src/main.py",
        language="python",
        file_hash="abc123def456",
        summary="Main module for testing",
        line_count=50,
        size_bytes=1024,
        mtime=1704067200.0,
    )


@pytest.fixture
def sample_chunk_point():
    """Sample chunk point for testing."""
    return ChunkPoint(
        path="src/main.py",
        abs_path="/project/src/main.py",
        language="python",
        file_hash="abc123def456",
        chunk_type="function",
        name="main",
        start_line=10,
        end_line=25,
        content="def main():\n    print('Hello')",
        context=None,
    )
