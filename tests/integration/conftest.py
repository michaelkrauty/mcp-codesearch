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
