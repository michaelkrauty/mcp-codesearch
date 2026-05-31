"""Unit tests for IndexingService helpers that do not require Qdrant/embeddings."""

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import Distance, VectorParams

from mcp_codesearch import helpers
from mcp_codesearch.indexer.discovery import FileInfo
from mcp_codesearch.services import indexing_service as idx_svc
from mcp_codesearch.services.indexing_service import IndexingService
from mcp_codesearch.storage.qdrant import EmbeddingDimMismatchError, QdrantStorage


def _make_file(name: str, content: str) -> FileInfo:
    """Build a FileInfo suitable for _prepare_files without touching disk."""
    return FileInfo(
        path=Path(f"/tmp/{name}"),
        rel_path=name,
        language="python",
        size_bytes=len(content),
        content=content,
        content_hash="deadbeef",
        line_count=content.count("\n") + 1,
        mtime=0.0,
    )


def _make_service() -> IndexingService:
    """Construct an IndexingService with stub dependencies.

    _prepare_files only touches self._global_vocab.tokenize and the static
    _chunk_embedding_text method, so the other deps can be bare mocks.
    """
    vocab = MagicMock()
    vocab.tokenize = MagicMock(return_value=[])
    return IndexingService(
        storage=MagicMock(),
        embedder=MagicMock(),
        global_vocab=vocab,
    )


class TestPrepareFilesFaultIsolation:
    """One unparseable file must not abort the whole indexing run."""

    def test_skips_file_when_chunker_raises(self, monkeypatch, caplog):
        """A chunker crash on one file logs a warning and lets the loop continue."""
        service = _make_service()

        good1 = _make_file("good1.py", "def a():\n    pass\n")
        bad = _make_file("bad.py", "def b():\n    pass\n")
        good2 = _make_file("good2.py", "def c():\n    pass\n")

        real_chunk_file = idx_svc.chunk_file

        def failing_chunk_file(content, language):
            if content == bad.content:
                raise RecursionError("simulated pathological file")
            return real_chunk_file(content, language)

        monkeypatch.setattr(idx_svc, "chunk_file", failing_chunk_file)

        with caplog.at_level("WARNING", logger="mcp_codesearch.services.indexing_service"):
            prepared, tokens_per_doc = service._prepare_files([good1, bad, good2])

        rel_paths = [p.file_info.rel_path for p in prepared]
        assert rel_paths == ["good1.py", "good2.py"]
        # The bad file should be absent from token rows too (one summary + N chunks per good file).
        assert len(tokens_per_doc) >= 2

        warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("bad.py" in m for m in warnings), warnings
        assert any("RecursionError" in m for m in warnings), warnings

    def test_all_files_prepared_when_chunker_is_healthy(self):
        """Sanity check: the happy path still works after adding the try/except."""
        service = _make_service()
        files = [
            _make_file("a.py", "def a():\n    pass\n"),
            _make_file("b.py", "def b():\n    pass\n"),
        ]

        prepared, _ = service._prepare_files(files)

        assert [p.file_info.rel_path for p in prepared] == ["a.py", "b.py"]
        # Each file should produce at least one chunk.
        assert all(len(p.chunks) >= 1 for p in prepared)


class TestVerifyEmbeddingDim:
    """Reusing a collection whose dense vectors no longer match the configured
    embedding dimension is refused, but an unknown/unreadable dim never blocks."""

    async def test_raises_on_dimension_mismatch(self, monkeypatch):
        service = _make_service()
        monkeypatch.setattr(idx_svc, "settings", SimpleNamespace(embedding_dim=4096))
        service._storage.get_dense_dim = AsyncMock(return_value=768)

        with pytest.raises(EmbeddingDimMismatchError) as excinfo:
            await service._verify_embedding_dim("codesearch_abc")

        err = excinfo.value
        assert err.collection == "codesearch_abc"
        assert err.expected == 4096
        assert err.actual == 768
        # The message names both dimensions so the cause is obvious in a log.
        assert "768" in str(err) and "4096" in str(err)

    async def test_passes_when_dimension_matches(self, monkeypatch):
        service = _make_service()
        monkeypatch.setattr(idx_svc, "settings", SimpleNamespace(embedding_dim=4096))
        service._storage.get_dense_dim = AsyncMock(return_value=4096)

        # Must not raise.
        await service._verify_embedding_dim("codesearch_abc")

    async def test_skips_storage_when_expected_dim_unknown(self, monkeypatch):
        """embedding_dim==0 means auto-detect has not resolved; do not even query."""
        service = _make_service()
        monkeypatch.setattr(idx_svc, "settings", SimpleNamespace(embedding_dim=0))
        service._storage.get_dense_dim = AsyncMock(return_value=768)

        await service._verify_embedding_dim("codesearch_abc")

        service._storage.get_dense_dim.assert_not_called()

    async def test_skips_when_stored_dim_unreadable(self, monkeypatch):
        """A stored dim of None ('cannot verify') is not treated as a mismatch."""
        service = _make_service()
        monkeypatch.setattr(idx_svc, "settings", SimpleNamespace(embedding_dim=4096))
        service._storage.get_dense_dim = AsyncMock(return_value=None)

        await service._verify_embedding_dim("codesearch_abc")


class TestGetDenseDim:
    """get_dense_dim reads the stored 'dense' vector size, or None if absent."""

    @staticmethod
    def _storage_with_vectors(monkeypatch, vectors):
        storage = QdrantStorage(url="http://localhost:6333")
        info = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=vectors))
        )
        client = MagicMock()
        client.get_collection = AsyncMock(return_value=info)
        monkeypatch.setattr(storage, "_get_client", AsyncMock(return_value=client))
        return storage

    async def test_reads_named_dense_vector_size(self, monkeypatch):
        storage = self._storage_with_vectors(
            monkeypatch, {"dense": SimpleNamespace(size=4096)}
        )
        assert await storage.get_dense_dim("codesearch_abc") == 4096

    async def test_returns_none_when_dense_vector_absent(self, monkeypatch):
        storage = self._storage_with_vectors(
            monkeypatch, {"other": SimpleNamespace(size=128)}
        )
        assert await storage.get_dense_dim("codesearch_abc") is None

    async def test_returns_none_for_single_unnamed_vector(self, monkeypatch):
        # A collection with one unnamed vector exposes VectorParams, not a dict.
        storage = self._storage_with_vectors(monkeypatch, SimpleNamespace(size=128))
        assert await storage.get_dense_dim("codesearch_abc") is None

    async def test_reads_real_qdrant_vectorparams(self, monkeypatch):
        # Guard against the real qdrant-client model: named vectors are a dict of
        # name -> VectorParams, and the dimension lives on `.size`.
        storage = self._storage_with_vectors(
            monkeypatch, {"dense": VectorParams(size=4096, distance=Distance.COSINE)}
        )
        assert await storage.get_dense_dim("codesearch_abc") == 4096


class TestIndexGuardBranch:
    """index() runs the dim guard when reusing an existing collection, and
    force=True skips it (recreating the collection is the escape hatch)."""

    @staticmethod
    def _ready_service(monkeypatch) -> IndexingService:
        """A service with the cross-process lock and stale-lock cleanup neutralized
        so index() can be driven without filesystem side effects."""
        service = _make_service()
        service._stale_locks_cleaned = True  # skip cleanup_stale_locks()

        @asynccontextmanager
        async def _noop_lock(name):
            yield

        monkeypatch.setattr(idx_svc, "async_file_lock", _noop_lock)
        return service

    async def test_force_reindex_skips_guard(self, monkeypatch):
        service = self._ready_service(monkeypatch)
        service._storage.collection_exists = AsyncMock(return_value=True)
        service._storage.delete_collection = AsyncMock()
        service._storage.create_collection = AsyncMock()
        service._verify_embedding_dim = AsyncMock()
        # discover_files returns nothing, so _full_index returns early.
        monkeypatch.setattr(idx_svc, "discover_files", lambda path: [])

        await service.index("/proj", force=True)

        service._verify_embedding_dim.assert_not_called()
        service._storage.create_collection.assert_awaited_once()

    async def test_existing_collection_invokes_guard(self, monkeypatch):
        service = self._ready_service(monkeypatch)
        service._storage.collection_exists = AsyncMock(return_value=True)
        service._storage.get_indexed_files_metadata = AsyncMock(return_value={})
        service._verify_embedding_dim = AsyncMock()
        # No changes detected -> index() returns after the guard runs.
        monkeypatch.setattr(
            idx_svc, "detect_changes_fast",
            lambda path, meta: SimpleNamespace(has_changes=False),
        )

        result = await service.index("/proj")

        service._verify_embedding_dim.assert_awaited_once()
        assert result == (0, 0, None)


class TestAutoIndexDimMismatchSurface:
    """auto_index turns a dim mismatch into an actionable force_reindex message."""

    async def test_maps_dim_mismatch_to_force_reindex_hint(self, monkeypatch):
        svc = MagicMock()
        svc.index = AsyncMock(
            side_effect=EmbeddingDimMismatchError(
                "codesearch_abc", expected=4096, actual=768
            )
        )

        async def fake_get_indexing_service():
            return svc

        monkeypatch.setattr(helpers, "get_indexing_service", fake_get_indexing_service)

        files, chunks, stats, error = await helpers.auto_index("/home/user/proj")

        assert (files, chunks, stats) == (0, 0, None)
        assert 'force_reindex(path="/home/user/proj")' in error
        assert "different embedding model" in error
