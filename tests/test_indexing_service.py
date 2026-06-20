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
from mcp_codesearch.storage import qdrant as storage_qdrant
from mcp_codesearch.storage.qdrant import (
    EmbeddingDimMismatchError,
    EmbeddingModelMismatchError,
    QdrantStorage,
)


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
        service._verify_embedding_model = AsyncMock()
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
        service._verify_embedding_model = AsyncMock()
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


class TestVerifyEmbeddingModel:
    """Reusing a collection recorded under a different embedding model is
    refused even when dimensions match; unknown/legacy metadata never blocks."""

    @staticmethod
    def _service_with_metadata(monkeypatch, metadata, model="Qwen3-Embedding-8B", dim=4096):
        service = _make_service()
        monkeypatch.setattr(
            idx_svc, "settings",
            SimpleNamespace(embedding_model=model, embedding_dim=dim),
        )
        service._storage.get_metadata = AsyncMock(return_value=metadata)
        service._storage.store_metadata = AsyncMock()
        return service

    async def test_raises_on_model_mismatch(self, monkeypatch):
        service = self._service_with_metadata(
            monkeypatch, {"codebase_path": "/proj", "embedding_model": "bge-large-en"}
        )

        with pytest.raises(EmbeddingModelMismatchError) as excinfo:
            await service._verify_embedding_model("codesearch_abc", "/proj")

        err = excinfo.value
        assert err.collection == "codesearch_abc"
        assert err.expected == "Qwen3-Embedding-8B"
        assert err.actual == "bge-large-en"
        # The message names both models so the cause is obvious in a log.
        assert "bge-large-en" in str(err) and "Qwen3-Embedding-8B" in str(err)
        service._storage.store_metadata.assert_not_called()

    async def test_passes_when_model_matches(self, monkeypatch):
        service = self._service_with_metadata(
            monkeypatch,
            {"codebase_path": "/proj", "embedding_model": "Qwen3-Embedding-8B"},
        )

        await service._verify_embedding_model("codesearch_abc", "/proj")

        service._storage.store_metadata.assert_not_called()

    async def test_skips_storage_when_no_model_configured(self, monkeypatch):
        """An empty configured model (auto-detect setups) disables the guard."""
        service = self._service_with_metadata(monkeypatch, None, model="")

        await service._verify_embedding_model("codesearch_abc", "/proj")

        service._storage.get_metadata.assert_not_called()
        service._storage.store_metadata.assert_not_called()

    async def test_backfills_when_metadata_missing(self, monkeypatch):
        """A collection with no metadata point gets stamped with the current model."""
        service = self._service_with_metadata(monkeypatch, None)

        await service._verify_embedding_model("codesearch_abc", "/proj")

        service._storage.store_metadata.assert_awaited_once_with("codesearch_abc", "/proj")

    async def test_backfills_when_model_key_absent(self, monkeypatch):
        """Metadata written before this guard exists lacks the model key."""
        service = self._service_with_metadata(monkeypatch, {"codebase_path": "/proj"})

        await service._verify_embedding_model("codesearch_abc", "/proj")

        service._storage.store_metadata.assert_awaited_once_with("codesearch_abc", "/proj")

    async def test_backfill_skipped_while_dim_unresolved(self, monkeypatch):
        """Metadata writes need a placeholder dense vector; don't stamp at dim 0."""
        service = self._service_with_metadata(monkeypatch, None, dim=0)

        await service._verify_embedding_model("codesearch_abc", "/proj")

        service._storage.store_metadata.assert_not_called()

    async def test_non_string_stored_model_fails_open(self, monkeypatch):
        """A JSON-coerced or foreign stored value is 'cannot verify', not a mismatch."""
        service = self._service_with_metadata(
            monkeypatch, {"codebase_path": "/proj", "embedding_model": 123}
        )

        await service._verify_embedding_model("codesearch_abc", "/proj")

        service._storage.store_metadata.assert_not_called()


class TestIndexBranchInvokesModelGuard:
    """The model guard runs exactly on the reuse branch, like the dim guard."""

    @staticmethod
    def _ready_service(monkeypatch) -> IndexingService:
        service = _make_service()
        service._stale_locks_cleaned = True

        @asynccontextmanager
        async def _noop_lock(name):
            yield

        monkeypatch.setattr(idx_svc, "async_file_lock", _noop_lock)
        return service

    async def test_force_reindex_skips_model_guard(self, monkeypatch):
        service = self._ready_service(monkeypatch)
        service._storage.collection_exists = AsyncMock(return_value=True)
        service._storage.delete_collection = AsyncMock()
        service._storage.create_collection = AsyncMock()
        service._verify_embedding_dim = AsyncMock()
        service._verify_embedding_model = AsyncMock()
        monkeypatch.setattr(idx_svc, "discover_files", lambda path: [])

        await service.index("/proj", force=True)

        service._verify_embedding_model.assert_not_called()

    async def test_existing_collection_invokes_model_guard(self, monkeypatch):
        service = self._ready_service(monkeypatch)
        service._storage.collection_exists = AsyncMock(return_value=True)
        service._storage.get_indexed_files_metadata = AsyncMock(return_value={})
        service._verify_embedding_dim = AsyncMock()
        service._verify_embedding_model = AsyncMock()
        monkeypatch.setattr(
            idx_svc, "detect_changes_fast",
            lambda path, meta: SimpleNamespace(has_changes=False),
        )

        result = await service.index("/proj")

        service._verify_embedding_model.assert_awaited_once()
        # The guard receives the resolved absolute path for backfill stamping.
        args = service._verify_embedding_model.await_args.args
        assert args[1] == str(Path("/proj").resolve())
        assert result == (0, 0, None)


class TestAutoIndexModelMismatchSurface:
    """auto_index turns a model mismatch into an actionable force_reindex message."""

    async def test_maps_model_mismatch_to_force_reindex_hint(self, monkeypatch):
        svc = MagicMock()
        svc.index = AsyncMock(
            side_effect=EmbeddingModelMismatchError(
                "codesearch_abc",
                expected="Qwen3-Embedding-8B",
                actual="bge-large-en",
            )
        )

        async def fake_get_indexing_service():
            return svc

        monkeypatch.setattr(helpers, "get_indexing_service", fake_get_indexing_service)

        files, chunks, stats, error = await helpers.auto_index("/home/user/proj")

        assert (files, chunks, stats) == (0, 0, None)
        assert 'force_reindex(path="/home/user/proj")' in error
        assert "different embedding model" in error
        assert "meaningless" in error


class TestStoreMetadataRecordsModel:
    """The storage wrapper records the configured embedding model."""

    @staticmethod
    def _storage(monkeypatch, model):
        storage = QdrantStorage(url="http://localhost:6333")
        storage._core = MagicMock()
        storage._core.store_metadata = AsyncMock()
        monkeypatch.setattr(
            storage_qdrant, "settings", SimpleNamespace(embedding_model=model)
        )
        return storage

    async def test_records_model_when_configured(self, monkeypatch):
        storage = self._storage(monkeypatch, "Qwen3-Embedding-8B")

        await storage.store_metadata("codesearch_abc", "/proj")

        storage._core.store_metadata.assert_awaited_once_with(
            "codesearch_abc",
            {"codebase_path": "/proj", "embedding_model": "Qwen3-Embedding-8B"},
        )

    async def test_omits_model_when_unconfigured(self, monkeypatch):
        storage = self._storage(monkeypatch, "")

        await storage.store_metadata("codesearch_abc", "/proj")

        storage._core.store_metadata.assert_awaited_once_with(
            "codesearch_abc", {"codebase_path": "/proj"}
        )


class TestForceReindexRollback:
    """A force re-index destroys the existing collection before rebuilding it.
    If the rebuild fails partway (a transient embedding/Qdrant error in Phase 2),
    the service must roll back to a clean "not indexed" state: drop this
    codebase's contribution from the shared global vocabulary (so a partial
    rebuild does not skew IDF for every other codebase) and drop the partial
    collection (so the next access does a fresh full index instead of an
    incremental one that double-counts the already-registered tokens)."""

    @staticmethod
    def _ready_service(monkeypatch) -> IndexingService:
        service = _make_service()
        service._stale_locks_cleaned = True

        @asynccontextmanager
        async def _noop_lock(name):
            yield

        monkeypatch.setattr(idx_svc, "async_file_lock", _noop_lock)
        return service

    async def test_rolls_back_vocab_and_collection_on_phase2_failure(self, monkeypatch):
        service = self._ready_service(monkeypatch)
        service._storage.collection_exists = AsyncMock(return_value=True)
        service._storage.delete_collection = AsyncMock()
        service._storage.create_collection = AsyncMock()
        # one file so _full_index proceeds past its empty-list early return
        monkeypatch.setattr(idx_svc, "discover_files", lambda path: [object()])
        # avoid real chunking; Phase 1 registers vocab, Phase 2 then fails
        service._prepare_files = MagicMock(return_value=([object()], {"doc-0": ["tok"]}))
        service._process_batch = AsyncMock(side_effect=RuntimeError("transient embed failure"))

        with pytest.raises(RuntimeError, match="transient embed failure"):
            await service.index("/proj", force=True)

        # Pre-delete unregister of the old contribution + rollback unregister == 2.
        assert service._global_vocab.unregister_codebase.call_count == 2
        # Pre-rebuild delete of the old collection + rollback delete of the partial == 2.
        assert service._storage.delete_collection.await_count == 2

    async def test_rolls_back_when_discovery_fails_after_recreate(self, monkeypatch):
        """File discovery runs after the old collection is dropped and the new
        one created. If it raises (e.g. a malformed ignore pattern), the freshly
        created empty collection must still be rolled back, or the next access
        takes the incremental path instead of a clean full rebuild."""
        service = self._ready_service(monkeypatch)
        service._storage.collection_exists = AsyncMock(return_value=True)
        service._storage.delete_collection = AsyncMock()
        service._storage.create_collection = AsyncMock()

        def _boom(path):
            raise ValueError("malformed ignore pattern")

        monkeypatch.setattr(idx_svc, "discover_files", _boom)

        with pytest.raises(ValueError, match="malformed ignore pattern"):
            await service.index("/proj", force=True)

        # Pre-rebuild delete of the old collection + rollback delete of the
        # freshly created (empty) collection == 2.
        assert service._storage.delete_collection.await_count == 2
        assert service._global_vocab.unregister_codebase.call_count == 2

    async def test_successful_force_reindex_does_not_roll_back(self, monkeypatch):
        service = self._ready_service(monkeypatch)
        service._storage.collection_exists = AsyncMock(return_value=True)
        service._storage.delete_collection = AsyncMock()
        service._storage.create_collection = AsyncMock()
        service._storage.store_metadata = AsyncMock()
        monkeypatch.setattr(idx_svc, "discover_files", lambda path: [object()])
        service._prepare_files = MagicMock(return_value=([object()], {"doc-0": ["tok"]}))
        service._process_batch = AsyncMock(return_value=1)  # succeeds

        _files, chunks, _stats = await service.index("/proj", force=True)

        # Only the pre-rebuild delete + pre-delete unregister run; no rollback.
        assert service._storage.delete_collection.await_count == 1
        assert service._global_vocab.unregister_codebase.call_count == 1
        assert chunks == 1


class TestIncrementalIndexRollback:
    """An incremental index commits the additive global-vocab delta before it
    embeds and upserts the new points. If a batch fails after that commit, the
    delta must be rolled back, or the failed files are double-counted in the
    shared vocabulary when the next run re-detects them as added (the force/full
    path has the equivalent rollback)."""

    async def test_batch_failure_rolls_back_the_vocab_delta(self, monkeypatch):
        service = _make_service()
        service._collect_removed_tokens = AsyncMock(return_value=[])
        service._prepare_files = MagicMock(return_value=([object()], [{"tok"}]))
        service._process_batch = AsyncMock(
            side_effect=RuntimeError("transient embed failure")
        )
        changes = SimpleNamespace(added=[object()], modified=[], deleted=[])

        with pytest.raises(RuntimeError, match="transient embed failure"):
            await service._incremental_index("codesearch_x", changes, "/proj")

        # The delta was applied once and undone once (inverse: added<->removed).
        calls = service._global_vocab.update_codebase_incremental.call_args_list
        assert len(calls) == 2
        orig, inv = calls[0].kwargs, calls[1].kwargs
        assert orig["added_tokens"] == [{"tok"}] and orig["removed_tokens"] == []
        assert inv["added_tokens"] == [] and inv["removed_tokens"] == [{"tok"}]
        assert inv["net_doc_change"] == -orig["net_doc_change"]

    async def test_successful_incremental_does_not_roll_back(self, monkeypatch):
        service = _make_service()
        service._collect_removed_tokens = AsyncMock(return_value=[])
        service._prepare_files = MagicMock(return_value=([object()], [{"tok"}]))
        service._process_batch = AsyncMock(return_value=1)
        service._storage.store_metadata = AsyncMock()
        changes = SimpleNamespace(added=[object()], modified=[], deleted=[])

        await service._incremental_index("codesearch_x", changes, "/proj")

        # Only the original delta; no inverse rollback.
        assert service._global_vocab.update_codebase_incremental.call_count == 1
