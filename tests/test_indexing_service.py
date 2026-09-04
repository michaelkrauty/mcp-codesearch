"""Unit tests for IndexingService helpers that do not require Qdrant/embeddings."""

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import Distance, VectorParams
from vector_core import GlobalVocabulary

from mcp_codesearch import helpers
from mcp_codesearch.indexer.chunker import (
    build_chunk_vocabulary_text,
    truncate_chunk_content,
)
from mcp_codesearch.indexer.discovery import FileInfo
from mcp_codesearch.indexer.treesitter import Chunk
from mcp_codesearch.services import indexing_service as idx_svc
from mcp_codesearch.services.indexing_service import IndexingService
from mcp_codesearch.settings import settings
from mcp_codesearch.storage import qdrant as storage_qdrant
from mcp_codesearch.storage.qdrant import (
    EmbeddingDeploymentMismatchError,
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

    _prepare_files only touches self._global_vocab.tokenize and text-building
    helpers, so the other deps can be bare mocks.
    """
    vocab = MagicMock()
    vocab.tokenize = MagicMock(return_value=[])
    vocab.get_codebase_doc_count = MagicMock(return_value=0)
    vocab.get_codebase_ids = MagicMock(return_value=[])
    vocab.get_tokens_by_indices = MagicMock(return_value={})
    storage = MagicMock()
    storage.count_index_documents = AsyncMock(return_value=0)
    storage.get_metadata = AsyncMock(return_value=None)
    storage.store_metadata = AsyncMock()
    journal = MagicMock()
    journal.get.return_value = None
    journal.list.return_value = []
    return IndexingService(
        storage=storage,
        embedder=MagicMock(),
        global_vocab=vocab,
        journal=journal,
    )


async def test_run_sync_waits_for_worker_before_propagating_cancellation() -> None:
    """A cancelled request cannot leave a vocabulary mutation running."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_write() -> None:
        started.set()
        release.wait()
        finished.set()

    task = asyncio.create_task(idx_svc._run_sync(blocking_write))
    assert await asyncio.to_thread(started.wait, 1.0)

    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()

    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


async def test_run_sync_propagates_worker_cancellation_without_spinning() -> None:
    """A worker raising CancelledError terminates instead of looping forever."""

    def cancelled() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(idx_svc._run_sync(cancelled), timeout=1.0)


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
        info = SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))
        client = MagicMock()
        client.get_collection = AsyncMock(return_value=info)
        monkeypatch.setattr(storage, "_get_client", AsyncMock(return_value=client))
        return storage

    async def test_reads_named_dense_vector_size(self, monkeypatch):
        storage = self._storage_with_vectors(monkeypatch, {"dense": SimpleNamespace(size=4096)})
        assert await storage.get_dense_dim("codesearch_abc") == 4096

    async def test_returns_none_when_dense_vector_absent(self, monkeypatch):
        storage = self._storage_with_vectors(monkeypatch, {"other": SimpleNamespace(size=128)})
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
        async def _noop_lock(name, **_kwargs):
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
            idx_svc,
            "detect_changes_fast",
            lambda path, meta: SimpleNamespace(has_changes=False),
        )

        result = await service.index("/proj")

        service._verify_embedding_dim.assert_awaited_once()
        assert result == (0, 0, None)

    async def test_force_resumes_collection_marked_in_progress(self, monkeypatch):
        service = self._ready_service(monkeypatch)
        service._storage.collection_exists = AsyncMock(return_value=True)
        service._storage.get_metadata = AsyncMock(return_value={"indexing_in_progress": True})
        service._storage.get_indexed_files_metadata = AsyncMock(return_value={})
        service._storage.delete_collection = AsyncMock()
        service._storage.create_collection = AsyncMock()
        service._storage.store_metadata = AsyncMock()
        service._verify_embedding_dim = AsyncMock()
        service._verify_embedding_model = AsyncMock()
        service._verify_embedding_deployment = AsyncMock()
        monkeypatch.setattr(
            idx_svc,
            "detect_changes_fast",
            lambda path, metadata: SimpleNamespace(has_changes=False),
        )

        assert await service.index("/proj", force=True) == (0, 0, None)

        service._storage.delete_collection.assert_not_awaited()
        service._storage.create_collection.assert_not_awaited()
        service._storage.store_metadata.assert_awaited_once_with(
            "codesearch_79903f0c2002",
            "/proj",
            indexing_in_progress=False,
        )

    async def test_force_restarts_incompatible_in_progress_collection(self, monkeypatch):
        service = self._ready_service(monkeypatch)
        service._storage.collection_exists = AsyncMock(return_value=True)
        service._storage.get_metadata = AsyncMock(return_value={"indexing_in_progress": True})
        service._storage.delete_collection = AsyncMock()
        service._storage.create_collection = AsyncMock()
        service._storage.store_metadata = AsyncMock()
        service._verify_embedding_dim = AsyncMock(
            side_effect=EmbeddingDimMismatchError(
                "codesearch_79903f0c2002",
                expected=4096,
                actual=768,
            )
        )
        service._verify_embedding_model = AsyncMock()
        service._verify_embedding_deployment = AsyncMock()
        monkeypatch.setattr(idx_svc, "discover_files", lambda path: [])

        await service.index("/proj", force=True)

        service._storage.delete_collection.assert_awaited_once()
        service._storage.create_collection.assert_awaited_once()

    async def test_fresh_index_removes_contribution_without_count(self, monkeypatch):
        service = self._ready_service(monkeypatch)
        collection = "codesearch_79903f0c2002"
        service._storage.collection_exists = AsyncMock(return_value=False)
        service._storage.create_collection = AsyncMock()
        service._storage.store_metadata = AsyncMock()
        service._global_vocab.get_codebase_ids = MagicMock(return_value=[collection])
        monkeypatch.setattr(idx_svc, "discover_files", lambda path: [])

        await service.index("/proj")

        service._global_vocab.unregister_codebase.assert_called_once_with(collection)

    async def test_new_collection_is_marked_before_discovery(self, monkeypatch):
        service = self._ready_service(monkeypatch)
        service._storage.collection_exists = AsyncMock(return_value=False)
        service._storage.create_collection = AsyncMock()
        service._storage.store_metadata = AsyncMock()
        service._global_vocab.get_codebase_doc_count = MagicMock(return_value=0)

        def fail_discovery(_path):
            raise RuntimeError("discovery failed")

        monkeypatch.setattr(idx_svc, "discover_files", fail_discovery)

        with pytest.raises(RuntimeError, match="discovery failed"):
            await service.index("/proj")

        service._storage.store_metadata.assert_awaited_once_with(
            "codesearch_79903f0c2002",
            "/proj",
            indexing_in_progress=True,
        )


class TestAutoIndexDimMismatchSurface:
    """auto_index turns a dim mismatch into an actionable force_reindex message."""

    async def test_maps_dim_mismatch_to_force_reindex_hint(self, monkeypatch):
        svc = MagicMock()
        svc.index = AsyncMock(
            side_effect=EmbeddingDimMismatchError("codesearch_abc", expected=4096, actual=768)
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
            idx_svc,
            "settings",
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
        async def _noop_lock(name, **_kwargs):
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
            idx_svc,
            "detect_changes_fast",
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


class TestEmbeddingDeploymentGuard:
    async def test_rejects_different_explicit_deployment(self, monkeypatch):
        service = _make_service()
        monkeypatch.setattr(
            idx_svc,
            "settings",
            SimpleNamespace(
                embedding_cache_namespace="revision-b",
                embedding_dim=4096,
            ),
        )
        service._storage.get_metadata = AsyncMock(
            return_value={"embedding_cache_namespace": "revision-a"}
        )

        with pytest.raises(EmbeddingDeploymentMismatchError):
            await service._verify_embedding_deployment("codesearch_abc", "/proj")

    async def test_backfills_legacy_deployment_identity(self, monkeypatch):
        service = _make_service()
        monkeypatch.setattr(
            idx_svc,
            "settings",
            SimpleNamespace(
                embedding_cache_namespace="revision-a",
                embedding_dim=4096,
            ),
        )
        service._storage.get_metadata = AsyncMock(return_value={})
        service._storage.store_metadata = AsyncMock()

        await service._verify_embedding_deployment("codesearch_abc", "/proj")

        service._storage.store_metadata.assert_awaited_once_with("codesearch_abc", "/proj")


class TestStoreMetadataRecordsModel:
    """The storage wrapper records the configured embedding model."""

    @staticmethod
    def _storage(monkeypatch, model, namespace=None):
        storage = QdrantStorage(url="http://localhost:6333")
        storage._core = MagicMock()
        storage._core.get_metadata = AsyncMock(return_value=None)
        storage._core.store_metadata = AsyncMock()
        monkeypatch.setattr(
            storage_qdrant,
            "settings",
            SimpleNamespace(
                embedding_model=model,
                embedding_cache_namespace=namespace,
            ),
        )
        return storage

    async def test_records_model_when_configured(self, monkeypatch):
        storage = self._storage(monkeypatch, "Qwen3-Embedding-8B")

        await storage.store_metadata("codesearch_abc", "/proj")

        storage._core.store_metadata.assert_awaited_once_with(
            "codesearch_abc",
            {
                "codebase_path": "/proj",
                "embedding_model": "Qwen3-Embedding-8B",
                "indexing_in_progress": False,
            },
        )

    async def test_records_cache_namespace_when_configured(self, monkeypatch):
        storage = self._storage(
            monkeypatch,
            "Qwen3-Embedding-8B",
            namespace="revision-a",
        )

        await storage.store_metadata("codesearch_abc", "/proj")

        storage._core.store_metadata.assert_awaited_once_with(
            "codesearch_abc",
            {
                "codebase_path": "/proj",
                "embedding_model": "Qwen3-Embedding-8B",
                "embedding_cache_namespace": "revision-a",
                "indexing_in_progress": False,
            },
        )

    async def test_omits_model_when_unconfigured(self, monkeypatch):
        storage = self._storage(monkeypatch, "")

        await storage.store_metadata("codesearch_abc", "/proj")

        storage._core.store_metadata.assert_awaited_once_with(
            "codesearch_abc",
            {"codebase_path": "/proj", "indexing_in_progress": False},
        )


class TestForceReindexResume:
    """A failed rebuild retains only its complete, resumable batches."""

    @staticmethod
    def _ready_service(monkeypatch) -> IndexingService:
        service = _make_service()
        service._stale_locks_cleaned = True

        @asynccontextmanager
        async def _noop_lock(name, **_kwargs):
            yield

        monkeypatch.setattr(idx_svc, "async_file_lock", _noop_lock)
        return service

    async def test_phase2_failure_keeps_the_new_collection_for_resume(self, monkeypatch):
        service = self._ready_service(monkeypatch)
        service._storage.collection_exists = AsyncMock(return_value=True)
        service._storage.delete_collection = AsyncMock()
        service._storage.create_collection = AsyncMock()
        monkeypatch.setattr(idx_svc, "discover_files", lambda path: [object()])
        service._prepare_files = MagicMock(return_value=([_prepared("one.py")], [{"tok"}]))
        service._process_batch = AsyncMock(side_effect=RuntimeError("transient embed failure"))

        with pytest.raises(RuntimeError, match="transient embed failure"):
            await service.index("/proj", force=True)

        assert service._global_vocab.unregister_codebase.call_count == 1
        assert service._storage.delete_collection.await_count == 1
        service._storage.create_collection.assert_awaited_once()

    async def test_rolls_back_when_discovery_fails_after_recreate(self, monkeypatch):
        """An empty recreated collection remains a safe incremental resume point."""
        service = self._ready_service(monkeypatch)
        service._storage.collection_exists = AsyncMock(return_value=True)
        service._storage.delete_collection = AsyncMock()
        service._storage.create_collection = AsyncMock()

        def _boom(path):
            raise ValueError("malformed ignore pattern")

        monkeypatch.setattr(idx_svc, "discover_files", _boom)

        with pytest.raises(ValueError, match="malformed ignore pattern"):
            await service.index("/proj", force=True)

        assert service._storage.delete_collection.await_count == 1
        assert service._global_vocab.unregister_codebase.call_count == 1
        service._storage.create_collection.assert_awaited_once()

    async def test_successful_force_reindex_does_not_roll_back(self, monkeypatch):
        service = self._ready_service(monkeypatch)
        service._storage.collection_exists = AsyncMock(return_value=True)
        service._storage.delete_collection = AsyncMock()
        service._storage.create_collection = AsyncMock()
        service._storage.store_metadata = AsyncMock()
        monkeypatch.setattr(idx_svc, "discover_files", lambda path: [object()])
        service._prepare_files = MagicMock(return_value=([_prepared("one.py")], [{"tok"}]))
        service._process_batch = AsyncMock(return_value=(1, 0))

        _files, chunks, _stats = await service.index("/proj", force=True)

        assert service._storage.delete_collection.await_count == 1
        assert service._global_vocab.unregister_codebase.call_count == 1
        assert chunks == 1


def _prepared(rel_path: str):
    """A stand-in for a PreparedFile with no chunks (one document: its summary)."""
    return SimpleNamespace(file_info=SimpleNamespace(rel_path=rel_path), chunks=[])


class TestIncrementalIndexTransactions:
    """Incremental work embeds first and commits complete batches one at a time."""

    async def test_embedding_failure_leaves_the_existing_points_alone(self):
        """Failing before the swap must not delete anything.

        Embedding is the slow, network-dependent step. When it fails the
        previous points are all still in place and no vocabulary delta has been
        attempted. The journaled transaction lives inside _process_batch.
        """
        service = _make_service()
        service._collect_removed_tokens = AsyncMock(return_value={"mod.py": [{"old"}]})
        service._prepare_files = MagicMock(return_value=([_prepared("mod.py")], [{"new"}]))
        service._process_batch = AsyncMock(side_effect=RuntimeError("transient embed failure"))
        service._storage.delete_by_paths_batch = AsyncMock()
        changes = SimpleNamespace(
            added=[], modified=[SimpleNamespace(rel_path="mod.py")], deleted=[]
        )

        with pytest.raises(RuntimeError, match="transient embed failure"):
            await service._incremental_index("codesearch_x", changes, "/proj")

        # Nothing was deleted: the file still has the points it started with.
        service._storage.delete_by_paths_batch.assert_not_awaited()

        service._global_vocab.update_codebase_incremental.assert_not_called()

    async def test_a_failed_batch_does_not_start_the_next_one(self, monkeypatch):
        """Files in later batches keep their points, so the index stays whole."""
        monkeypatch.setattr(idx_svc, "INDEXING_BATCH_SIZE", 1)
        service = _make_service()
        service._collect_removed_tokens = AsyncMock(
            return_value={"a.py": [{"old_a"}], "b.py": [{"old_b"}]}
        )
        service._prepare_files = MagicMock(
            return_value=([_prepared("a.py"), _prepared("b.py")], [{"new_a"}, {"new_b"}])
        )
        service._process_batch = AsyncMock(side_effect=RuntimeError("embed failure"))
        service._storage.delete_by_paths_batch = AsyncMock()
        changes = SimpleNamespace(
            added=[],
            modified=[SimpleNamespace(rel_path="a.py"), SimpleNamespace(rel_path="b.py")],
            deleted=[],
        )

        with pytest.raises(RuntimeError, match="embed failure"):
            await service._incremental_index("codesearch_x", changes, "/proj")

        # Only the first batch ran; b.py was never touched at all.
        assert service._process_batch.await_count == 1
        assert service._process_batch.await_args.args[0][0].file_info.rel_path == "a.py"
        service._global_vocab.update_codebase_incremental.assert_not_called()

    async def test_deleted_files_are_removed_without_waiting_for_a_batch(self):
        """A file gone from disk has no replacement to wait for."""
        service = _make_service()
        service._collect_removed_tokens = AsyncMock(return_value={"gone.py": [{"old"}]})
        service._prepare_files = MagicMock(return_value=([_prepared("new.py")], [{"new"}]))
        service._process_batch = AsyncMock(return_value=(1, 0))
        service._storage.store_metadata = AsyncMock()
        service._storage.delete_by_paths_batch = AsyncMock()
        changes = SimpleNamespace(
            added=[SimpleNamespace(rel_path="new.py")], modified=[], deleted=["gone.py"]
        )

        await service._incremental_index("codesearch_x", changes, "/proj")

        service._storage.delete_by_paths_batch.assert_awaited_once()
        assert service._storage.delete_by_paths_batch.await_args.args[1] == ["gone.py"]

    async def test_successful_incremental_does_not_roll_back(self, monkeypatch):
        service = _make_service()
        service._collect_removed_tokens = AsyncMock(return_value={})
        service._prepare_files = MagicMock(return_value=([_prepared("new.py")], [{"tok"}]))
        service._process_batch = AsyncMock(return_value=(1, 0))
        service._storage.store_metadata = AsyncMock()
        service._storage.delete_by_paths_batch = AsyncMock()
        changes = SimpleNamespace(
            added=[SimpleNamespace(rel_path="new.py")], modified=[], deleted=[]
        )

        await service._incremental_index("codesearch_x", changes, "/proj")

        # The mocked batch owns the entire transaction, so no outer rollback or
        # vocabulary mutation occurs.
        service._global_vocab.update_codebase_incremental.assert_not_called()
        service._storage.delete_by_paths_batch.assert_not_awaited()


class TestCollectRemovedTokensFetchFailure:
    """Outgoing sparse vectors must be readable before any mutation begins."""

    async def test_aborts_and_does_not_delete_on_fetch_failure(self):
        service = _make_service()
        service._storage.get_stored_sparse_indices_by_paths = AsyncMock(
            side_effect=RuntimeError("scroll timeout")
        )
        service._storage.delete_by_paths_batch = AsyncMock()
        changes = SimpleNamespace(deleted=["foo.py"], modified=[], added=[])

        with pytest.raises(RuntimeError, match="scroll timeout"):
            await service._collect_removed_tokens("col", changes)

        # Nothing was deleted: a clean abort leaves Qdrant and the vocab in sync.
        service._storage.delete_by_paths_batch.assert_not_awaited()

    async def test_returns_tokens_per_path_and_deletes_nothing(self):
        """Collecting is a read. Removal is deferred to the swap.

        Keyed by path because the vocabulary delta is committed per batch, and
        a batch may only account for the files it actually replaced.
        """
        service = _make_service()
        service._storage.get_stored_sparse_indices_by_paths = AsyncMock(
            return_value={"foo.py": [[1, 2]]}
        )
        service._global_vocab.get_tokens_by_indices = MagicMock(return_value={1: "some", 2: "code"})
        service._storage.delete_by_paths_batch = AsyncMock()
        changes = SimpleNamespace(deleted=["foo.py"], modified=[], added=[])

        tokens = await service._collect_removed_tokens("col", changes)

        service._storage.delete_by_paths_batch.assert_not_awaited()
        assert tokens == {"foo.py": [{"some", "code"}]}


class TestVocabularyAccountingInvariant:
    """Adding and removing the same file must be an exact vocabulary inverse."""

    async def test_register_then_remove_with_imports_restores_counters(self, tmp_path):
        vocab = GlobalVocabulary(db_path=tmp_path / "vocabulary.db")
        storage = QdrantStorage()
        journal = MagicMock()
        journal.get.return_value = None
        journal.list.return_value = []
        service = IndexingService(
            storage=storage,
            embedder=MagicMock(),
            global_vocab=vocab,
            journal=journal,
        )
        file_info = _make_file(
            "imported.py",
            "import drift_only_dependency\n\ndef calculate_result(value):\n    return value + 1\n",
        )

        try:
            _prepared_files, added_tokens = service._prepare_files([file_info])

            baseline_tokens = set().union(*added_tokens)
            vocab.register_codebase("baseline", [baseline_tokens])
            starting_doc_freq = vocab._get_doc_freq().copy()
            starting_doc_count = vocab.total_docs

            vocab.register_codebase("subject", added_tokens)
            token_to_index = vocab._get_vocab()
            storage.get_stored_sparse_indices_by_paths = AsyncMock(
                return_value={
                    file_info.rel_path: [
                        sorted(token_to_index[token] for token in token_set)
                        for token_set in added_tokens
                    ]
                }
            )
            changes = SimpleNamespace(deleted=[file_info.rel_path], modified=[], added=[])
            removed_by_path = await service._collect_removed_tokens("collection", changes)
            removed_tokens = [t for toks in removed_by_path.values() for t in toks]
            vocab.update_codebase_incremental(
                "subject",
                added_tokens=[],
                removed_tokens=removed_tokens,
                net_doc_change=-len(removed_tokens),
            )

            assert vocab._get_doc_freq() == starting_doc_freq
            assert vocab.total_docs == starting_doc_count
            assert vocab.get_codebase_doc_count("subject") == 0
        finally:
            vocab.close()


class TestStoredVocabularyText:
    """Stored chunk payloads preserve the text needed for exact removal."""

    def test_vocabulary_text_uses_stored_content_limit_but_dense_text_does_not(self):
        content = "x" * settings.max_payload_content_chars + " tail_only_token"
        chunk = Chunk(
            content=content,
            chunk_type="block",
            name=None,
            start_line=1,
            end_line=1,
            context=None,
            imports=["example.module"],
        )

        stored_content = truncate_chunk_content(chunk.content)
        vocabulary_text = build_chunk_vocabulary_text(stored_content, chunk.imports)

        assert vocabulary_text == "Uses: example.module\n\n" + (
            "x" * settings.max_payload_content_chars
        )
        assert "tail_only_token" not in vocabulary_text
        assert build_chunk_vocabulary_text(chunk.content, chunk.imports).endswith("tail_only_token")
        assert IndexingService._chunk_embedding_text(chunk).endswith("tail_only_token")

    async def test_get_stored_content_paginates_past_1000_points(self):
        storage = QdrantStorage()
        client = MagicMock()
        first_page = [
            SimpleNamespace(payload={"type": "chunk", "content": f"chunk {i}", "imports": []})
            for i in range(1000)
        ]
        last_page = [
            SimpleNamespace(payload={"type": "chunk", "content": "chunk 1000", "imports": []})
        ]
        client.scroll = AsyncMock(side_effect=[(first_page, "next-page"), (last_page, None)])
        storage._get_client = AsyncMock(return_value=client)

        texts = await storage.get_stored_content_for_path("collection", "large.py")

        assert len(texts) == 1001
        assert texts[-1] == "chunk 1000"
        assert client.scroll.await_count == 2
        assert client.scroll.await_args_list[1].kwargs["offset"] == "next-page"

    async def test_stored_content_is_not_retruncated_during_removal(self):
        storage = QdrantStorage()
        client = MagicMock()
        stored_content = "x" * (settings.max_payload_content_chars + 1) + " retained_tail"
        client.scroll = AsyncMock(
            return_value=(
                [
                    SimpleNamespace(
                        payload={
                            "type": "chunk",
                            "content": stored_content,
                            "imports": ["example.module"],
                        }
                    )
                ],
                None,
            )
        )
        storage._get_client = AsyncMock(return_value=client)

        texts = await storage.get_stored_content_for_path("collection", "older.py")

        assert texts == ["Uses: example.module\n\n" + stored_content]

    async def test_legacy_chunk_without_imports_degrades_to_raw_content(self, caplog):
        storage = QdrantStorage()
        client = MagicMock()
        client.scroll = AsyncMock(
            return_value=(
                [SimpleNamespace(payload={"type": "chunk", "content": "legacy body"})],
                None,
            )
        )
        storage._get_client = AsyncMock(return_value=client)

        with caplog.at_level("WARNING", logger="mcp_codesearch.storage.qdrant"):
            texts = await storage.get_stored_content_for_path("collection", "legacy.py")

        assert texts == ["legacy body"]
        assert "legacy chunk point" in caplog.text
        assert "may omit import or truncated-content tokens" in caplog.text
