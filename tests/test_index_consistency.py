"""Crash-recovery and global consistency tests for indexing."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.models import SparseVector as QdrantSparseVector
from vector_core import GlobalVocabulary

from mcp_codesearch.indexer.journal import IndexIntentStore
from mcp_codesearch.services import indexing_service as indexing_module
from mcp_codesearch.services.indexing_service import (
    IndexingService,
    _canonical_qdrant_identity,
)
from mcp_codesearch.services.search_service import SearchQuery, SearchService
from mcp_codesearch.storage import qdrant as qdrant_module
from mcp_codesearch.storage.qdrant import QdrantStorage


def _service(tmp_path: Path) -> tuple[IndexingService, IndexIntentStore]:
    storage = MagicMock()
    embedder = MagicMock()
    vocab = MagicMock()
    vocab.get_tokens_by_indices.side_effect = lambda indices: {
        index: {1: "alpha", 2: "beta"}[index] for index in indices
    }
    journal = IndexIntentStore(tmp_path / "journal.db")
    return IndexingService(storage, embedder, vocab, journal=journal), journal


def _install_noop_lock(monkeypatch, calls: list[tuple[str, bool]] | None = None) -> None:
    @asynccontextmanager
    async def noop_lock(name, timeout=60.0, lock_dir=None, shared=False):
        _ = timeout, lock_dir
        if calls is not None:
            calls.append((name, shared))
        yield

    monkeypatch.setattr(indexing_module, "async_file_lock", noop_lock)


def test_sparse_recovery_reads_real_qdrant_vectors_and_rejects_missing() -> None:
    point = SimpleNamespace(
        id="point",
        vector={"sparse": QdrantSparseVector(indices=[1, 7], values=[1.0, 2.0])},
    )
    assert QdrantStorage._sparse_indices(point, "collection") == [1, 7]

    with pytest.raises(RuntimeError, match="no named sparse vector"):
        QdrantStorage._sparse_indices(
            SimpleNamespace(id="missing", vector={}),
            "collection",
        )


def test_common_loopback_qdrant_aliases_share_consistency_identity() -> None:
    expected = "http://loopback:6333"
    assert _canonical_qdrant_identity("http://localhost:6333/") == expected
    assert _canonical_qdrant_identity("HTTP://127.0.0.1:6333") == expected
    assert _canonical_qdrant_identity("http://[::1]:6333") == expected


def test_consistency_wait_exceeds_longest_qdrant_write_budget(tmp_path) -> None:
    service, _journal = _service(tmp_path)
    assert service._consistency_timeout > indexing_module.settings.upsert_batch_timeout


async def test_embedding_failure_precedes_every_consistency_mutation(tmp_path, monkeypatch) -> None:
    service, journal = _service(tmp_path)
    _install_noop_lock(monkeypatch)
    service._embedder.embed_all = AsyncMock(side_effect=RuntimeError("embedding failed"))
    service._storage.delete_by_paths_batch = AsyncMock()
    service._storage.upsert_batch = AsyncMock()
    prepared = SimpleNamespace(
        summary="summary",
        chunk_embedding_texts=[],
        file_info=SimpleNamespace(rel_path="module.py"),
        chunks=[],
    )

    with pytest.raises(RuntimeError, match="embedding failed"):
        await service._process_batch(
            [prepared],
            "codesearch_test",
            "/repo",
            {},
            added_tokens=[{"alpha"}],
            removed_tokens=[],
            net_doc_change=1,
        )

    assert journal.list() == []
    service._global_vocab.update_codebase_incremental.assert_not_called()
    service._storage.delete_by_paths_batch.assert_not_awaited()
    service._storage.upsert_batch.assert_not_awaited()


async def test_failed_qdrant_batch_is_cleared_and_reconciled_before_unlock(
    tmp_path, monkeypatch
) -> None:
    service, journal = _service(tmp_path)
    _install_noop_lock(monkeypatch)
    service._storage.collection_exists = AsyncMock(return_value=True)
    service._storage.delete_by_paths_batch = AsyncMock()

    async def stored_indices(collection, batch_size=1000):
        _ = collection, batch_size
        yield [[1], [1, 2]]

    service._storage.iter_stored_sparse_indices = stored_indices

    with pytest.raises(RuntimeError, match="write failed"):
        async with service._write_intent("codesearch_test", "/repo", "index"):
            await service._set_pending_paths("codesearch_test", ["module.py"])
            raise RuntimeError("write failed")

    service._storage.delete_by_paths_batch.assert_awaited_once_with(
        "codesearch_test", ["module.py"]
    )
    service._global_vocab.register_codebase_frequencies.assert_called_once_with(
        "codesearch_test",
        {"alpha": 2, "beta": 1},
        2,
    )
    assert journal.get("codesearch_test") is None


async def test_delete_failure_retries_from_durable_intent(tmp_path, monkeypatch) -> None:
    service, journal = _service(tmp_path)
    _install_noop_lock(monkeypatch)
    service._storage.collection_exists = AsyncMock(return_value=True)
    service._storage.delete_collection = AsyncMock(
        side_effect=[RuntimeError("response lost"), None]
    )
    service._global_vocab.unregister_codebase = MagicMock()

    with pytest.raises(RuntimeError, match="response lost"):
        async with service._write_intent("codesearch_test", "/repo", "delete"):
            assert await service._safe_unregister_vocab("codesearch_test")
            await service._storage.delete_collection("codesearch_test")

    assert service._global_vocab.unregister_codebase.call_count == 2
    assert service._storage.delete_collection.await_count == 2
    assert journal.get("codesearch_test") is None


async def test_reader_recovers_abandoned_intent_before_shared_snapshot(
    tmp_path, monkeypatch
) -> None:
    service, journal = _service(tmp_path)
    lock_calls: list[tuple[str, bool]] = []
    _install_noop_lock(monkeypatch, lock_calls)
    journal.mark("codesearch_test", "/repo", "index")
    service._storage.collection_exists = AsyncMock(return_value=True)

    async def stored_indices(collection, batch_size=1000):
        _ = collection, batch_size
        yield [[1]]

    service._storage.iter_stored_sparse_indices = stored_indices

    async with service.consistent_read():
        assert journal.list() == []

    assert lock_calls == [
        (service._admission_lock_name, False),
        (service._consistency_lock_name, True),
        ("codesearch_test_admission", False),
        ("codesearch_test", False),
        (service._admission_lock_name, False),
        (service._consistency_lock_name, False),
        (service._admission_lock_name, False),
        (service._consistency_lock_name, True),
    ]
    service._global_vocab.register_codebase_frequencies.assert_called_once_with(
        "codesearch_test",
        {"alpha": 1},
        1,
    )


async def test_reader_refreshes_vocabulary_only_when_generation_changes(
    tmp_path, monkeypatch
) -> None:
    service, journal = _service(tmp_path)
    _install_noop_lock(monkeypatch)
    service._global_vocab.invalidate_cache = MagicMock()

    async with service.consistent_read():
        pass
    async with service.consistent_read():
        pass
    service._global_vocab.invalidate_cache.assert_called_once()

    journal.bump_generation()
    async with service.consistent_read():
        pass
    assert service._global_vocab.invalidate_cache.call_count == 2


async def test_global_lock_is_released_before_remote_read_body(tmp_path, monkeypatch) -> None:
    service, _journal = _service(tmp_path)
    events: list[tuple[str, str, bool]] = []

    @asynccontextmanager
    async def recording_lock(name, timeout=60.0, lock_dir=None, shared=False):
        _ = timeout, lock_dir
        events.append(("enter", name, shared))
        try:
            yield
        finally:
            events.append(("exit", name, shared))

    monkeypatch.setattr(indexing_module, "async_file_lock", recording_lock)

    async with service.consistent_read(
        ["codesearch_test"],
        prepare=lambda: "sparse-vector",
    ) as snapshot:
        assert snapshot.prepared == "sparse-vector"
        assert events == [
            ("enter", "codesearch_test_admission", False),
            ("enter", "codesearch_test", True),
            ("exit", "codesearch_test_admission", False),
            ("enter", service._admission_lock_name, False),
            ("enter", service._consistency_lock_name, True),
            ("exit", service._admission_lock_name, False),
            ("exit", service._consistency_lock_name, True),
        ]

    assert events[-1] == ("exit", "codesearch_test", True)


async def test_recovery_ignores_live_intent_that_clears_while_waiting(
    tmp_path, monkeypatch
) -> None:
    service, journal = _service(tmp_path)
    journal.mark("codesearch_live", "/repo", "index")

    @asynccontextmanager
    async def disappearing_lock(name, **_kwargs):
        if name == "codesearch_live":
            journal.clear(name)
            raise TimeoutError("live writer still owns collection")
        yield

    monkeypatch.setattr(indexing_module, "async_file_lock", disappearing_lock)

    assert await service.recover_pending_intents() == 0
    assert journal.get("codesearch_live") is None


async def test_full_repair_rebuilds_live_stale_and_aggregate_state(tmp_path, monkeypatch) -> None:
    _install_noop_lock(monkeypatch)
    vocab = GlobalVocabulary(tmp_path / "vocabulary.db")
    vocab.register_codebase("codesearch_live", [{"right", "wrong"}, {"wrong"}])
    vocab.register_codebase("codesearch_stale", [{"stale"}])
    token_to_index = vocab._get_vocab()
    indices_before = token_to_index.copy()
    conn = vocab._get_conn()
    conn.execute("UPDATE vocabulary SET doc_freq = doc_freq + 50")
    conn.commit()

    storage = MagicMock()
    storage.list_collections = AsyncMock(return_value=["codesearch_live"])
    storage.collection_exists = AsyncMock(side_effect=lambda name: name == "codesearch_live")
    storage.count_index_documents = AsyncMock(return_value=2)
    storage.delete_collection = AsyncMock()
    storage.delete_by_paths_batch = AsyncMock()

    async def stored_indices(collection, batch_size=1000):
        _ = collection, batch_size
        yield [[token_to_index["right"]], [token_to_index["right"]]]

    storage.iter_stored_sparse_indices = stored_indices
    journal = IndexIntentStore(tmp_path / "journal.db")
    service = IndexingService(storage, MagicMock(), vocab, journal=journal)

    try:
        stats = await service.repair_vocabulary(repair=True, full=True)

        assert stats.collections_repaired == 1
        assert stats.registrations_removed == 1
        assert stats.aggregate_frequencies_rebuilt is True
        assert vocab.get_codebase_doc_count("codesearch_live") == 2
        assert vocab.get_codebase_doc_count("codesearch_stale") == 0
        assert vocab.total_docs == 2
        frequencies = vocab._get_doc_freq()
        assert frequencies["right"] == 2
        assert frequencies["wrong"] == 0
        assert frequencies["stale"] == 0
        assert vocab._get_vocab() == indices_before
        storage.delete_collection.assert_not_awaited()
        storage.delete_by_paths_batch.assert_not_awaited()
    finally:
        vocab.close()


async def test_search_cache_is_scoped_to_shared_index_generation(tmp_path) -> None:
    class Coordinator:
        generation = 1
        inside = False

        @asynccontextmanager
        async def consistent_read(self, collections, prepare=None):
            assert collections
            self.inside = True
            try:
                yield SimpleNamespace(
                    generation=self.generation,
                    prepared=prepare() if prepare else None,
                )
            finally:
                self.inside = False

    coordinator = Coordinator()
    embedder = MagicMock()

    async def embed_outside_snapshot(_query):
        assert coordinator.inside is False
        return [0.1]

    embedder.embed_single_cached = AsyncMock(side_effect=embed_outside_snapshot)
    service = SearchService(
        MagicMock(),
        embedder,
        MagicMock(),
        indexing_service=coordinator,  # type: ignore[arg-type]
    )
    query = SearchQuery(query="cache generation", path=str(tmp_path))

    with patch(
        "mcp_codesearch.services.search_service.search_codebase",
        new=AsyncMock(side_effect=[[], []]),
    ) as search:
        await service.search(query)
        await service.search(query)
        assert search.await_count == 1

        coordinator.generation = 2
        await service.search(query)
        assert search.await_count == 2


async def test_explicit_delete_can_remove_an_unrecoverable_target_intent(
    tmp_path, monkeypatch
) -> None:
    service, journal = _service(tmp_path)
    _install_noop_lock(monkeypatch)
    collection = indexing_module.collection_name(str(Path("/repo").resolve()))
    journal.mark(collection, "/repo", "index")
    journal.set_pending_paths(collection, ["broken.py"])
    service._storage.collection_exists = AsyncMock(return_value=True)
    service._storage.delete_collection = AsyncMock()
    service._global_vocab.get_codebase_ids = MagicMock(return_value=[collection])
    service._global_vocab.unregister_codebase = MagicMock()

    assert await service.delete("/repo") is True

    service._storage.delete_collection.assert_awaited_once_with(collection)
    service._storage.delete_by_paths_batch.assert_not_called()
    assert journal.get(collection) is None


async def test_force_reindex_replaces_stale_delete_intent_for_absent_collection(
    tmp_path, monkeypatch
) -> None:
    service, journal = _service(tmp_path)
    _install_noop_lock(monkeypatch)
    collection = indexing_module.collection_name(str(Path("/repo").resolve()))
    journal.mark(collection, "/repo", "delete")
    service._storage.collection_exists = AsyncMock(return_value=False)
    service._storage.create_collection = AsyncMock()
    service._storage.store_metadata = AsyncMock()
    service._global_vocab.get_codebase_ids = MagicMock(return_value=[])
    monkeypatch.setattr(indexing_module, "discover_files", lambda path: [])

    await service.index("/repo", force=True)

    assert journal.get(collection) is None
    service._storage.create_collection.assert_awaited_once_with(collection)


async def test_metadata_cancellation_waits_for_remote_completion(monkeypatch) -> None:
    storage = QdrantStorage()
    core = MagicMock()
    core.get_metadata = AsyncMock(return_value={})
    started = asyncio.Event()
    release = asyncio.Event()
    completed = False

    async def store_metadata(_collection, _metadata):
        nonlocal completed
        started.set()
        await release.wait()
        completed = True

    core.store_metadata = AsyncMock(side_effect=store_metadata)
    storage._core = core
    monkeypatch.setattr(
        qdrant_module,
        "settings",
        SimpleNamespace(
            embedding_model="model",
            embedding_cache_namespace=None,
        ),
    )
    task = asyncio.create_task(storage.store_metadata("collection", "/repo"))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed is True


async def test_delete_forgets_text_index_state_before_remote_completion() -> None:
    storage = QdrantStorage()
    storage._text_indexed_collections.add("collection")
    storage._text_index_failed_collections.add("collection")
    started = asyncio.Event()
    release = asyncio.Event()

    async def delete_collection(_name):
        started.set()
        await release.wait()

    storage._core = MagicMock()
    storage._core.delete_collection = AsyncMock(side_effect=delete_collection)
    task = asyncio.create_task(storage.delete_collection("collection"))
    await started.wait()

    assert "collection" not in storage._text_indexed_collections
    assert "collection" not in storage._text_index_failed_collections
    release.set()
    await task
