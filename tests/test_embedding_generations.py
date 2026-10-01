"""Offline regressions for independently retained embedding generations."""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient
from vector_core import AsyncSingleton, GlobalVocabulary
from vector_core.embeddings.identity import EmbeddingIdentity
from vector_core.settings import settings as core_settings

from mcp_codesearch import singletons
from mcp_codesearch.indexer.journal import IndexIntentStore
from mcp_codesearch.services import indexing_service as indexing_module
from mcp_codesearch.services.indexing_service import IndexingService
from mcp_codesearch.services.search_service import SearchQuery, SearchService
from mcp_codesearch.storage.qdrant import (
    EmbeddingDeploymentMismatchError,
    EmbeddingModelMismatchError,
    QdrantStorage,
)


class FakeEmbedder:
    """Deterministic vectors with visibly different same-width model spaces."""

    def __init__(self, identity: EmbeddingIdentity):
        self.identity = identity

    async def embed_all(self, texts: list[str], *, role: str) -> list[list[float]]:
        assert role == "document"
        return [
            [
                (hashlib.sha256(f"{self.identity.model}:{text}:{i}".encode()).digest()[0] + 1) / 256
                for i in range(self.identity.dimension)
            ]
            for text in texts
        ]


@pytest.fixture
def identity():
    return EmbeddingIdentity(
        model="model-a", namespace="revision-1", endpoint="http://offline.invalid", dimension=8
    )


@pytest.fixture
async def generations(tmp_path, monkeypatch):
    """Share only the in-memory server; give each identity its own durable state."""
    client = AsyncQdrantClient(":memory:")
    vocabularies = {}

    @asynccontextmanager
    async def isolated_lock(*_args, **_kwargs):
        yield

    monkeypatch.setattr(indexing_module, "async_file_lock", isolated_lock)

    def make(identity):
        storage = QdrantStorage(url="http://offline.invalid", identity=identity)
        storage._core._get_client = AsyncMock(return_value=client)
        storage._core.get_client = AsyncMock(return_value=client)
        key = identity.fingerprint
        if key not in vocabularies:
            vocabularies[key] = GlobalVocabulary(tmp_path / f"vocab_{key}.db")
        vocab = vocabularies[key]
        journal = IndexIntentStore(tmp_path / "intents.db", namespace=key)
        service = IndexingService(storage, FakeEmbedder(identity), vocab, journal=journal)
        service._stale_locks_cleaned = True
        return service

    yield make, client
    for vocab in vocabularies.values():
        vocab.close()
    await client.close()


def write_module(root: Path, name: str, value: str) -> None:
    (root / name).write_text(f'def {value}():\n    """Return {value}."""\n    return "{value}"\n')


@pytest.mark.parametrize("corruption", ["missing", "partial", "preprocessing"])
async def test_scoped_name_does_not_bypass_full_provenance(
    generations, identity, tmp_path, corruption
):
    make, _client = generations
    service = make(identity)
    root = tmp_path / "source"
    root.mkdir()
    write_module(root, "one.py", "alpha")
    await service.index(str(root))
    collection = service._storage.collection_name(str(root))
    metadata = await service._storage.get_metadata(collection)
    if corruption == "missing":
        metadata.pop("embedding_identity")
    elif corruption == "partial":
        metadata["embedding_identity"].pop("preprocessing")
    else:
        metadata["embedding_identity"]["preprocessing"] = "different-format"
    await service._storage._core.store_metadata(collection, metadata)
    metadata = await service._storage.get_metadata(collection)
    with pytest.raises(EmbeddingDeploymentMismatchError):
        await service.index(str(root))
    assert await service._storage.get_metadata(collection) == metadata


async def snapshot(client, collection):
    points, offset = await client.scroll(collection, limit=1000, with_vectors=True)
    assert offset is None
    return {str(point.id): (point.payload, point.vector) for point in points}


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "model-b"},
        {"namespace": "revision-2"},
        {"dimension": 16},
        {"endpoint": "http://another.invalid"},
        {"preprocessing": "another-version"},
        {"max_text_chars": 4000},
    ],
)
def test_collection_names_cover_full_identity_and_freeze_configuration(
    identity, changes, monkeypatch
):
    storage = QdrantStorage(identity=identity)
    other = QdrantStorage(identity=replace(identity, **changes))
    original = storage.collection_name("/repo/")
    assert original == f"csg_{identity.fingerprint}_{hashlib.sha256(b'/repo').hexdigest()[:12]}"
    assert original != other.collection_name("/repo")
    assert storage.owns_collection(original)
    assert not other.owns_collection(original)
    assert not storage.owns_collection(f"codesearch_{identity.fingerprint}")
    monkeypatch.setattr(core_settings, "embedding_model", "mutated-global-model")
    monkeypatch.setattr(core_settings, "embedding_dim", 32)
    monkeypatch.setattr(core_settings, "embedding_cache_namespace", "mutated-global-revision")
    assert storage.collection_name("/repo") == original
    assert storage.identity == identity


async def test_same_width_migration_reindexes_without_modifying_old_points(
    tmp_path, identity, generations
):
    make, client = generations
    root = tmp_path / "repo"
    root.mkdir()
    write_module(root, "module.py", "alpha")
    a = make(identity)
    b = make(replace(identity, model="model-b"))
    await a.index(str(root))
    name_a = a._storage.collection_name(str(root))
    before = await snapshot(client, name_a)
    files, chunks, stats = await b.index(str(root))
    name_b = b._storage.collection_name(str(root))
    assert files == 1 and chunks > 0 and not stats.was_incremental
    assert await snapshot(client, name_a) == before
    assert await b._storage.get_dense_dim(name_b) == identity.dimension
    after = await snapshot(client, name_b)
    assert set(after) == set(before)
    assert any(after[key][1] != before[key][1] for key in before)
    assert await a._storage.list_collections() == [name_a]
    assert await b._storage.list_collections() == [name_b]
    assert await b._storage.list_preserved_collections() == [name_a]
    assert a._global_vocab.total_docs == b._global_vocab.total_docs


def test_shared_collection_override_cannot_collapse_codesearch_generations(identity, monkeypatch):
    monkeypatch.setattr(core_settings, "collection_name", "shared-notes-and-docs")
    a = QdrantStorage(identity=identity)
    b = QdrantStorage(identity=replace(identity, model="model-b"))
    names = {a.collection_name("/repo"), a.collection_name("/other"), b.collection_name("/repo")}
    assert len(names) == 3
    assert all(name.startswith("csg_") for name in names)


async def test_return_to_retained_generation_reconciles_edits_and_deletions(
    tmp_path, identity, generations
):
    make, client = generations
    root = tmp_path / "repo"
    root.mkdir()
    write_module(root, "keep.py", "alpha")
    write_module(root, "delete.py", "obsolete")
    a = make(identity)
    b = make(replace(identity, model="model-b"))
    await a.index(str(root))
    await b.index(str(root))
    name_b = b._storage.collection_name(str(root))
    before_b = await snapshot(client, name_b)
    write_module(root, "keep.py", "updated_longer")
    (root / "delete.py").unlink()
    write_module(root, "added.py", "brand_new")
    returned = make(identity)
    files, _chunks, stats = await returned.index(str(root))
    assert files == 2 and stats.was_incremental
    name_a = returned._storage.collection_name(str(root))
    assert set(await returned._storage.get_indexed_files(name_a)) == {"keep.py", "added.py"}
    stored = await snapshot(client, name_a)
    contents = " ".join(payload.get("content", "") for payload, _vector in stored.values())
    assert "updated_longer" in contents and "brand_new" in contents
    assert "obsolete" not in contents
    assert await snapshot(client, name_b) == before_b
    assert returned._global_vocab.total_docs == await returned._storage.count_index_documents(
        name_a
    )
    assert returned._journal.list() == []


@pytest.mark.parametrize("cancel", [False, True])
async def test_partial_failed_or_cancelled_generation_recovers_without_touching_old(
    tmp_path, identity, generations, monkeypatch, cancel
):
    make, client = generations
    root = tmp_path / "repo"
    root.mkdir()
    write_module(root, "module.py", "alpha")
    a = make(identity)
    b = make(replace(identity, model="model-b"))
    await a.index(str(root))
    name_a = a._storage.collection_name(str(root))
    before = await snapshot(client, name_a)
    vocab_before = a._global_vocab._get_doc_freq().copy()
    original = b._storage.upsert_batch

    async def partially_written(*args, **kwargs):
        await original(*args, **kwargs)
        if cancel:
            raise asyncio.CancelledError
        raise RuntimeError("injected failure after accepted write")

    monkeypatch.setattr(b._storage, "upsert_batch", partially_written)
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await b.index(str(root))
    name_b = b._storage.collection_name(str(root))
    assert await snapshot(client, name_a) == before
    assert a._global_vocab._get_doc_freq() == vocab_before
    assert b._journal.list() == []
    assert b._global_vocab.total_docs == await b._storage.count_index_documents(name_b) == 0
    monkeypatch.setattr(b._storage, "upsert_batch", original)
    await b.index(str(root))
    assert await b._storage.count_index_documents(name_b) > 0
    assert await snapshot(client, name_a) == before


async def test_repair_counts_only_current_generation(tmp_path, identity, generations):
    make, client = generations
    root = tmp_path / "repo"
    root.mkdir()
    write_module(root, "module.py", "alpha")
    a = make(identity)
    b = make(replace(identity, model="model-b"))
    await a.index(str(root))
    await b.index(str(root))
    name_a = a._storage.collection_name(str(root))
    name_b = b._storage.collection_name(str(root))
    before = await snapshot(client, name_a)
    a_frequencies = a._global_vocab._get_doc_freq().copy()
    b._global_vocab.unregister_codebase(name_b)
    stats = await b.repair_vocabulary(repair=True, full=True)
    assert stats.collections_checked == stats.collections_repaired == 1
    assert not stats.failures
    assert b._global_vocab.get_codebase_ids() == [name_b]
    assert b._global_vocab.total_docs == await b._storage.count_index_documents(name_b)
    assert await snapshot(client, name_a) == before
    assert a._global_vocab._get_doc_freq() == a_frequencies


async def test_embedding_failure_leaves_retained_points_and_vocabulary_intact(
    tmp_path, identity, generations, monkeypatch
):
    make, client = generations
    root = tmp_path / "repo"
    root.mkdir()
    write_module(root, "module.py", "alpha")
    a = make(identity)
    b = make(replace(identity, model="model-b"))
    await a.index(str(root))
    name_a = a._storage.collection_name(str(root))
    before = await snapshot(client, name_a)
    count_before = a._global_vocab.total_docs
    monkeypatch.setattr(b._embedder, "embed_all", AsyncMock(side_effect=RuntimeError("inference")))
    with pytest.raises(RuntimeError, match="inference"):
        await b.index(str(root))
    assert await snapshot(client, name_a) == before
    assert a._global_vocab.total_docs == count_before
    assert b._global_vocab.total_docs == 0
    assert b._journal.list() == []
    assert await b._storage.count_index_documents(b._storage.collection_name(str(root))) == 0


async def test_legacy_unknown_metadata_is_preserved_and_never_stamped(
    tmp_path, identity, generations
):
    make, client = generations
    root = tmp_path / "repo"
    root.mkdir()
    write_module(root, "module.py", "alpha")
    service = make(identity)
    legacy = "codesearch_legacy"
    await service._storage.create_collection(legacy)
    await service._storage._core.store_metadata(legacy, {"codebase_path": str(root)})
    before = await snapshot(client, legacy)
    with pytest.raises(EmbeddingModelMismatchError):
        await service._verify_embedding_model(legacy, str(root))
    await service.index(str(root))
    await service.repair_vocabulary(repair=True, full=True)
    assert await snapshot(client, legacy) == before
    assert "embedding_model" not in await service._storage.get_metadata(legacy)
    assert await service._storage.list_preserved_collections() == [legacy]
    assert legacy not in service._global_vocab.get_codebase_ids()


def test_search_result_cache_keys_are_embedding_identity_specific(identity):
    a = SearchService(QdrantStorage(identity=identity), None, None)
    b = SearchService(QdrantStorage(identity=replace(identity, model="model-b")), None, None)
    query = SearchQuery(query="alpha", path="/repo")
    key_a = a._cache_key(query, "/repo", generation=7)
    key_b = b._cache_key(query, "/repo", generation=7)
    assert key_a != key_b
    assert key_a != a._cache_key(query, "/repo", generation=8)
    b._cache = a._cache
    a._cache.set(key_a, "old generation result")
    assert b._cache.get(key_b) is None


def test_explicit_consistency_namespace_keeps_generation_journals_separate(
    tmp_path, identity, monkeypatch
):
    monkeypatch.setattr(indexing_module.settings, "consistency_namespace", "shared-service")
    monkeypatch.setattr(core_settings, "cache_dir", tmp_path)
    vocab = GlobalVocabulary(tmp_path / "shared.db")
    try:
        other = replace(identity, model="model-b")
        a = IndexingService(QdrantStorage(identity=identity), FakeEmbedder(identity), vocab)
        b = IndexingService(QdrantStorage(identity=other), FakeEmbedder(other), vocab)
        assert a._consistency_scope != b._consistency_scope
        assert a._journal.namespace != b._journal.namespace
        assert a._journal.db_path == b._journal.db_path
        name_a = a._storage.collection_name("/repo")
        a._journal.mark(name_a, "/repo", "index")
        assert len(a._journal.list()) == 1
        assert b._journal.list() == []
    finally:
        vocab.close()


async def test_singletons_resolve_identity_before_storage_and_isolate_vocabulary(
    tmp_path, identity, monkeypatch
):
    monkeypatch.setattr(core_settings, "cache_dir", tmp_path)
    resolved = []
    vocabularies = []
    try:
        for current in (identity, replace(identity, namespace="revision-2")):
            monkeypatch.setattr(singletons, "_storage", AsyncSingleton("test-storage"))
            monkeypatch.setattr(singletons, "_global_vocab", AsyncSingleton("test-vocab"))
            embedder = FakeEmbedder(current)
            embedder.resolve_identity = AsyncMock(return_value=current)
            monkeypatch.setattr(singletons, "get_embedder", AsyncMock(return_value=embedder))
            storage = await singletons.get_storage()
            embedder.resolve_identity.assert_awaited_once()
            assert storage.identity == current
            resolved.append(storage.identity)
            vocab = await singletons.get_global_vocab()
            vocabularies.append(vocab)
            assert (
                Path(vocab.db_path) == tmp_path / f"codesearch_vocabulary_{current.fingerprint}.db"
            )
            assert await singletons.get_global_vocab() is vocab
        assert resolved[0] != resolved[1]
        assert vocabularies[0].db_path != vocabularies[1].db_path
    finally:
        for vocab in vocabularies:
            vocab.close()
