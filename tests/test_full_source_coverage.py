"""Complete-source regressions using the real splitter and an offline Qdrant."""

from __future__ import annotations

import json
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from vector_core import EmbeddingClient, GlobalVocabulary
from vector_core.embeddings.client import (
    EmbeddingInputRejectedError,
    EmbeddingInputTooLongError,
    EmbeddingRequestTooLargeError,
)

from mcp_codesearch.indexer import discovery as discovery_module
from mcp_codesearch.indexer.chunker import chunk_file
from mcp_codesearch.indexer.discovery import discover_files
from mcp_codesearch.indexer.journal import IndexIntentStore
from mcp_codesearch.indexer.treesitter import Chunk
from mcp_codesearch.search.query import DenseQueryPreparation, _merge_results, search_codebase
from mcp_codesearch.services import indexing_service as indexing_module
from mcp_codesearch.services.indexing_service import IndexingService
from mcp_codesearch.services.search_service import SearchQuery, SearchService
from mcp_codesearch.storage import qdrant as storage_module
from mcp_codesearch.storage.qdrant import QdrantStorage, SearchResult, index_identity

MARKER = "quasarneedle"


class MarkerEmbedder(EmbeddingClient):
    """Real budget/partition behavior; deterministic inference without HTTP."""

    def __init__(self):
        super().__init__(
            base_url="http://offline.invalid",
            model="offline-marker-model",
            dim=4,
            cache_namespace="offline-coverage",
            profile="raw",
            query_prefix="",
            document_prefix="passage: ",
            max_text_chars=1024,
            max_input_bytes=1100,
            global_concurrency=0,
        )
        self.identity = self.configured_identity()
        self.document_inputs: list[str] = []

    @staticmethod
    def marker_vector(text):
        return [1.0, 0.0, 0.0, 0.0] if MARKER in text else [0.0, 1.0, 0.0, 0.0]

    async def embed_all(self, texts, *, role="document"):
        # Calling the production formatter validates the complete inputs. A
        # truncating formatter would also fail the equality assertion below.
        prepared = self._prepare_texts(texts, role=role)
        prefix = self.document_prefix if role == "document" else self.query_prefix
        assert prepared == [prefix + text for text in texts]
        if role == "document":
            self.document_inputs.extend(texts)
        return [self.marker_vector(text) for text in texts]

    async def embed_single_cached(self, text, *, role="query"):
        return (await self.embed_all([text], role=role))[0]


@pytest.fixture
async def offline_index(tmp_path, monkeypatch):
    """Isolate vocabulary, journal, locks, and all vector storage."""
    client = AsyncQdrantClient(":memory:")
    vocabularies = {}
    embedders = []

    async def forbidden_http(*_args, **_kwargs):
        pytest.fail("Complete-source tests must never make HTTP requests")

    @asynccontextmanager
    async def isolated_lock(*_args, **_kwargs):
        yield

    monkeypatch.setattr(httpx.AsyncClient, "request", forbidden_http)
    monkeypatch.setattr(indexing_module, "async_file_lock", isolated_lock)

    def make():
        embedder = MarkerEmbedder()
        embedders.append(embedder)
        storage = QdrantStorage(url="http://offline.invalid", identity=embedder.identity)
        storage._core._get_client = AsyncMock(return_value=client)
        storage._core.get_client = AsyncMock(return_value=client)
        key = storage.identity.fingerprint
        if key not in vocabularies:
            vocabularies[key] = GlobalVocabulary(tmp_path / f"vocab_{key}.db")
        service = IndexingService(
            storage,
            embedder,
            vocabularies[key],
            journal=IndexIntentStore(tmp_path / "intents.db", namespace=key),
        )
        service._stale_locks_cleaned = True
        return service

    yield make, client
    for embedder in embedders:
        await embedder.close()
    for vocabulary in vocabularies.values():
        vocabulary.close()
    await client.close()


def assert_complete_source(source, chunks):
    """Verify byte-accurate payloads and the union of all real source spans."""
    raw = source.encode("utf-8")
    spans = []
    for chunk in chunks:
        if not chunk.source_coverage:
            continue
        assert chunk.start_byte is not None and chunk.end_byte is not None
        assert 0 <= chunk.start_byte <= chunk.end_byte <= len(raw)
        assert raw[chunk.start_byte : chunk.end_byte].decode("utf-8") == chunk.content
        assert chunk.start_line == raw[: chunk.start_byte].count(b"\n") + 1
        spans.append((chunk.start_byte, chunk.end_byte))
    cursor = 0
    for start, end in sorted(spans):
        assert start <= cursor, f"Unindexed source bytes [{cursor}, {start})"
        cursor = max(cursor, end)
    assert cursor == len(raw)


SOURCES = [
    pytest.param(
        "# module head\nHEAD_VALUE = 'head'\n\ndef first():\n    return 1\n"
        "\nBETWEEN_VALUE = 'middle'\n\ndef last():\n    return 2\n\nTAIL_VALUE = 'tail'\n",
        "python",
        id="module-gaps",
    ),
    pytest.param(
        "class Registry:\n"
        + "".join(f"    attribute_{i} = 'value_{i}'\n" for i in range(75))
        + "    secret = 'class_attribute_tail'\n\n    def lookup(self):\n        return 1\n",
        "python",
        id="large-class-attributes",
    ),
    pytest.param(
        "def oversized():\n" + "    value = 'payload'\n" * 2100 + f"    return '{MARKER}'\n",
        "python",
        id="long-function",
    ),
    pytest.param("DATA = '" + "x" * 40000 + f" {MARKER}'\n", "python", id="single-line"),
    pytest.param(
        "function compact(){" + "let_value=1;" * 4000 + f"return '{MARKER}';}}",
        "javascript",
        id="minified",
    ),
    pytest.param(
        "# café 🛰️\npréface = '你好'\n\ndef unicode_value():\n"
        + "    value = '😀é漢字'\n" * 120
        + "    return '終わり'\n\nfinal = 'naïve'\n",
        "python",
        id="unicode-offsets",
    ),
]


@pytest.mark.parametrize(("source", "language"), SOURCES)
def test_chunker_covers_every_source_byte(source, language):
    chunks = chunk_file(source, language)
    assert_complete_source(source, chunks)
    if "class Registry" in source:
        overviews = [chunk for chunk in chunks if chunk.chunk_type == "class_overview"]
        assert overviews and all(not chunk.source_coverage for chunk in overviews)
        assert any(
            "class_attribute_tail" in chunk.content for chunk in chunks if chunk.source_coverage
        )
        attribute = next(chunk for chunk in chunks if "class_attribute_tail" in chunk.content)
        assert attribute.name == "Registry" and attribute.chunk_type == "class"


@pytest.mark.parametrize(("source", "language"), SOURCES)
async def test_prepare_partitions_all_oversized_source_without_losing_content(
    source, language, offline_index, tmp_path
):
    make, _client = offline_index
    root = tmp_path / "repo"
    root.mkdir()
    filename = "module.js" if language == "javascript" else "module.py"
    (root / filename).write_text(source, encoding="utf-8")
    service = make()
    prepared, tokens = service._prepare_files(list(discover_files(root)))
    assert len(prepared) == 1
    file = prepared[0]
    assert_complete_source(source, file.chunks)
    assert len(tokens) == 1 + len(file.chunks)
    service._embedder._prepare_texts([file.summary, *file.chunk_embedding_texts], role="document")
    assert all(len(text) <= 1024 for text in file.chunk_embedding_texts)
    if len(source) > 30000:
        assert len(file.chunks) > 1
        assert any(MARKER in chunk.content for chunk in file.chunks)


async def snapshot(client, collection):
    points, offset = await client.scroll(collection, limit=1000, with_vectors=True)
    assert offset is None
    return {str(point.id): (point.payload, point.vector) for point in points}


async def test_tail_beyond_old_cutoffs_is_dense_sparse_and_exact_searchable(
    offline_index, tmp_path
):
    make, client = offline_index
    root = tmp_path / "repo"
    root.mkdir()
    source = "def haystack():\n    data = '" + "x" * 40000 + f"'\n    return '{MARKER}'\n"
    assert source.index(MARKER) > 30000
    (root / "tail.py").write_text(source)
    (root / "decoy.py").write_text("def unrelated():\n    return 'ordinary'\n")
    service = make()
    await service.index(str(root))
    collection = service._storage.collection_name(str(root))
    assert any(MARKER in text for text in service._embedder.document_inputs)
    dense = await client.query_points(
        collection,
        query=service._embedder.marker_vector(MARKER),
        using="dense",
        query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="chunk"))]),
        score_threshold=0.9,
        limit=10,
        with_payload=True,
    )
    assert dense.points
    assert all(point.payload["path"] == "tail.py" for point in dense.points)
    assert any(MARKER in point.payload["content"] for point in dense.points)
    sparse = await service._storage.sparse_only_search(
        collection, service._global_vocab.vectorize_query(MARKER), mode="chunk", limit=10
    )
    exact = await service._storage.exact_match_search(collection, MARKER, mode="chunk", limit=10)
    assert sparse and exact
    assert all(result.path == "tail.py" for result in sparse + exact)
    assert any(MARKER in (result.content or "") for result in sparse)
    assert any(MARKER in (result.content or "") for result in exact)
    stored = await snapshot(client, collection)
    chunks = [payload for payload, _ in stored.values() if payload.get("type") == "chunk"]
    assert_complete_source(
        source, [Chunk(**payload) for payload in chunks if payload["path"] == "tail.py"]
    )


@pytest.mark.parametrize("query", [MARKER, f'"{MARKER}"'])
async def test_user_facing_file_mode_rolls_up_tail_only_chunk_hits(query, offline_index, tmp_path):
    make, client = offline_index
    root = tmp_path / "repo"
    root.mkdir()
    (root / "tail.py").write_text(
        "def haystack():\n    data = '" + "x" * 40000 + f"'\n    return '{MARKER}'\n"
    )
    service = make()
    await service.index(str(root))
    collection = service._storage.collection_name(str(root))
    points, _ = await client.scroll(collection, limit=1000)
    file_payloads = [point.payload for point in points if point.payload.get("type") == "file"]
    assert file_payloads and all(MARKER not in payload["summary"] for payload in file_payloads)
    search = SearchService(
        service._storage, service._embedder, service._global_vocab, indexing_service=service
    )
    response = await search.search(
        SearchQuery(query=query, path=str(root), mode="file", output_format="json"), skip_cache=True
    )
    results = json.loads(response.formatted_output)
    assert len(results) == 1
    assert results[0]["path"] == "tail.py" and results[0]["type"] == "file"
    assert response.raw_results[0].line_count is not None


def test_storage_policy_identity_is_idempotent_and_does_not_change_core_cache_identity():
    embedder = MarkerEmbedder()
    core_identity = embedder.configured_identity()
    fingerprint = core_identity.fingerprint
    storage = QdrantStorage(identity=core_identity)
    assert storage.identity.preprocessing == core_identity.preprocessing + ":codesearch-source-v1"
    assert index_identity(storage.identity) == storage.identity
    assert QdrantStorage(identity=storage.identity).identity == storage.identity
    assert embedder.configured_identity() == core_identity
    assert embedder.configured_identity().fingerprint == fingerprint
    assert storage.identity.fingerprint != fingerprint
    assert replace(storage.identity, preprocessing=core_identity.preprocessing) == core_identity


async def test_policy_only_change_builds_new_retained_generation(
    offline_index, tmp_path, monkeypatch
):
    make, client = offline_index
    root = tmp_path / "repo"
    root.mkdir()
    (root / "source.py").write_text("def unchanged():\n    return 1\n")
    old = make()
    await old.index(str(root))
    old_name = old._storage.collection_name(str(root))
    old_points = await snapshot(client, old_name)
    assert (await old._storage.get_metadata(old_name))["source_policy_version"] == "v1"
    monkeypatch.setattr(storage_module, "SOURCE_POLICY_VERSION", "v2")
    current = make()
    assert current._embedder.identity == old._embedder.identity
    new_name = current._storage.collection_name(str(root))
    assert new_name != old_name
    assert old._storage.collection_name(str(root)) == old_name
    files, _chunks, stats = await current.index(str(root))
    assert files == 1 and not stats.was_incremental
    assert (await current._storage.get_metadata(new_name))["source_policy_version"] == "v2"
    assert await snapshot(client, old_name) == old_points
    assert old_name in await current._storage.list_preserved_collections()


async def assert_vocabulary_matches_points(service, client, collection):
    points, _ = await client.scroll(collection, limit=1000, with_vectors=["sparse"])
    indexed = [point for point in points if point.payload.get("type") in {"file", "chunk"}]
    assert service._global_vocab.total_docs == len(indexed)
    frequencies = Counter()
    for point in indexed:
        indices = point.vector["sparse"].indices
        tokens = service._global_vocab.get_tokens_by_indices(indices)
        frequencies.update(tokens[index] for index in indices)
    registered = service._global_vocab._get_doc_freq()
    assert all(count >= 0 for count in registered.values())
    assert {token: count for token, count in registered.items() if count} == dict(frequencies)


async def test_incremental_segment_shrink_grow_and_delete_remove_stale_vocabulary(
    offline_index, tmp_path
):
    make, client = offline_index
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "segmented.py"
    path.write_text("DATA = '" + "oldword " * 5000 + "'\n")
    service = make()
    await service.index(str(root))
    collection = service._storage.collection_name(str(root))
    initial = await snapshot(client, collection)
    await assert_vocabulary_matches_points(service, client, collection)
    path.write_text("DATA = 'shortword'\n")
    _, _, stats = await service.index(str(root))
    assert stats.was_incremental and stats.files_modified == 1
    shortened = await snapshot(client, collection)
    assert len(shortened) < len(initial)
    assert set(initial) - set(shortened)
    assert all("oldword" not in payload.get("content", "") for payload, _ in shortened.values())
    await assert_vocabulary_matches_points(service, client, collection)
    path.write_text("DATA = '" + "newword " * 6000 + "'\n")
    _, _, stats = await service.index(str(root))
    assert stats.was_incremental and stats.files_modified == 1
    grown = await snapshot(client, collection)
    assert len(grown) > len(shortened)
    assert all("shortword" not in payload.get("content", "") for payload, _ in grown.values())
    await assert_vocabulary_matches_points(service, client, collection)
    path.unlink()
    _, _, stats = await service.index(str(root))
    assert stats.was_incremental and stats.files_deleted == 1
    assert await service._storage.count_index_documents(collection) == 0
    assert service._global_vocab.total_docs == 0
    assert not any(service._global_vocab._get_doc_freq().values())
    assert service._journal.list() == []


async def test_preparation_failure_names_source_and_never_silently_skips_file(
    offline_index, tmp_path, monkeypatch
):
    make, _client = offline_index
    root = tmp_path / "repo"
    root.mkdir()
    (root / "broken.py").write_text("def real_code():\n    return 1\n")
    service = make()

    def failed_split(*_args, **_kwargs):
        raise ValueError("context budget exhausted")

    monkeypatch.setattr(service._embedder, "split_text", failed_split)
    with pytest.raises(ValueError, match="broken.py.*context budget exhausted"):
        await service.index(str(root))
    collection = service._storage.collection_name(str(root))
    assert await service._storage.count_index_documents(collection) == 0
    assert service._global_vocab.total_docs == 0


async def test_import_context_and_role_prefix_share_each_segment_budget(offline_index, tmp_path):
    make, _client = offline_index
    root = tmp_path / "repo"
    root.mkdir()
    source = "import " + "module" * 80 + "\n\ndef unicode_value():\n"
    source += "    value = '😀é漢字'\n" * 100
    source += f"    return '{MARKER}'\n"
    (root / "context.py").write_text(source, encoding="utf-8")
    service = make()
    prepared, _tokens = service._prepare_files(list(discover_files(root)))
    file = prepared[0]
    assert_complete_source(source, file.chunks)
    assert len(file.chunks) > 5
    assert all(text.startswith("Uses: ") for text in file.chunk_embedding_texts)
    formatted = service._embedder._prepare_texts(file.chunk_embedding_texts, role="document")
    assert all(len(text.encode("utf-8")) <= 1100 for text in formatted)
    assert any(MARKER in text for text in formatted)


async def test_unsplittable_import_context_fails_explicitly(offline_index, tmp_path):
    make, _client = offline_index
    root = tmp_path / "repo"
    root.mkdir()
    (root / "oversized_context.py").write_text(
        "import " + "module" * 250 + "\n\ndef real_code():\n    return 1\n"
    )
    service = make()
    with pytest.raises(ValueError, match="oversized_context.py"):
        await service.index(str(root))
    collection = service._storage.collection_name(str(root))
    assert await service._storage.count_index_documents(collection) == 0
    assert service._global_vocab.total_docs == 0


@pytest.mark.parametrize(
    "error",
    [
        EmbeddingInputTooLongError(0, "query", 20, 10, "tokens"),
        EmbeddingRequestTooLargeError("Embedding backend rejected the complete request"),
        EmbeddingInputRejectedError(400, "complete input exceeds backend context"),
        EmbeddingInputRejectedError(422, "complete input exceeds backend context"),
    ],
)
async def test_explicit_input_rejection_never_falls_back_to_sparse_search(error):
    storage = MagicMock()
    storage.hybrid_search = AsyncMock()
    storage.sparse_only_search = AsyncMock()
    storage.exact_match_search = AsyncMock()
    embedder = MagicMock()
    embedder.embed_single_cached = AsyncMock(side_effect=error)
    vocabulary = MagicMock()
    with pytest.raises(type(error), match="Embedding"):
        await search_codebase("complete query", "/offline/repo", storage, embedder, vocabulary)
    storage.hybrid_search.assert_not_awaited()
    storage.sparse_only_search.assert_not_awaited()
    storage.exact_match_search.assert_not_awaited()
    vocabulary.vectorize_query.assert_not_called()


def test_same_line_source_segments_survive_result_merging():
    first = SearchResult(
        path="one.py",
        score=1,
        point_type="chunk",
        language="python",
        start_line=1,
        start_byte=0,
        end_byte=100,
    )
    second = first.model_copy(update={"start_byte": 100, "end_byte": 200})
    assert _merge_results([first], [first, second]) == [first, second]


@pytest.mark.parametrize("kind", ["struct", "impl"])
def test_large_container_gap_preserves_original_symbol_kind(kind):
    source = f"{kind} Registry {{\n" + "    // filler\n" * 60
    source += f"    {MARKER}: i32,\n" if kind == "struct" else f"    const {MARKER}: i32 = 1;\n"
    source += "}\n"
    chunks = chunk_file(source, "rust")
    assert_complete_source(source, chunks)
    match = next(chunk for chunk in chunks if chunk.source_coverage and MARKER in chunk.content)
    assert match.name == "Registry" and match.chunk_type == kind


async def test_exact_search_keeps_distinct_same_line_source_offsets(offline_index, tmp_path):
    make, _client = offline_index
    root = tmp_path / "repo"
    root.mkdir()
    source = "DATA = '" + (MARKER + " ") * 300 + "'\n"
    (root / "one.py").write_text(source)
    service = make()
    await service.index(str(root))
    collection = service._storage.collection_name(str(root))
    results = await service._storage.exact_match_search(collection, MARKER, mode="chunk", limit=100)
    assert len(results) > 2
    assert all(result.start_line == 1 for result in results)
    assert all(result.start_byte is not None and result.end_byte is not None for result in results)
    assert len({result.start_byte for result in results}) == len(results)
    assert _merge_results(results, results) == results
    raw = source.encode("utf-8")
    for result in results:
        assert raw[result.start_byte : result.end_byte].decode("utf-8") == result.content


@pytest.mark.parametrize("platform", ["linux", "win32"])
async def test_crlf_byte_offsets_match_original_file_bytes(
    platform, offline_index, tmp_path, monkeypatch
):
    make, client = offline_index
    monkeypatch.setattr(discovery_module, "sys", SimpleNamespace(platform=platform))
    root = tmp_path / "repo"
    root.mkdir()
    source = "HEAD = 'café 😀'\r\ndef target():\r\n    return 'tail'\r\nTAIL = 1\r\n"
    raw = source.encode("utf-8")
    (root / "crlf.py").write_bytes(raw)
    files = list(discover_files(root))
    assert len(files) == 1 and files[0].content.encode("utf-8") == raw
    service = make()
    prepared, _tokens = service._prepare_files(files)
    assert_complete_source(source, prepared[0].chunks)
    target = next(chunk for chunk in prepared[0].chunks if chunk.name == "target")
    assert target.start_byte == raw.index(b"def target")
    await service.index(str(root))
    collection = service._storage.collection_name(str(root))
    stored = await snapshot(client, collection)
    chunks = [Chunk(**payload) for payload, _ in stored.values() if payload.get("type") == "chunk"]
    assert_complete_source(source, chunks)


@pytest.mark.parametrize("query", [f'"{MARKER}"', MARKER, f"fn:{MARKER}", f"fn:{MARKER} filler"])
async def test_file_rollup_expands_past_repeated_segments(
    query, offline_index, tmp_path, monkeypatch
):
    make, client = offline_index
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text(
        f"def {MARKER}():\n" + f"    # {MARKER} filler\n" * 1000 + "    return 1\n"
    )
    (root / "b.py").write_text(f"def {MARKER}():\n    return 2\n")
    service = make()
    await service.index(str(root))
    collection = service._storage.collection_name(str(root))
    points, _offset = await client.scroll(collection, limit=1000)
    candidates = [
        SearchResult(
            path=point.payload["path"],
            point_type="chunk",
            language="python",
            score=1.0 if point.payload["path"] == "a.py" else 0.9,
            name=point.payload.get("name"),
            chunk_type=point.payload["chunk_type"],
            content=point.payload["content"],
            start_line=point.payload["start_line"],
            start_byte=point.payload["start_byte"],
            end_byte=point.payload["end_byte"],
        )
        for point in points
        if point.payload.get("type") == "chunk" and point.payload.get("name")
    ]
    candidates.sort(key=lambda result: (result.path, result.start_byte))
    assert sum(result.path == "a.py" for result in candidates) > 20
    limits = []

    async def ordered_candidates(*_args, **kwargs):
        limits.append(kwargs["limit"])
        return candidates[: kwargs["limit"]]

    monkeypatch.setattr(service._storage, "exact_match_search", ordered_candidates)
    monkeypatch.setattr(service._storage, "hybrid_search", ordered_candidates)
    monkeypatch.setattr(service._storage, "sparse_only_search", ordered_candidates)
    for dense in (
        DenseQueryPreparation(vector=[1.0, 0.0, 0.0, 0.0]),
        DenseQueryPreparation(degraded_reason="offline test"),
    ):
        limits.clear()
        results = await search_codebase(
            query,
            root,
            service._storage,
            service._embedder,
            service._global_vocab,
            mode="file",
            limit=2,
            dense_preparation=dense,
        )
        assert {result.path for result in results} == {"a.py", "b.py"}
        assert len(set(limits)) > 1


async def test_hybrid_prefetch_grows_with_adaptive_point_budget(offline_index, monkeypatch):
    make, _client = offline_index
    service = make()
    searcher = MagicMock()
    searcher.search = AsyncMock(return_value=[])
    monkeypatch.setattr(storage_module, "HybridSearcher", MagicMock(return_value=searcher))
    await service._storage.hybrid_search(
        "offline-collection",
        [1.0, 0.0, 0.0, 0.0],
        service._global_vocab.vectorize_query(MARKER),
        limit=128,
        prefetch_limit=4,
    )
    assert searcher.search.await_args.kwargs["prefetch_limit"] == 128


@pytest.mark.parametrize("platform", ["linux", "win32"])
async def test_invalid_utf8_is_reported_and_never_invents_source_bytes(
    platform, offline_index, tmp_path, monkeypatch, caplog
):
    make, client = offline_index
    monkeypatch.setattr(discovery_module, "sys", SimpleNamespace(platform=platform))
    root = tmp_path / "repo"
    root.mkdir()
    (root / "bad.py").write_bytes(b"HEAD = 1\r\nINVALID = '\xff'\r\n")
    good = root / "good.py"
    good.write_bytes(b"def valid():\r\n    return 1\r\n")
    with caplog.at_level("WARNING", logger="mcp_codesearch.indexer.discovery"):
        files = list(discover_files(root))
    assert [file.rel_path for file in files] == ["good.py"]
    assert "bad.py" in caplog.text and "invalid UTF-8" in caplog.text
    service = make()
    await service.index(str(root))
    collection = service._storage.collection_name(str(root))
    stored = await snapshot(client, collection)
    assert all(payload.get("path") != "bad.py" for payload, _ in stored.values())
    good.write_bytes(b"def invalid():\r\n    return '\xff'\r\n")
    with caplog.at_level("WARNING", logger="mcp_codesearch.indexer.discovery"):
        await service.index(str(root))
    assert "good.py" in caplog.text and "invalid UTF-8" in caplog.text
    assert await service._storage.count_index_documents(collection) == 0
