"""Tests for the identity of stored points.

Qdrant is keyed on point ID, so two points that compute the same ID cannot
coexist: the second upsert overwrites the first, silently. Chunk IDs were
derived from (type, path, start_line) alone, which is not unique whenever a
file puts more than one definition on a source line.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from vector_core import SparseVector, generate_point_id

from mcp_codesearch.indexer.chunker import chunk_file
from mcp_codesearch.indexer.discovery import FileInfo
from mcp_codesearch.services.indexing_service import IndexingService, PreparedFile
from mcp_codesearch.storage.qdrant import QdrantStorage

# Idiomatic one-liners, not contrived minification: an inline C++ struct, a
# compact JS class, and a compact Java class all put several definitions on one
# source line.
SAME_LINE_SOURCES = [
    ("javascript", "class A { foo() {} bar() {} }\n", 3),
    ("cpp", "struct W { void render() {} };\n", 2),
    ("java", "class B { void x() {} void y() {} }\n", 3),
]


@pytest.fixture
def storage() -> QdrantStorage:
    return QdrantStorage()


@pytest.fixture
def service(storage: QdrantStorage) -> IndexingService:
    return IndexingService(
        storage=storage,
        embedder=MagicMock(),
        global_vocab=MagicMock(),
    )


class TestPointIdKey:
    def test_chunks_sharing_a_line_get_distinct_ids(self, storage: QdrantStorage) -> None:
        first = storage._point_id("chunk", "src/main.js", start_line=1, ordinal=0)
        second = storage._point_id("chunk", "src/main.js", start_line=1, ordinal=1)

        assert first != second

    def test_ids_are_deterministic(self, storage: QdrantStorage) -> None:
        assert storage._point_id("chunk", "src/main.js", 1, 0) == storage._point_id(
            "chunk", "src/main.js", 1, 0
        )

    def test_start_line_still_participates(self, storage: QdrantStorage) -> None:
        assert storage._point_id("chunk", "src/main.js", 1, 0) != storage._point_id(
            "chunk", "src/main.js", 2, 0
        )

    def test_file_point_ids_are_unchanged(self, storage: QdrantStorage) -> None:
        """File points keep their existing key.

        One point per path is already unique, and changing the key would strand
        every file point in an existing collection: nothing would ever overwrite
        or delete them by ID.
        """
        assert storage._point_id("file", "src/main.py") == generate_point_id("file:src/main.py")


def _prepared(language: str, source: str) -> tuple[PreparedFile, list]:
    """A PreparedFile exactly as _prepare_files would build it."""
    chunks = chunk_file(source, language)
    file_info = FileInfo(
        path=Path(f"/repo/src/sample.{language}"),
        rel_path=f"src/sample.{language}",
        language=language,
        size_bytes=len(source),
        content=source,
        content_hash="deadbeef",
        line_count=source.count("\n"),
        mtime=0.0,
    )
    prepared = PreparedFile(
        file_info=file_info,
        chunks=chunks,
        summary="summary",
        chunk_embedding_texts=[chunk.content for chunk in chunks],
        chunk_vocabulary_texts=[chunk.content for chunk in chunks],
    )
    return prepared, chunks


async def _upserted_points(service: IndexingService, prepared: PreparedFile) -> tuple[list, int]:
    """Drive the real batch path and return (points upserted, chunks counted)."""
    upserted: list = []

    async def capture(_collection, points):
        upserted.extend(points)

    service._storage.upsert_batch = capture
    service._embedder.embed_all = AsyncMock(
        side_effect=lambda texts: [[0.1] for _ in texts]
    )
    service._global_vocab.vectorize_document = MagicMock(
        return_value=SparseVector(indices=[], values=[])
    )

    counted = await service._process_batch([prepared], "test_collection", {})
    return upserted, counted


class TestChunkPointsFromRealSources:
    """These drive ``_process_batch``, the path production indexing takes, so
    they fail on the old point-ID scheme for the right reason rather than on a
    changed signature."""

    @pytest.mark.parametrize(("language", "source", "expected"), SAME_LINE_SOURCES)
    @pytest.mark.asyncio
    async def test_every_chunk_gets_its_own_point(
        self, service: IndexingService, language: str, source: str, expected: int
    ) -> None:
        """Regression: a class and its inline methods collapsed to one point.

        The chunker emits a chunk for the container and one for each definition
        inside it. On a single-line declaration they share a start_line, so all
        of them hashed to the same ID and Qdrant kept whichever was written
        last. The loss was invisible: chunks are counted before the upsert, so
        the tool reported indexing every one of them.
        """
        prepared, chunks = _prepared(language, source)
        assert len(chunks) == expected, "fixture no longer produces same-line chunks"
        assert len({chunk.start_line for chunk in chunks}) == 1

        points, counted = await _upserted_points(service, prepared)

        chunk_points = [p for p in points if p.payload["type"] == "chunk"]
        assert counted == len(chunks)
        assert len({p.id for p in chunk_points}) == len(chunks)

    @pytest.mark.asyncio
    async def test_chunks_on_distinct_lines_are_unaffected(
        self, service: IndexingService
    ) -> None:
        source = "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
        prepared, chunks = _prepared("python", source)
        assert len(chunks) >= 2

        points, _ = await _upserted_points(service, prepared)

        chunk_points = [p for p in points if p.payload["type"] == "chunk"]
        assert len({p.id for p in chunk_points}) == len(chunks)
