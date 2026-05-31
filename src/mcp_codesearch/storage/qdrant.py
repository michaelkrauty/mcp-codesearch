"""Qdrant storage layer with hybrid search support for code search.

This module uses vector-core for shared infrastructure (client lifecycle,
collection management, point ID generation) while providing code-search-specific
functionality (hybrid search with RRF, file/chunk indexing, exact match search).
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)
from qdrant_client.models import (
    SparseVector as QdrantSparseVector,
)
from vector_core.embeddings.sparse import SparseVector
from vector_core.storage.hybrid import HybridSearcher
from vector_core.storage.qdrant import (
    QdrantStorage as VectorCoreStorage,
)
from vector_core.storage.qdrant import (
    generate_collection_name,
    generate_point_id,
)

from mcp_codesearch.settings import settings

logger = logging.getLogger(__name__)

# Prefix for all codesearch collections
COLLECTION_PREFIX = "codesearch"


class EmbeddingDimMismatchError(Exception):
    """Raised when an existing collection's stored vectors were indexed with a
    different embedding dimension than the one currently configured.

    Changing the embedding model so its output dimension differs makes a stored
    index unusable: Qdrant rejects every query and upsert because the vector
    sizes no longer match. Rather than surfacing that cryptic dimension error
    deep inside a search, indexing refuses to touch the collection and points
    the user at ``force_reindex`` to rebuild it with the current model.
    """

    def __init__(self, collection: str, expected: int, actual: int) -> None:
        self.collection = collection
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"collection {collection!r} was indexed with {actual}-dimensional "
            f"dense vectors, but the configured embedding model now produces "
            f"{expected}-dimensional vectors"
        )


def collection_name(codebase_path: str) -> str:
    """
    Generate deterministic collection name from absolute path.

    Uses vector-core's generate_collection_name with "codesearch" prefix.

    Args:
        codebase_path: Absolute path to codebase root

    Returns:
        Collection name in format "codesearch_{hash[:12]}"
    """
    return generate_collection_name(codebase_path, prefix=COLLECTION_PREFIX)


# Point ID cache size - eliminates ~11,000+ redundant SHA256 operations per index
_POINT_ID_CACHE_SIZE = 10000


@lru_cache(maxsize=_POINT_ID_CACHE_SIZE)
def _cached_point_id(key: str) -> str:
    """Cached point ID generation to avoid redundant SHA256 operations."""
    return generate_point_id(key)


class FilePoint(BaseModel):
    """File-level index point."""

    path: str
    abs_path: str
    language: str
    file_hash: str
    summary: str
    line_count: int
    size_bytes: int
    mtime: float = 0.0  # Modification time for fast change detection


class ChunkPoint(BaseModel):
    """Chunk-level index point."""

    path: str
    abs_path: str
    language: str
    file_hash: str
    chunk_type: str  # function, class, method, block
    name: str | None
    start_line: int
    end_line: int
    content: str
    context: str | None


class SearchResult(BaseModel):
    """Search result with score and metadata."""

    path: str
    score: float
    point_type: str  # file or chunk
    language: str
    # File-specific
    summary: str | None = None
    line_count: int | None = None
    # Chunk-specific
    chunk_type: str | None = None
    name: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    content: str | None = None
    # Degraded mode indicator (sparse-only fallback)
    degraded: bool = False


class QdrantStorage:
    """
    Qdrant storage with hybrid dense+sparse vectors for code search.

    Uses vector-core for shared infrastructure:
    - Client lifecycle management
    - Collection operations (create, delete, exists)

    Provides code-search-specific functionality:
    - String point IDs (full SHA256 for collision resistance)
    - File/chunk indexing with domain-specific payloads
    - Hybrid search with weighted RRF fusion
    - Exact substring search fallback
    """

    def __init__(self, url: str | None = None):
        self.url = url or settings.qdrant_url
        # Use vector-core for shared infrastructure
        self._core = VectorCoreStorage(
            url=self.url,
            embedding_dim=settings.embedding_dim,
        )

    async def _get_client(self) -> AsyncQdrantClient:
        """Get the underlying Qdrant client."""
        return await self._core.get_client()

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self._core.close()

    # =========================================================================
    # Collection Management - Delegated to vector-core
    # =========================================================================

    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        return await self._core.collection_exists(name)

    async def create_collection(self, name: str) -> None:
        """Create collection with hybrid vector config."""
        await self._core.create_collection(name, dense_dim=settings.embedding_dim)

    async def delete_collection(self, name: str) -> None:
        """Delete collection."""
        await self._core.delete_collection(name)

    async def list_collections(self) -> list[str]:
        """List all codesearch collections."""
        return await self._core.list_collections(prefix="codesearch_")

    async def get_dense_dim(self, name: str) -> int | None:
        """Return the dense-vector dimension recorded for an existing collection.

        Reads the collection's stored vector configuration from Qdrant. Returns
        ``None`` when the collection does not expose a named ``dense`` vector
        (for example a collection created by some other tool), so callers can
        treat an unreadable dimension as "cannot verify" rather than a mismatch.
        """
        client = await self._get_client()
        info = await client.get_collection(name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            dense = vectors.get("dense")
            if dense is not None:
                return dense.size
        return None

    # =========================================================================
    # Point ID Generation - Code-search-specific (string IDs)
    # =========================================================================

    def _point_id(self, point_type: str, path: str, start_line: int | None = None) -> str:
        """
        Generate deterministic point ID for a file or chunk.

        Uses UUID format for Qdrant compatibility (Qdrant only accepts UUIDs or integers).
        Point IDs are cached to eliminate redundant SHA256 operations during batch upserts.

        Args:
            point_type: "file" or "chunk"
            path: Relative file path
            start_line: Starting line number (for chunks)

        Returns:
            UUID-formatted string derived from SHA256 hash
        """
        key = f"{point_type}:{path}"
        if start_line is not None:
            key += f":{start_line}"
        return _cached_point_id(key)

    # =========================================================================
    # Point Operations - Code-search-specific (domain payloads, string IDs)
    # =========================================================================

    async def upsert_file(
        self,
        collection: str,
        file: FilePoint,
        dense_vector: list[float],
        sparse_vector: SparseVector,
    ) -> None:
        """Upsert a file-level point."""
        client = await self._get_client()

        point = PointStruct(
            id=self._point_id("file", file.path),
            vector={
                "dense": dense_vector,
                "sparse": QdrantSparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
            },
            payload={
                "type": "file",
                "path": file.path,
                "abs_path": file.abs_path,
                "language": file.language,
                "file_hash": file.file_hash,
                "summary": file.summary,
                "line_count": file.line_count,
                "size_bytes": file.size_bytes,
                "mtime": file.mtime,
                "indexed_at": datetime.now(UTC).isoformat(),
            },
        )

        await client.upsert(collection, [point])

    async def upsert_chunk(
        self,
        collection: str,
        chunk: ChunkPoint,
        dense_vector: list[float],
        sparse_vector: SparseVector,
    ) -> None:
        """Upsert a chunk-level point."""
        client = await self._get_client()

        point = PointStruct(
            id=self._point_id("chunk", chunk.path, chunk.start_line),
            vector={
                "dense": dense_vector,
                "sparse": QdrantSparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
            },
            payload={
                "type": "chunk",
                "path": chunk.path,
                "abs_path": chunk.abs_path,
                "language": chunk.language,
                "file_hash": chunk.file_hash,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content[:settings.max_payload_content_chars],
                "context": chunk.context,
                "indexed_at": datetime.now(UTC).isoformat(),
            },
        )

        await client.upsert(collection, [point])

    async def upsert_batch(
        self,
        collection: str,
        points: list[PointStruct],
        batch_size: int = 100,  # Reduced from 500 for stability
        concurrency: int | None = None,
        max_retries: int = 3,
    ) -> None:
        """
        Batch upsert points with concurrent batch processing and retry logic.

        Args:
            collection: Collection name
            points: Points to upsert
            batch_size: Size of each batch
            concurrency: Max concurrent upserts (default from settings)
            max_retries: Max retry attempts per batch on transient failures
        """
        if not points:
            return

        client = await self._get_client()
        concurrency = concurrency or settings.upsert_concurrency

        # Split into batches
        batches = [
            points[i : i + batch_size]
            for i in range(0, len(points), batch_size)
        ]

        # Use semaphore to limit concurrent upserts
        semaphore = asyncio.Semaphore(concurrency)

        async def upsert_with_retry(batch: list[PointStruct]) -> None:
            """Upsert a batch with exponential backoff retry."""
            async with semaphore:
                last_error = None
                for attempt in range(max_retries):
                    try:
                        await client.upsert(collection, batch)
                        return
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            # Exponential backoff: 1s, 2s, 4s
                            await asyncio.sleep(2 ** attempt)
                if last_error:
                    raise last_error

        # Run all batches concurrently (limited by semaphore) with timeout
        # Create explicit tasks for proper cancellation handling on timeout
        tasks = [
            asyncio.create_task(upsert_with_retry(batch))
            for batch in batches
        ]
        try:
            async with asyncio.timeout(settings.upsert_batch_timeout):
                await asyncio.gather(*tasks)
        except TimeoutError as e:
            # Cancel all pending tasks to prevent resource leak
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellations to complete (suppress CancelledError)
            await asyncio.gather(*tasks, return_exceptions=True)
            msg = (
                f"Batch upsert timed out after {settings.upsert_batch_timeout}s "
                f"({len(batches)} batches, {len(points)} points)"
            )
            logger.error(msg)
            # Raise with a message so the MCP tool layer doesn't surface an
            # empty-string TimeoutError to the client.
            raise TimeoutError(msg) from e

    async def delete_by_path(self, collection: str, path: str) -> None:
        """Delete all points for a file path."""
        await self._core.delete_by_filter(collection, "path", path)

    async def delete_by_paths_batch(
        self,
        collection: str,
        paths: list[str],
        batch_size: int = 100,
    ) -> int:
        """
        Delete all points for multiple file paths in batched operations.

        Uses OR filter to delete multiple paths per Qdrant call, reducing
        network overhead for incremental indexing with many deletions.

        Args:
            collection: Collection name
            paths: List of file paths to delete
            batch_size: Max paths per Qdrant delete call (default 100)

        Returns:
            Number of paths processed
        """
        if not paths:
            return 0

        client = await self._get_client()
        processed = 0

        # Process paths in batches to avoid too-large filter conditions
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]

            # Build OR filter for all paths in this batch
            filter_conditions = [
                FieldCondition(key="path", match=MatchValue(value=p))
                for p in batch_paths
            ]

            delete_filter = Filter(should=filter_conditions)  # type: ignore[arg-type]

            await client.delete(
                collection_name=collection,
                points_selector=delete_filter,
            )

            processed += len(batch_paths)

        return processed

    async def get_stored_content_for_path(
        self, collection: str, path: str
    ) -> list[str]:
        """
        Get stored text content for a file path (summary + all chunk content).

        Used before deletion to retrieve tokens for vocabulary update.

        Returns:
            List of text strings (file summary, chunk content) for tokenization.
        """
        client = await self._get_client()

        points, _ = await client.scroll(
            collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="path", match=MatchValue(value=path)),
                ],
            ),
            limit=1000,  # Should be enough for any file
            with_payload=["type", "summary", "content"],
        )

        texts = []
        for point in points:
            if point.payload:
                # File point has summary
                if point.payload.get("type") == "file":
                    summary = point.payload.get("summary", "")
                    if summary:
                        texts.append(summary)
                # Chunk point has content
                elif point.payload.get("type") == "chunk":
                    content = point.payload.get("content", "")
                    if content:
                        texts.append(content)

        return texts

    # =========================================================================
    # Indexed Files Queries - Code-search-specific
    # =========================================================================

    async def get_indexed_files(self, collection: str) -> dict[str, str]:
        """Get map of path -> file_hash for all indexed files."""
        metadata = await self.get_indexed_files_metadata(collection)
        return {path: meta["file_hash"] for path, meta in metadata.items()}

    async def get_indexed_files_metadata(
        self, collection: str
    ) -> dict[str, dict[str, Any]]:
        """
        Get full metadata for all indexed files.

        Returns:
            Dict mapping rel_path -> {"file_hash", "mtime", "size_bytes"}
        """
        client = await self._get_client()

        result: dict[str, dict[str, Any]] = {}
        offset = None

        while True:
            points, offset = await client.scroll(
                collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="type", match=MatchValue(value="file")),
                    ],
                ),
                limit=5000,
                offset=offset,
                with_payload=["path", "file_hash", "mtime", "size_bytes"],
            )

            for point in points:
                if point.payload:
                    path = point.payload.get("path")
                    if path is None:
                        logger.warning(f"Point missing 'path' in payload: {point.id}")
                        continue
                    result[path] = {
                        "file_hash": point.payload.get("file_hash", ""),
                        "mtime": point.payload.get("mtime", 0.0),
                        "size_bytes": point.payload.get("size_bytes", 0),
                    }

            if offset is None:
                break

        return result

    # =========================================================================
    # Hybrid Search - Uses vector-core HybridSearcher for RRF fusion
    # =========================================================================

    async def hybrid_search(
        self,
        collection: str,
        dense_query: list[float],
        sparse_query: SparseVector,
        mode: Literal["file", "chunk", "both"] = "both",
        language: str | None = None,
        limit: int = 10,
        prefetch_limit: int | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search using RRF fusion of dense and sparse results.

        Delegates RRF computation to vector-core's HybridSearcher while handling
        code-search-specific filtering (mode, language) and result conversion.

        Args:
            prefetch_limit: How many candidates to fetch from each search type.
                           Higher = better quality but slower. Default from settings.
            dense_weight: Weight for dense (semantic) vectors in RRF. Default from settings.
            sparse_weight: Weight for sparse (TF-IDF) vectors in RRF. Default from settings.
        """
        # Build code-search-specific filters
        filter_conditions: list[FieldCondition] = []
        if mode == "file":
            filter_conditions.append(
                FieldCondition(key="type", match=MatchValue(value="file"))
            )
        elif mode == "chunk":
            filter_conditions.append(
                FieldCondition(key="type", match=MatchValue(value="chunk"))
            )
        if language:
            filter_conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language))
            )

        # Use vector-core's HybridSearcher for RRF fusion
        searcher = HybridSearcher(
            storage=self._core,
            dense_weight=dense_weight if dense_weight is not None else settings.dense_weight,
            sparse_weight=sparse_weight if sparse_weight is not None else settings.sparse_weight,
            rrf_k=settings.rrf_k,
        )

        generic_results = await searcher.search(
            collection=collection,
            dense_query=dense_query,
            sparse_query=sparse_query,
            limit=limit,
            prefetch_limit=prefetch_limit or settings.rrf_prefetch_limit,
            filter_conditions=filter_conditions if filter_conditions else None,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        # Convert generic SearchResult to code-search-specific SearchResult
        results = []
        for generic in generic_results:
            p = generic.payload or {}
            point_type = p.get("type", "unknown")

            # Skip results with missing path (malformed data)
            path = p.get("path")
            if not path:
                logger.warning(f"Skipping hybrid search result with missing path: score={generic.score}")
                continue

            result = SearchResult(
                path=path,
                score=generic.score,
                point_type=point_type,
                language=p.get("language", ""),
            )

            if point_type == "file":
                result.summary = p.get("summary")
                result.line_count = p.get("line_count")
            elif point_type == "chunk":
                result.chunk_type = p.get("chunk_type")
                result.name = p.get("name")
                result.start_line = p.get("start_line")
                result.end_line = p.get("end_line")
                result.content = p.get("content")

            results.append(result)

        return results

    # =========================================================================
    # Sparse-Only Search - Fallback when embedding service is unavailable
    # =========================================================================

    async def sparse_only_search(
        self,
        collection: str,
        sparse_query: SparseVector,
        mode: Literal["file", "chunk", "both"] = "both",
        language: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Sparse-only search using TF-IDF vectors when embedding service is unavailable.

        This is a degraded fallback that provides keyword-based results when the
        dense embedding service is down. Results are marked with
        degraded=True to indicate reduced search quality.

        Args:
            collection: Collection name
            sparse_query: Sparse vector from GlobalVocabulary.vectorize_query()
            mode: "file", "chunk", or "both"
            language: Optional language filter
            limit: Max results

        Returns:
            List of SearchResult with degraded=True
        """
        client = await self._get_client()

        # Build filters
        filter_conditions: list[FieldCondition] = []
        if mode == "file":
            filter_conditions.append(
                FieldCondition(key="type", match=MatchValue(value="file"))
            )
        elif mode == "chunk":
            filter_conditions.append(
                FieldCondition(key="type", match=MatchValue(value="chunk"))
            )
        if language:
            filter_conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None  # type: ignore[arg-type]

        # Search using only sparse vectors
        response = await client.query_points(
            collection,
            query=QdrantSparseVector(
                indices=sparse_query.indices,
                values=sparse_query.values,
            ),
            using="sparse",
            limit=limit,
            query_filter=query_filter,
        )

        # Convert to SearchResult with degraded flag
        results = []
        for point in response.points:
            p = point.payload or {}
            point_type = p.get("type", "unknown")

            # Skip results with missing path (malformed data)
            path = p.get("path")
            if not path:
                logger.warning(f"Skipping sparse search result with missing path: point_id={point.id}")
                continue

            result = SearchResult(
                path=path,
                score=point.score or 0.0,
                point_type=point_type,
                language=p.get("language", ""),
                degraded=True,  # Mark as degraded mode
            )

            if point_type == "file":
                result.summary = p.get("summary")
                result.line_count = p.get("line_count")
            elif point_type == "chunk":
                result.chunk_type = p.get("chunk_type")
                result.name = p.get("name")
                result.start_line = p.get("start_line")
                result.end_line = p.get("end_line")
                result.content = p.get("content")

            results.append(result)

        return results

    # =========================================================================
    # Exact Match Search - Code-search-specific (fallback for semantic failures)
    # =========================================================================

    # Safety limit for scroll loops to prevent runaway resource consumption
    _MAX_SCROLL_ITERATIONS = 1000  # 1000 * 1000 = 1M points max

    # Payload fields needed for exact match search (avoiding full content load)
    # This reduces payload size significantly (2-10x faster for large codebases)
    _EXACT_MATCH_PAYLOAD_FIELDS = [
        "type", "path", "name", "summary", "content", "language",
        "chunk_type", "start_line", "end_line", "line_count",
    ]

    async def exact_match_search(  # noqa: PLR0912, PLR0915
        self,
        collection: str,
        query: str,
        mode: Literal["file", "chunk", "both"] = "both",
        language: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Fallback exact substring search for when semantic search fails.

        Scans stored content/summary fields for substring matches.
        This is slower than vector search but catches exact matches.
        """
        client = await self._get_client()
        query_lower = query.lower()

        # Build filter
        filter_conditions = []
        if mode == "file":
            filter_conditions.append(
                FieldCondition(key="type", match=MatchValue(value="file"))
            )
        elif mode == "chunk":
            filter_conditions.append(
                FieldCondition(key="type", match=MatchValue(value="chunk"))
            )
        if language:
            filter_conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None  # type: ignore[arg-type]

        results: list[SearchResult] = []
        offset = None
        high_quality_count = 0  # Track high-quality matches for early termination

        # Early termination thresholds
        _EARLY_TERMINATION_THRESHOLD = 10
        _HIGH_QUALITY_SCORE = 2.5  # Name match (3.0) or summary match (2.0)

        # Pre-compile regex pattern outside loop with IGNORECASE (15-25% faster fallback search)
        # Using IGNORECASE avoids .lower() calls on each field
        try:
            compiled_pattern = re.compile(r"\b" + re.escape(query) + r"\b", re.IGNORECASE)
        except re.error as e:
            logger.warning(f"Regex compilation failed for query '{query[:50]}': {e}")
            compiled_pattern = None  # Fall back to substring search

        def _safe_regex_search(
            compiled: re.Pattern[str] | None,
            text: str,
            fallback_query: str,
            timeout_chars: int = 50000,
        ) -> re.Match[str] | bool | None:
            """
            Safe regex search with size limit to prevent catastrophic backtracking.

            For very large texts or invalid patterns, fall back to simple substring match.
            Returns Match object, True (for substring fallback), or None.
            """
            if not text:
                return None
            if len(text) > timeout_chars or compiled is None:
                # Fall back to simple case-insensitive substring for large texts or invalid patterns
                return fallback_query in text.lower()
            return compiled.search(text)

        for _iteration in range(self._MAX_SCROLL_ITERATIONS):
            if len(results) >= limit:
                break
            # Early termination: if we have enough high-quality results, stop searching
            if high_quality_count >= _EARLY_TERMINATION_THRESHOLD:
                break
            points, offset = await client.scroll(
                collection,
                scroll_filter=query_filter,
                limit=1000,  # Increased from 200 for 25-40% faster fallback
                offset=offset,
                with_payload=self._EXACT_MATCH_PAYLOAD_FIELDS,
            )

            for point in points:
                p = point.payload or {}
                point_type = p.get("type", "")

                # Skip metadata point
                if point_type == "__metadata__":
                    continue

                # Check for word boundary match in searchable fields
                # Using IGNORECASE regex avoids .lower() calls
                content = p.get("content", "") or ""
                summary = p.get("summary", "") or ""
                name = p.get("name", "") or ""

                # Check each field separately for field-based scoring
                # No .lower() needed since regex has IGNORECASE flag
                name_match = _safe_regex_search(compiled_pattern, name, query_lower)
                summary_match = _safe_regex_search(compiled_pattern, summary, query_lower)
                content_match = _safe_regex_search(compiled_pattern, content, query_lower)

                if name_match or summary_match or content_match:
                    # Skip results with missing path (malformed data)
                    path = p.get("path")
                    if not path:
                        logger.warning(f"Skipping exact match result with missing path: point_id={point.id}")
                        continue

                    # Field-based scoring: name > summary > content
                    if name_match:
                        score = 3.0  # Name match gets highest score
                    elif summary_match:
                        score = 2.0  # Summary match medium
                    else:
                        score = 1.0  # Content match baseline

                    # Track high-quality results for early termination
                    if score >= _HIGH_QUALITY_SCORE:
                        high_quality_count += 1

                    result = SearchResult(
                        path=path,
                        score=score,
                        point_type=point_type,
                        language=p.get("language", ""),
                    )

                    if point_type == "file":
                        result.summary = p.get("summary")
                        result.line_count = p.get("line_count")
                    elif point_type == "chunk":
                        result.chunk_type = p.get("chunk_type")
                        result.name = name
                        result.start_line = p.get("start_line")
                        result.end_line = p.get("end_line")
                        result.content = p.get("content")

                    results.append(result)

                    if len(results) >= limit:
                        break

            if offset is None:
                break
        else:
            # Loop completed without break - hit max iterations
            logger.warning(
                f"exact_match_search exceeded {self._MAX_SCROLL_ITERATIONS} iterations "
                f"for query '{query[:50]}...' in collection {collection}"
            )

        return results

    # =========================================================================
    # Metadata Storage - Adapted from vector-core
    # =========================================================================

    async def store_metadata(
        self,
        collection: str,
        codebase_path: str,
    ) -> None:
        """Store collection metadata."""
        await self._core.store_metadata(
            collection,
            {
                "codebase_path": codebase_path,
            },
        )

    async def get_metadata(self, collection: str) -> dict[str, Any] | None:
        """Get collection metadata."""
        return await self._core.get_metadata(collection)

    async def infer_codebase_path(self, collection: str) -> str | None:
        """
        Infer codebase path from file points when metadata is missing.

        Looks at abs_path of the first file point and extracts the root directory.
        """
        client = await self._get_client()

        try:
            # Get one file point to extract the abs_path
            points, _ = await client.scroll(
                collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="type", match=MatchValue(value="file")),
                    ],
                ),
                limit=1,
                with_payload=["abs_path", "path"],
            )

            if points and points[0].payload:
                abs_path = str(points[0].payload.get("abs_path", ""))
                rel_path = str(points[0].payload.get("path", ""))

                if abs_path and rel_path:
                    # Remove the relative path from absolute to get codebase root
                    # e.g., abs_path="/home/user/project/src/main.py", rel_path="src/main.py"
                    # -> codebase = "/home/user/project"
                    if abs_path.endswith(rel_path):
                        codebase: str = abs_path[:-len(rel_path)].rstrip("/")
                        return codebase

            return None
        except (KeyError, AttributeError, TypeError, UnexpectedResponse) as e:
            logger.debug(f"Could not extract codebase from point: {e}")
            return None
