"""Indexing service for code search.

Handles all indexing operations: full indexing, incremental updates,
vocabulary management, and batch processing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from itertools import chain
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from qdrant_client.models import PointStruct
from vector_core import (
    EmbeddingClient,
    GlobalVocabulary,
    SparseVector,
    async_file_lock,
    cleanup_stale_locks,
    sparse_to_qdrant,
)

from mcp_codesearch.indexer.change_detect import ChangeSet, detect_changes_fast
from mcp_codesearch.indexer.chunker import chunk_file, generate_file_summary
from mcp_codesearch.indexer.discovery import (
    FileInfo,
    discover_files,
)
from mcp_codesearch.indexer.treesitter import Chunk
from mcp_codesearch.settings import settings
from mcp_codesearch.storage.qdrant import (
    EmbeddingDimMismatchError,
    EmbeddingModelMismatchError,
    QdrantStorage,
    collection_name,
)

logger = logging.getLogger(__name__)

# Batch size for memory-efficient streaming indexing
INDEXING_BATCH_SIZE = 50  # Files per batch


class IndexingStats(BaseModel):
    """Statistics from an indexing operation."""

    files_indexed: int
    chunks_indexed: int
    languages: dict[str, int]  # language -> file count
    indexing_time_ms: int = 0
    was_incremental: bool = False
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    new_tokens: int = 0  # New tokens added to global vocabulary

    def to_response(self) -> dict[str, int | bool | dict[str, int]]:
        """Convert to response dict for MCP tools."""
        return {
            "files_indexed": self.files_indexed,
            "chunks_indexed": self.chunks_indexed,
            "languages": self.languages,
            "indexing_time_ms": self.indexing_time_ms,
            "was_incremental": self.was_incremental,
            "files_added": self.files_added,
            "files_modified": self.files_modified,
            "files_deleted": self.files_deleted,
            "new_tokens": self.new_tokens,
        }


class PreparedFile(BaseModel):
    """A file prepared for indexing with its chunks and summary."""

    model_config = {"arbitrary_types_allowed": True}

    file_info: FileInfo
    chunks: list[Chunk]
    summary: str
    chunk_embedding_texts: list[str] = []  # Pre-computed embedding texts for chunks


class IndexingService:
    """Service for indexing codebases.

    Handles full and incremental indexing, vocabulary registration,
    and batch processing of files.
    """

    def __init__(
        self,
        storage: QdrantStorage,
        embedder: EmbeddingClient,
        global_vocab: GlobalVocabulary,
    ):
        self._storage = storage
        self._embedder = embedder
        self._global_vocab = global_vocab
        self._stale_locks_cleaned = False
        self._stale_locks_lock = asyncio.Lock()

    async def _ensure_stale_locks_cleaned(self) -> None:
        """One-time cleanup of stale lock files (thread-safe)."""
        if not self._stale_locks_cleaned:
            async with self._stale_locks_lock:
                # Double-check after acquiring lock
                if not self._stale_locks_cleaned:
                    self._stale_locks_cleaned = True
                    removed = cleanup_stale_locks()
                    if removed > 0:
                        logger.info(
                            f"Cleaned up {removed} stale lock file(s) from previous sessions"
                        )

    async def index(
        self,
        codebase_path: str,
        force: bool = False,
    ) -> tuple[int, int, IndexingStats | None]:
        """
        Index a codebase (full or incremental).

        Uses cross-process file locking to prevent race conditions when multiple
        Claude Code instances index the same codebase simultaneously.

        Args:
            codebase_path: Path to the codebase root
            force: If True, force full re-index even if collection exists

        Returns:
            Tuple of (files_indexed, chunks_indexed, stats)
        """
        await self._ensure_stale_locks_cleaned()

        abs_path = str(Path(codebase_path).resolve())
        col_name = collection_name(abs_path)

        # Acquire cross-process lock for this collection
        async with async_file_lock(col_name):
            # Check if collection exists (inside lock to prevent TOCTOU)
            exists = await self._storage.collection_exists(col_name)

            if not exists or force:
                # Full index. The existing collection is destroyed before the
                # rebuild, so a failure partway through must not leave a corrupt
                # state behind: the shared global vocabulary would otherwise keep
                # this codebase's full token contribution (skewing IDF for every
                # other indexed codebase) and a surviving partial collection
                # would make the next access run an *incremental* index that
                # double-counts those already-registered tokens. On any failure
                # roll back to a clean "not indexed" state instead.
                if exists:
                    # Unregister vocab FIRST, then delete collection
                    self._safe_unregister_vocab(col_name)
                    await self._storage.delete_collection(col_name)

                # Everything from collection (re)creation onward can fail and
                # leave a partial or empty collection plus a stale vocabulary
                # registration behind (file discovery can raise on a malformed
                # ignore pattern, the batch loop on a transient embed/Qdrant
                # error), so guard the whole rebuild and roll back on any failure.
                try:
                    await self._storage.create_collection(col_name)
                    files = list(discover_files(codebase_path))
                    return await self._full_index(col_name, files, abs_path)
                except Exception:
                    await self._rollback_failed_full_index(col_name)
                    raise
            else:
                # Reuse of an existing collection: make sure its stored vectors
                # are still compatible with the current embedding model before
                # we index into or search against it.
                await self._verify_embedding_dim(col_name)
                await self._verify_embedding_model(col_name, abs_path)

                # Incremental index with fast change detection
                indexed_metadata = await self._storage.get_indexed_files_metadata(col_name)
                changes = detect_changes_fast(codebase_path, indexed_metadata)

                if not changes.has_changes:
                    return 0, 0, None

                return await self._incremental_index(col_name, changes, abs_path)

    async def _verify_embedding_dim(self, col_name: str) -> None:
        """Refuse to reuse a collection whose dense vectors no longer match the
        configured embedding dimension.

        Detects the case where the embedding model was changed (to one with a
        different output dimension) after a codebase was indexed. Continuing
        would make Qdrant reject every upsert and dense query with a confusing
        dimension error, so we fail fast with an actionable message instead.

        The guard is deliberately query-agnostic: it gates *any* reuse of the
        collection — incremental indexing and search alike — not only dense
        queries. A dimension change leaves the whole collection unusable (new
        points can't even be upserted into it), so steering the user to
        ``force_reindex`` before any use is simpler and clearer than letting an
        exact-only lookup limp along on a half-broken index.

        Only a *definite* mismatch raises. This is a no-op when the expected
        dimension is unknown (``embedding_dim`` is still 0 because auto-detection
        has not resolved it) or when the stored dimension is absent from the
        collection config (``get_dense_dim`` returns ``None``); a genuine Qdrant
        read failure propagates to the caller's existing error handling.
        """
        expected = settings.embedding_dim
        if not expected:
            return
        stored = await self._storage.get_dense_dim(col_name)
        if stored is not None and stored != expected:
            raise EmbeddingDimMismatchError(col_name, expected=expected, actual=stored)

    async def _verify_embedding_model(self, col_name: str, codebase_path: str) -> None:
        """Refuse to reuse a collection whose vectors were embedded with a
        different model, even when the output dimension matches.

        The dimension guard above catches model swaps that change the vector
        size. This guard catches the silent case — same dimension, different
        model — where every Qdrant operation succeeds but query vectors and
        stored vectors come from incompatible embedding spaces, so searches
        quietly return meaningless results. The model name is recorded in
        collection metadata at index time and compared on every reuse.

        Fail-open by design, mirroring ``_verify_embedding_dim``: a missing
        configured model, missing collection metadata, or a stored value that
        is not a string (metadata predating this guard, or foreign collections)
        never blocks. Only a definite name mismatch raises.

        Collections indexed before the model was recorded are stamped with the
        current model on first reuse (backfill). Their true model is
        unknowable after the fact, and any subsequent incremental indexing
        embeds new chunks with the current model anyway, so recording the
        current model starts protection from now on without changing search
        behavior. The stamp is skipped while ``embedding_dim`` is unresolved
        because metadata writes need the dimension for their placeholder
        vector. Note the stamp refreshes the metadata ``updated_at`` (reported
        as ``last_updated`` by index_status) — a one-time cosmetic effect per
        legacy collection.
        """
        expected = settings.embedding_model
        if not expected:
            return
        metadata = await self._storage.get_metadata(col_name)
        stored = metadata.get("embedding_model") if metadata else None
        if stored is None:
            if settings.embedding_dim:
                await self._storage.store_metadata(col_name, codebase_path)
            return
        if isinstance(stored, str) and stored != expected:
            raise EmbeddingModelMismatchError(col_name, expected=expected, actual=stored)

    async def get_status(self, codebase_path: str) -> dict[str, Any]:
        """
        Get indexing status for a codebase.

        Args:
            codebase_path: Path to the codebase root

        Returns:
            Status dict with file counts, pending changes, vocab stats
        """
        abs_path = str(Path(codebase_path).resolve())
        col_name = collection_name(abs_path)

        if not await self._storage.collection_exists(col_name):
            return {
                "indexed": False,
                "path": abs_path,
                "message": "Not indexed. Run code_search to auto-index.",
            }

        indexed_metadata = await self._storage.get_indexed_files_metadata(col_name)
        changes = detect_changes_fast(abs_path, indexed_metadata)
        metadata = await self._storage.get_metadata(col_name)
        updated = metadata.get("updated_at", "unknown") if metadata else "unknown"

        return {
            "indexed": True,
            "path": abs_path,
            "collection": col_name,
            "files_indexed": len(indexed_metadata),
            "last_updated": updated,
            "pending_changes": {
                "added": len(changes.added),
                "modified": len(changes.modified),
                "deleted": len(changes.deleted),
            },
            "vocabulary": {
                "total_tokens": self._global_vocab.vocab_size,
                "total_docs": self._global_vocab.total_docs,
                "codebase_docs": self._global_vocab.get_codebase_doc_count(col_name),
            },
        }

    def _safe_unregister_vocab(self, col_name: str) -> bool:
        """Safely unregister a codebase from the vocabulary.

        Returns: True if succeeded, False if failed (vocab may be stale)
        """
        try:
            self._global_vocab.unregister_codebase(col_name)
            return True
        except Exception as e:
            logger.warning(f"Failed to unregister vocabulary for {col_name}: {e}")
            return False

    async def _rollback_failed_full_index(self, col_name: str) -> None:
        """Undo the partial side effects of a full index that failed midway.

        A full (re)index registers this codebase's token contribution with the
        shared global vocabulary in Phase 1, before any points are embedded and
        upserted in Phase 2. If Phase 2 raises, the contribution is left behind
        even though the collection holds few or no points, which skews IDF for
        every other codebase; and the partial collection, if kept, would steer
        the next access onto the incremental path, which adds the same tokens a
        second time. Removing both leaves a clean "not indexed" state so the
        next access does a fresh full index.

        Best-effort: each step is independent and its failure is logged, not
        raised, so the original indexing error is the one that propagates.
        """
        self._safe_unregister_vocab(col_name)
        try:
            await self._storage.delete_collection(col_name)
        except Exception as e:
            logger.warning(
                f"Failed to drop partial collection {col_name} after a failed "
                f"full index: {e}"
            )

    async def delete(self, codebase_path: str) -> bool:
        """
        Delete index for a codebase.

        Args:
            codebase_path: Path to the codebase root

        Returns:
            True if deleted, False if not found
        """
        abs_path = str(Path(codebase_path).resolve())
        col_name = collection_name(abs_path)

        async with async_file_lock(col_name):
            if not await self._storage.collection_exists(col_name):
                return False

            # Unregister vocab FIRST, then delete collection
            # This prevents orphaned vocabulary data if collection delete succeeds
            # but vocab unregister fails
            self._safe_unregister_vocab(col_name)
            await self._storage.delete_collection(col_name)
            return True

    async def delete_by_collection_id(self, collection_id: str) -> bool:
        """
        Delete a collection by its ID (for orphan cleanup).

        Args:
            collection_id: Collection ID (e.g., "codesearch_abc123")

        Returns:
            True if deleted, False if not found
        """
        async with async_file_lock(collection_id):
            if not await self._storage.collection_exists(collection_id):
                return False

            # Unregister vocab FIRST, then delete collection
            # This prevents orphaned vocabulary data if collection delete succeeds
            # but vocab unregister fails
            self._safe_unregister_vocab(collection_id)
            await self._storage.delete_collection(collection_id)
            return True

    # ============= Private Implementation =============

    async def _full_index(
        self,
        col_name: str,
        files: list[FileInfo],
        codebase_path: str,
    ) -> tuple[int, int, IndexingStats]:
        """
        Perform full indexing of codebase with memory-efficient batching.

        Phase 1: Scan all files and register tokens with global vocabulary
        Phase 2: Process files in batches for embedding and storage
        """
        start_time = time.time()
        if not files:
            return 0, 0, IndexingStats(files_indexed=0, chunks_indexed=0, languages={})

        # Phase 1: Prepare files and register tokens with global vocabulary
        prepared_files, tokens_per_doc = self._prepare_files(files)
        new_tokens = self._global_vocab.register_codebase(col_name, tokens_per_doc)
        del tokens_per_doc  # Free memory

        # Phase 2: Process files in batches for embedding and storage
        total_chunks = 0
        languages: dict[str, int] = {}

        for batch_start in range(0, len(prepared_files), INDEXING_BATCH_SIZE):
            batch_end = min(batch_start + INDEXING_BATCH_SIZE, len(prepared_files))
            batch = prepared_files[batch_start:batch_end]

            chunk_count = await self._process_batch(batch, col_name, languages)
            total_chunks += chunk_count

        # Store codebase path metadata
        await self._storage.store_metadata(col_name, codebase_path)

        elapsed_ms = int((time.time() - start_time) * 1000)
        stats = IndexingStats(
            files_indexed=len(files),
            chunks_indexed=total_chunks,
            languages=languages,
            indexing_time_ms=elapsed_ms,
            was_incremental=False,
            new_tokens=new_tokens,
        )

        return len(files), total_chunks, stats

    async def _incremental_index(
        self,
        col_name: str,
        changes: ChangeSet,
        codebase_path: str,
    ) -> tuple[int, int, IndexingStats]:
        """Perform incremental indexing with memory-efficient batching."""
        start_time = time.time()

        # Collect tokens from files being removed/modified BEFORE deleting
        removed_tokens = await self._collect_removed_tokens(col_name, changes)

        # Index new and modified files
        files_to_index = changes.added + changes.modified
        if not files_to_index and not removed_tokens:
            stats = IndexingStats(
                files_indexed=0,
                chunks_indexed=0,
                languages={},
                was_incremental=True,
                files_deleted=len(changes.deleted),
            )
            return 0, 0, stats

        # Handle case where only deletions occurred
        if not files_to_index:
            self._global_vocab.update_codebase_incremental(
                col_name,
                added_tokens=[],
                removed_tokens=removed_tokens,
                net_doc_change=-len(removed_tokens),
            )
            stats = IndexingStats(
                files_indexed=0,
                chunks_indexed=0,
                languages={},
                was_incremental=True,
                files_deleted=len(changes.deleted),
            )
            return 0, 0, stats

        # Prepare file data and collect tokens for vocabulary update
        prepared_files, added_tokens = self._prepare_files(files_to_index)

        # Update vocabulary: add new tokens, remove old tokens
        net_doc_change = len(added_tokens) - len(removed_tokens)
        new_tokens = self._global_vocab.update_codebase_incremental(
            col_name,
            added_tokens=added_tokens,
            removed_tokens=removed_tokens,
            net_doc_change=net_doc_change,
        )

        # Clear token sets to free memory
        del added_tokens
        del removed_tokens

        # Process files in batches for embedding and storage
        total_chunks = 0
        languages: dict[str, int] = {}

        for batch_start in range(0, len(prepared_files), INDEXING_BATCH_SIZE):
            batch_end = min(batch_start + INDEXING_BATCH_SIZE, len(prepared_files))
            batch = prepared_files[batch_start:batch_end]

            chunk_count = await self._process_batch(batch, col_name, languages)
            total_chunks += chunk_count

        elapsed_ms = int((time.time() - start_time) * 1000)
        stats = IndexingStats(
            files_indexed=len(files_to_index),
            chunks_indexed=total_chunks,
            languages=languages,
            indexing_time_ms=elapsed_ms,
            was_incremental=True,
            files_added=len(changes.added),
            files_modified=len(changes.modified),
            files_deleted=len(changes.deleted),
            new_tokens=new_tokens,
        )

        return len(files_to_index), total_chunks, stats

    def _prepare_files(
        self,
        files: list[FileInfo],
    ) -> tuple[list[PreparedFile], list[set[str]]]:
        """
        Prepare files for indexing by chunking and collecting tokens.

        Pre-computes chunk embedding texts to avoid redundant computation
        during batch processing (15-25% indexing speedup).

        Returns:
            Tuple of (prepared_files, tokens_per_doc)
        """
        prepared_files: list[PreparedFile] = []
        tokens_per_doc: list[set[str]] = []

        for f in files:
            try:
                chunks = chunk_file(f.content, f.language)
                summary = generate_file_summary(f.content, chunks, f.language)
                # Pre-compute chunk embedding texts (avoids recomputation in _process_batch)
                chunk_texts = [self._chunk_embedding_text(chunk) for chunk in chunks]
            except Exception as e:
                # Chunking operates on arbitrary untrusted source; one pathological
                # file (malformed encoding, parser crash, etc.) must not abort the
                # whole indexing run. Log and skip.
                logger.warning(
                    f"Failed to chunk {f.rel_path}: {type(e).__name__}: {e}"
                )
                continue

            prepared_files.append(PreparedFile(
                file_info=f,
                chunks=chunks,
                summary=summary,
                chunk_embedding_texts=chunk_texts,
            ))

            # Tokenize summary
            tokens_per_doc.append(set(self._global_vocab.tokenize(summary)))
            # Tokenize each chunk using pre-computed texts
            for chunk_text in chunk_texts:
                tokens_per_doc.append(set(self._global_vocab.tokenize(chunk_text)))

        return prepared_files, tokens_per_doc

    async def _process_batch(
        self,
        batch: list[PreparedFile],
        col_name: str,
        languages: dict[str, int],
    ) -> int:
        """
        Process a batch of prepared files: generate embeddings and upsert to Qdrant.

        Args:
            batch: List of PreparedFile objects
            col_name: Collection name
            languages: Dict to track language counts (mutated in place)

        Returns:
            Number of chunks indexed
        """
        # Collect texts for this batch (using pre-computed chunk texts)
        batch_texts = []
        for prepared in batch:
            batch_texts.append(prepared.summary)
            batch_texts.extend(prepared.chunk_embedding_texts)

        # Generate embeddings for this batch
        dense_embeddings = await self._embedder.embed_all(batch_texts)

        # Build points for this batch
        points = []
        embed_idx = 0
        chunk_count = 0

        for prepared in batch:
            file_info = prepared.file_info
            languages[file_info.language] = languages.get(file_info.language, 0) + 1

            # File point
            dense_vec = dense_embeddings[embed_idx]
            sparse_vec = self._global_vocab.vectorize_document(prepared.summary)
            embed_idx += 1

            points.append(
                self._build_file_point(file_info, prepared.summary, dense_vec, sparse_vec)
            )

            # Chunk points (using pre-computed chunk texts)
            for i, chunk in enumerate(prepared.chunks):
                dense_vec = dense_embeddings[embed_idx]
                sparse_vec = self._global_vocab.vectorize_document(
                    prepared.chunk_embedding_texts[i]
                )
                embed_idx += 1
                chunk_count += 1

                points.append(self._build_chunk_point(file_info, chunk, dense_vec, sparse_vec))

        # Upsert this batch to Qdrant
        await self._storage.upsert_batch(col_name, points)

        return chunk_count

    async def _collect_removed_tokens(
        self,
        col_name: str,
        changes: ChangeSet,
    ) -> list[set[str]]:
        """
        Collect tokens from files being deleted/modified and delete from Qdrant.

        This must happen BEFORE indexing new content to properly update vocabulary.
        Uses parallel I/O for content fetching, then batch deletion for efficiency.

        Returns:
            List of token sets from removed content
        """
        # Collect all paths to process
        all_paths = list(changes.deleted) + [f.rel_path for f in changes.modified]

        if not all_paths:
            return []

        # Phase 1: Collect content from all paths in parallel (for vocabulary update)
        # Semaphore limits concurrent Qdrant reads
        semaphore = asyncio.Semaphore(settings.deletion_concurrency)

        async def fetch_content(path: str) -> list[set[str]]:
            """Fetch stored content and tokenize it."""
            async with semaphore:
                stored_texts = await self._storage.get_stored_content_for_path(col_name, path)
                return [set(self._global_vocab.tokenize(text)) for text in stored_texts]

        results = await asyncio.gather(
            *[fetch_content(p) for p in all_paths], return_exceptions=True
        )

        # Filter failed results and log warnings
        valid_results: list[list[set[str]]] = []
        for path, result in zip(all_paths, results):
            if isinstance(result, BaseException):
                logger.warning(f"Failed to fetch content for {path}: {result}")
                valid_results.append([])  # Empty token set for failed file
            else:
                valid_results.append(result)

        # Phase 2: Batch delete all paths (much faster than individual deletes)
        await self._storage.delete_by_paths_batch(col_name, all_paths)

        # Flatten results into single list using itertools.chain
        return list(chain.from_iterable(valid_results))

    @staticmethod
    def _chunk_embedding_text(chunk: Chunk) -> str:
        """Build embedding text for a chunk, including imports if available."""
        text: str = chunk.content
        # Prepend imports for better semantic matching
        if chunk.imports:
            import_line = "Uses: " + ", ".join(chunk.imports)
            text = import_line + "\n\n" + text
        return text

    def _build_file_point(
        self,
        file_info: FileInfo,
        summary: str,
        dense_vec: list[float],
        sparse_vec: SparseVector,
    ) -> PointStruct:
        """Build a Qdrant point for a file-level entry."""
        return PointStruct(
            id=self._storage._point_id("file", file_info.rel_path),
            vector={
                "dense": dense_vec,
                "sparse": sparse_to_qdrant(sparse_vec),
            },
            payload={
                "type": "file",
                "path": file_info.rel_path,
                "abs_path": str(file_info.path),
                "language": file_info.language,
                "file_hash": file_info.content_hash,
                "summary": summary,
                "line_count": file_info.line_count,
                "size_bytes": file_info.size_bytes,
                "mtime": file_info.mtime,
            },
        )

    def _build_chunk_point(
        self,
        file_info: FileInfo,
        chunk: Chunk,
        dense_vec: list[float],
        sparse_vec: SparseVector,
    ) -> PointStruct:
        """Build a Qdrant point for a chunk-level entry."""
        return PointStruct(
            id=self._storage._point_id("chunk", file_info.rel_path, chunk.start_line),
            vector={
                "dense": dense_vec,
                "sparse": sparse_to_qdrant(sparse_vec),
            },
            payload={
                "type": "chunk",
                "path": file_info.rel_path,
                "abs_path": str(file_info.path),
                "language": file_info.language,
                "file_hash": file_info.content_hash,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content[: settings.max_payload_content_chars],
                "context": chunk.context,
            },
        )
