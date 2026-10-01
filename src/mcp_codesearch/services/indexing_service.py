"""Indexing service for code search.

Handles all indexing operations: full indexing, incremental updates,
vocabulary management, and batch processing.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from itertools import chain
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field
from qdrant_client.models import PointStruct
from vector_core import (
    EmbeddingClient,
    GlobalVocabulary,
    SparseVector,
    async_file_lock,
    cleanup_stale_locks,
    sparse_to_qdrant,
)
from vector_core.embeddings.identity import EmbeddingIdentity

from mcp_codesearch.indexer.change_detect import ChangeSet, detect_changes_fast
from mcp_codesearch.indexer.chunker import (
    build_chunk_vocabulary_text,
    chunk_file,
    generate_file_summary,
    truncate_chunk_content,
)
from mcp_codesearch.indexer.discovery import (
    FileInfo,
    discover_files,
)
from mcp_codesearch.indexer.journal import (
    IndexIntentStore,
    IntentOperation,
)
from mcp_codesearch.indexer.treesitter import Chunk
from mcp_codesearch.settings import settings
from mcp_codesearch.storage.qdrant import (
    EmbeddingDeploymentMismatchError,
    EmbeddingDimMismatchError,
    EmbeddingModelMismatchError,
    QdrantStorage,
)

logger = logging.getLogger(__name__)

# Batch size for memory-efficient streaming indexing
INDEXING_BATCH_SIZE = 50  # Files per batch
CONSISTENCY_LOCK_NAME = "codesearch_global_consistency"


def _canonical_qdrant_identity(url: str) -> str:
    """Normalize common aliases that address the same local Qdrant endpoint."""
    parsed = urlsplit(url.rstrip("/"))
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        host = "loopback"
    default_port = 443 if parsed.scheme.lower() == "https" else 80
    port = parsed.port or default_port
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.lower()}://{host}:{port}{path}"


async def _run_sync[**P, R](fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run blocking work off-loop without abandoning it on cancellation.

    Some callers mutate the shared vocabulary. Waiting for the worker before
    propagating cancellation preserves the old synchronous contract: no write
    can outlive the indexing request and race its rollback or lock release.
    """
    worker = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(worker)
            break
        except asyncio.CancelledError as exc:
            cancellation = exc
            if worker.done():
                result = worker.result()
                break
            # More than one cancellation can arrive while a server or task
            # group is shutting down. Every wait stays shielded until the
            # underlying thread has really finished.
    if cancellation is not None:
        raise cancellation
    return result


async def _finish_recovery(awaitable: Any) -> None:
    """Finish recovery despite repeated cancellation of the request task."""
    worker = asyncio.create_task(awaitable)
    while True:
        try:
            await asyncio.shield(worker)
            return
        except asyncio.CancelledError:
            if worker.done():
                # Distinguish cancellation of the recovery operation itself
                # from cancellation of only the caller waiting on shield.
                await worker
            continue


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


class VocabularyRepairStats(BaseModel):
    """Summary of a vocabulary consistency audit or repair."""

    collections_checked: int = 0
    mismatches_found: int = 0
    collections_repaired: int = 0
    stale_registrations: int = 0
    registrations_removed: int = 0
    pending_intents: int = 0
    intents_recovered: int = 0
    aggregate_frequencies_rebuilt: bool = False
    documents_before: int = 0
    documents_after: int = 0
    failures: list[str] = Field(default_factory=list)


class ConsistentReadSnapshot:
    """Generation plus optional work prepared under its vocabulary lock."""

    __slots__ = ("generation", "prepared")

    def __init__(self, generation: int, prepared: Any = None) -> None:
        self.generation = generation
        self.prepared = prepared


class PreparedFile(BaseModel):
    """A file prepared for indexing with its chunks and summary."""

    model_config = {"arbitrary_types_allowed": True}

    file_info: FileInfo
    chunks: list[Chunk]
    summary: str
    chunk_embedding_texts: list[str] = []  # Pre-computed embedding texts for chunks
    chunk_vocabulary_texts: list[str] = []


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
        journal: IndexIntentStore | None = None,
    ):
        self._storage = storage
        self._embedder = embedder
        self._global_vocab = global_vocab
        storage_identity = "\0".join(
            (
                settings.consistency_namespace or _canonical_qdrant_identity(str(storage.url)),
                str(Path(global_vocab.db_path).resolve()),
                str(storage.identity.fingerprint),
            )
        )
        self._consistency_scope = hashlib.sha256(storage_identity.encode()).hexdigest()[:16]
        self._consistency_lock_name = f"{CONSISTENCY_LOCK_NAME}_{self._consistency_scope}"
        self._admission_lock_name = f"{self._consistency_lock_name}_admission"
        self._consistency_timeout = max(60.0, settings.upsert_batch_timeout + 60.0)
        self._collection_timeout = max(3600.0, self._consistency_timeout)
        self._journal = journal or IndexIntentStore(namespace=self._consistency_scope)
        self._observed_generation: int | None = None
        self._stale_locks_cleaned = False
        self._stale_locks_lock = asyncio.Lock()

    @asynccontextmanager
    async def _admitted_lock(
        self,
        name: str,
        *,
        shared: bool,
        timeout: float,
    ) -> AsyncIterator[None]:
        """Queue behind one admission gate, then release it after lock acquisition."""
        async with AsyncExitStack() as held:
            async with async_file_lock(f"{name}_admission", timeout=timeout):
                await held.enter_async_context(
                    async_file_lock(name, timeout=timeout, shared=shared)
                )
            yield

    def _consistency_lock(self, *, shared: bool) -> AbstractAsyncContextManager[None]:
        """Acquire the global lock with writer-friendly admission ordering."""
        return self._admitted_lock(
            self._consistency_lock_name,
            shared=shared,
            timeout=self._consistency_timeout,
        )

    def _collection_lock(
        self,
        collection: str,
        *,
        shared: bool = False,
        timeout: float | None = None,
    ) -> AbstractAsyncContextManager[None]:
        """Acquire one collection lock without allowing new readers to starve writers."""
        return self._admitted_lock(
            collection,
            shared=shared,
            timeout=timeout or self._collection_timeout,
        )

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

    async def index(  # noqa: PLR0912, PLR0915
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
        col_name = self._storage.collection_name(abs_path)
        await self.recover_pending_intents(skip={col_name} if force else None)

        # Acquire cross-process lock for this collection
        async with self._collection_lock(col_name):
            # Close the gap between the pre-lock sweep and collection-lock
            # acquisition: another owner may have died and left an intent while
            # this request was waiting.
            if not force:
                async with self._consistency_lock(shared=False):
                    await self._recover_pending_intent(col_name)
            # Check if collection exists (inside lock to prevent TOCTOU)
            exists = await self._storage.collection_exists(col_name)
            target_intent = await _run_sync(self._journal.get, col_name)
            metadata = await self._storage.get_metadata(col_name) if exists else None
            resume_build = bool(
                target_intent is None and metadata and metadata.get("indexing_in_progress") is True
            )
            compatibility_verified = False
            if force and resume_build:
                try:
                    await self._verify_embedding_dim(col_name)
                    await self._verify_embedding_model(col_name, abs_path)
                    await self._verify_embedding_deployment(col_name, abs_path)
                    await self._verify_embedding_identity(col_name)
                    compatibility_verified = True
                except (
                    EmbeddingDeploymentMismatchError,
                    EmbeddingDimMismatchError,
                    EmbeddingModelMismatchError,
                ):
                    # force=True is the escape hatch. An interrupted build is
                    # resumable only in the same embedding space; otherwise
                    # discard it and start the requested rebuild from scratch.
                    resume_build = False

            if not exists or (force and not resume_build):
                if exists:
                    async with self._write_intent(
                        col_name,
                        abs_path,
                        "index",
                        recover_existing=target_intent is None,
                    ):
                        if not await self._safe_unregister_vocab(col_name):
                            raise RuntimeError(
                                f"Could not unregister vocabulary for {col_name}; "
                                "the existing collection was left untouched"
                            )
                        await self._storage.delete_collection(col_name)
                else:
                    registered = col_name in await _run_sync(self._global_vocab.get_codebase_ids)
                    if registered or target_intent is not None:
                        async with self._write_intent(
                            col_name,
                            abs_path,
                            "index",
                            recover_existing=target_intent is None,
                        ):
                            if registered and not await self._safe_unregister_vocab(col_name):
                                raise RuntimeError(
                                    f"Could not remove stale vocabulary for {col_name}"
                                )

                await self._storage.create_collection(col_name)
                await self._storage.store_metadata(
                    col_name,
                    abs_path,
                    indexing_in_progress=True,
                )
                files = await _run_sync(lambda: list(discover_files(codebase_path)))
                return await self._full_index(col_name, files, abs_path)
            else:
                # Reuse of an existing collection: make sure its stored vectors
                # are still compatible with the current embedding model before
                # we index into or search against it.
                if not compatibility_verified:
                    await self._verify_embedding_dim(col_name)
                    await self._verify_embedding_model(col_name, abs_path)
                    await self._verify_embedding_deployment(col_name, abs_path)
                    await self._verify_embedding_identity(col_name)
                await self._reconcile_if_count_mismatch(col_name)

                # Incremental index with fast change detection
                indexed_metadata = await self._storage.get_indexed_files_metadata(col_name)
                changes = await _run_sync(detect_changes_fast, codebase_path, indexed_metadata)

                if not changes.has_changes:
                    if resume_build:
                        await self._storage.store_metadata(
                            col_name,
                            abs_path,
                            indexing_in_progress=False,
                        )
                    return 0, 0, None

                result = await self._incremental_index(col_name, changes, abs_path)
                if resume_build:
                    await self._storage.store_metadata(
                        col_name,
                        abs_path,
                        indexing_in_progress=False,
                    )
                return result

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
        expected = self._storage.identity.dimension
        if not expected:
            return
        stored = await self._storage.get_dense_dim(col_name)
        if stored is not None and stored != expected:
            raise EmbeddingDimMismatchError(col_name, expected=expected, actual=stored)

    async def _verify_embedding_model(self, col_name: str, codebase_path: str) -> None:
        """Reject unknown provenance; retained legacy indexes are never stamped.

        Normal configuration changes select a different physical collection.
        This guard detects damaged or foreign metadata within the selected one.
        """
        expected = self._storage.identity.model
        metadata = await self._storage.get_metadata(col_name)
        stored = metadata.get("embedding_model") if metadata else None
        if not isinstance(stored, str) or stored != expected:
            raise EmbeddingModelMismatchError(col_name, expected=expected, actual=stored)

    async def _verify_embedding_deployment(
        self,
        col_name: str,
        codebase_path: str,
    ) -> None:
        """Guard same-name model revisions through the explicit cache namespace."""
        expected = self._storage.identity.namespace
        metadata = await self._storage.get_metadata(col_name)
        stored = metadata.get("embedding_cache_namespace") if metadata else None
        if stored != expected:
            raise EmbeddingDeploymentMismatchError(
                col_name,
                expected=expected,
                actual=stored,
            )

    async def _verify_embedding_identity(self, col_name: str) -> None:
        """Require complete provenance even when a collection has a scoped name."""
        metadata = await self._storage.get_metadata(col_name)
        payload = metadata.get("embedding_identity") if metadata else None
        expected = self._storage.identity
        try:
            if not isinstance(payload, dict) or set(payload) != set(expected.to_dict()):
                raise ValueError("Incomplete embedding identity")
            stored = EmbeddingIdentity.from_dict(payload)
        except (ValueError, TypeError):
            stored = None
        if stored != expected:
            raise EmbeddingDeploymentMismatchError(
                col_name,
                expected=expected.fingerprint,
                actual=stored.fingerprint if stored else "unknown",
            )

    async def get_status(self, codebase_path: str) -> dict[str, Any]:
        """
        Get indexing status for a codebase.

        Args:
            codebase_path: Path to the codebase root

        Returns:
            Status dict with file counts, pending changes, vocab stats
        """
        abs_path = str(Path(codebase_path).resolve())
        col_name = self._storage.collection_name(abs_path)
        await self.recover_pending_intents()

        async with self.consistent_read([col_name]):
            if not await self._storage.collection_exists(col_name):
                return {
                    "indexed": False,
                    "path": abs_path,
                    "embedding_identity": self._storage.identity.to_dict(),
                    "message": (
                        "Not indexed for this embedding identity. Run code_search to auto-index. "
                        "Retained generations are unchanged."
                    ),
                }

            indexed_metadata = await self._storage.get_indexed_files_metadata(col_name)
            changes = await _run_sync(detect_changes_fast, abs_path, indexed_metadata)
            metadata = await self._storage.get_metadata(col_name)
            updated = metadata.get("updated_at", "unknown") if metadata else "unknown"

            return {
                "indexed": True,
                "path": abs_path,
                "collection": col_name,
                "embedding_identity": self._storage.identity.to_dict(),
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

    async def _mark_intent(
        self,
        col_name: str,
        codebase_path: str,
        operation: IntentOperation,
    ) -> None:
        """Durably record an operation before either backing store changes."""
        await _run_sync(self._journal.mark, col_name, codebase_path, operation)

    async def _clear_intent(self, col_name: str) -> None:
        """Clear an intent only after Qdrant and vocabulary state agree."""
        await _run_sync(self._journal.clear, col_name)

    async def _set_pending_paths(self, col_name: str, paths: list[str]) -> None:
        """Record a batch immediately before its first Qdrant mutation."""
        await _run_sync(self._journal.set_pending_paths, col_name, paths)

    async def _clear_pending_paths(self, col_name: str) -> None:
        """Record that the current Qdrant batch completed in full."""
        await _run_sync(self._journal.clear_pending_paths, col_name)

    async def recover_pending_intents(self, *, skip: set[str] | None = None) -> int:
        """Recover writes whose process exited before its final journal clear."""
        recovered = 0
        for intent in await _run_sync(self._journal.list):
            if skip and intent.collection in skip:
                continue
            deadline = asyncio.get_running_loop().time() + self._consistency_timeout
            while await _run_sync(self._journal.get, intent.collection):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError(f"Timeout waiting to recover {intent.collection}")
                try:
                    async with self._collection_lock(
                        intent.collection,
                        timeout=min(0.5, remaining),
                    ):
                        async with self._consistency_lock(shared=False):
                            if await self._recover_pending_intent(intent.collection):
                                recovered += 1
                    break
                except TimeoutError:
                    # A live indexer holds the collection lock across embedding
                    # phases but exposes an intent only for its short commit.
                    # Once that intent disappears there is nothing to recover.
                    if not await _run_sync(self._journal.get, intent.collection):
                        break
        return recovered

    @asynccontextmanager
    async def _write_intent(
        self,
        col_name: str,
        codebase_path: str,
        operation: IntentOperation,
        *,
        recover_existing: bool = True,
    ) -> AsyncIterator[None]:
        """Run a cross-store mutation as one recoverable consistency window."""
        async with self._consistency_lock(shared=False):
            if recover_existing:
                await self._recover_pending_intent(col_name)
            await self._mark_intent(col_name, codebase_path, operation)
            try:
                yield
                # This is the commit record and therefore the final I/O before
                # releasing the global writer lock.
                await self._clear_intent(col_name)
            except BaseException:
                try:
                    await _finish_recovery(self._recover_pending_intent(col_name))
                except BaseException as recovery_error:
                    logger.error(
                        "Recovery failed for %s after an interrupted %s: %s",
                        col_name,
                        operation,
                        recovery_error,
                    )
                raise

    @asynccontextmanager
    async def consistent_read(
        self,
        collections: list[str] | None = None,
        prepare: Callable[[], Any] | None = None,
    ) -> AsyncIterator[ConsistentReadSnapshot]:
        """Prepare under a global snapshot, then retain only collection locks."""
        while True:
            retry = False
            generation = 0
            prepared: Any = None
            async with AsyncExitStack() as stack:
                for collection in sorted(set(collections or [])):
                    await stack.enter_async_context(self._collection_lock(collection, shared=True))
                async with self._consistency_lock(shared=True):
                    if await _run_sync(self._journal.list):
                        retry = True
                    else:
                        generation = await _run_sync(self._journal.generation)
                        if generation != self._observed_generation:
                            await _run_sync(self._global_vocab.invalidate_cache)
                            self._observed_generation = generation
                        prepared = await _run_sync(prepare) if prepare else None
                if not retry:
                    yield ConsistentReadSnapshot(generation, prepared)
                    return
            # A writer that died releases flock but leaves its intent. Repair it
            # under the exclusive lock, then retry the shared acquisition.
            await self.recover_pending_intents()

    async def _reconcile_collection_vocab(self, col_name: str) -> int:
        """Replace one vocabulary contribution from Qdrant's actual points.

        Token IDs remain append-only inside ``GlobalVocabulary``. Re-registering
        changes only this codebase's document frequencies and count, so existing
        sparse vectors keep referring to the same indices.
        """
        doc_frequencies: Counter[str] = Counter()
        doc_count = 0
        async for index_batch in self._storage.iter_stored_sparse_indices(col_name):
            token_sets = await self._token_sets_from_indices(
                index_batch,
                source=col_name,
            )
            for token_set in token_sets:
                doc_frequencies.update(token_set)
            doc_count += len(token_sets)

        await _run_sync(
            self._global_vocab.register_codebase_frequencies,
            col_name,
            doc_frequencies,
            doc_count,
        )
        return doc_count

    async def _token_sets_from_indices(
        self,
        documents: list[list[int]],
        *,
        source: str,
    ) -> list[set[str]]:
        """Map persisted sparse indices back to their append-only token names."""
        unique_indices = sorted({index for document in documents for index in document})
        try:
            index_to_token = await _run_sync(
                self._global_vocab.get_tokens_by_indices, unique_indices
            )
        except KeyError as exc:
            raise RuntimeError(
                f"{source} contains sparse indices absent from the vocabulary: {exc}"
            ) from exc
        return [{index_to_token[index] for index in document} for document in documents]

    async def _reconcile_if_count_mismatch(self, col_name: str) -> bool:
        """Repair legacy drift when registered and stored document counts differ."""
        stored = await self._storage.count_index_documents(col_name)
        registered = await _run_sync(self._global_vocab.get_codebase_doc_count, col_name)
        if stored == registered:
            return False
        async with self._consistency_lock(shared=False):
            stored = await self._storage.count_index_documents(col_name)
            registered = await _run_sync(
                self._global_vocab.get_codebase_doc_count,
                col_name,
            )
            if stored == registered:
                return False
            logger.warning(
                "Repairing vocabulary drift for %s: registered=%d, stored=%d",
                col_name,
                registered,
                stored,
            )
            # Publish invalidation before mutation. A crash after this point can
            # cause an extra cache miss, but can never hide a committed repair.
            await _run_sync(self._journal.bump_generation)
            await self._reconcile_collection_vocab(col_name)
            return True

    async def _recover_pending_intent(self, col_name: str) -> bool:
        """Finish or reconcile a write interrupted after its intent committed."""
        intent = await _run_sync(self._journal.get, col_name)
        if intent is None:
            return False

        exists = await self._storage.collection_exists(col_name)
        if intent.operation == "delete":
            if not await self._safe_unregister_vocab(col_name):
                raise RuntimeError(
                    f"Could not resume deletion of {col_name}; vocabulary is unchanged"
                )
            if exists:
                await self._storage.delete_collection(col_name)
        elif exists:
            # A process can die after only part of a Qdrant batch lands. Remove
            # the ambiguous paths so ordinary change detection re-adds each as
            # one complete file on this same indexing call.
            if intent.pending_paths:
                await self._storage.delete_by_paths_batch(col_name, list(intent.pending_paths))
            await self._reconcile_collection_vocab(col_name)
        elif not await self._safe_unregister_vocab(col_name):
            raise RuntimeError(
                f"Could not recover missing collection {col_name}; vocabulary is unchanged"
            )

        await self._clear_intent(col_name)
        logger.info("Recovered interrupted %s operation for %s", intent.operation, col_name)
        return True

    async def repair_vocabulary(  # noqa: PLR0912, PLR0915
        self,
        *,
        repair: bool = False,
        full: bool = False,
    ) -> VocabularyRepairStats:
        """Audit or reconstruct vocabulary state from authoritative Qdrant points.

        ``repair=False`` is read-only. A normal repair rebuilds collections whose
        registered document count differs from their stored file/chunk count and
        removes registrations with no collection. ``full=True`` re-registers
        every live collection, also repairing a same-count token-frequency drift.
        Collection locks make this safe alongside ordinary indexing; a busy or
        unreadable collection is reported and left unchanged.
        """
        stats = VocabularyRepairStats()
        stats.documents_before = await _run_sync(lambda: self._global_vocab.total_docs)

        if repair:
            stats.intents_recovered = await self.recover_pending_intents()

        collections = set(await self._storage.list_collections())
        registered = {
            codebase_id
            for codebase_id in await _run_sync(self._global_vocab.get_codebase_ids)
            if self._storage.owns_collection(codebase_id)
        }
        intents = {intent.collection: intent for intent in await _run_sync(self._journal.list)}
        stats.pending_intents = stats.intents_recovered + len(intents)

        for col_name in sorted(collections):
            try:
                async with self._collection_lock(col_name):
                    async with self._consistency_lock(shared=not repair):
                        if not await self._storage.collection_exists(col_name):
                            continue

                        stored = await self._storage.count_index_documents(col_name)
                        current = await _run_sync(
                            self._global_vocab.get_codebase_doc_count, col_name
                        )
                        stats.collections_checked += 1
                        mismatch = stored != current
                        if mismatch:
                            stats.mismatches_found += 1
                        if repair and (full or mismatch):
                            await _run_sync(self._journal.bump_generation)
                            await self._reconcile_collection_vocab(col_name)
                            stats.collections_repaired += 1
            except Exception as exc:
                logger.warning("Vocabulary repair skipped %s: %s", col_name, exc)
                stats.failures.append(f"{col_name}: {type(exc).__name__}")

        stale = registered - collections
        stats.stale_registrations = len(stale)
        for col_name in sorted(stale | (set(intents) - collections)):
            if not repair:
                continue
            try:
                async with self._collection_lock(col_name):
                    if await self._storage.collection_exists(col_name):
                        continue
                    async with self._write_intent(col_name, "", "delete"):
                        if not await self._safe_unregister_vocab(col_name):
                            raise RuntimeError("vocabulary unregistration failed")
                    if col_name in stale:
                        stats.registrations_removed += 1
            except Exception as exc:
                logger.warning("Stale vocabulary cleanup skipped %s: %s", col_name, exc)
                stats.failures.append(f"{col_name}: {type(exc).__name__}")

        if repair:
            stats.intents_recovered += await self.recover_pending_intents()
            async with self._consistency_lock(shared=False):
                await _run_sync(self._journal.bump_generation)
                await _run_sync(self._global_vocab.rebuild_aggregate_doc_frequencies)
                stats.aggregate_frequencies_rebuilt = True

        stats.documents_after = await _run_sync(lambda: self._global_vocab.total_docs)
        return stats

    async def _safe_unregister_vocab(self, col_name: str) -> bool:
        """Safely unregister a codebase from the vocabulary.

        Returns: True if succeeded, False if failed (vocab may be stale)
        """
        try:
            await _run_sync(self._global_vocab.unregister_codebase, col_name)
            return True
        except Exception as e:
            logger.warning(f"Failed to unregister vocabulary for {col_name}: {e}")
            return False

    async def delete(self, codebase_path: str) -> bool:
        """
        Delete index for a codebase.

        Args:
            codebase_path: Path to the codebase root

        Returns:
            True if deleted, False if not found
        """
        abs_path = str(Path(codebase_path).resolve())
        col_name = self._storage.collection_name(abs_path)

        async with self._collection_lock(col_name):
            exists = await self._storage.collection_exists(col_name)
            registered = col_name in await _run_sync(self._global_vocab.get_codebase_ids)
            if not exists and not registered:
                return False

            async with self._write_intent(
                col_name,
                abs_path,
                "delete",
                recover_existing=False,
            ):
                if not await self._safe_unregister_vocab(col_name):
                    raise RuntimeError(
                        f"Could not unregister vocabulary for {col_name}; "
                        "the collection was left untouched"
                    )
                if exists:
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
        if not self._storage.owns_collection(collection_id):
            raise ValueError(
                "Retained embedding generations are read-only; select their identity first"
            )
        async with self._collection_lock(collection_id):
            exists = await self._storage.collection_exists(collection_id)
            registered = collection_id in await _run_sync(self._global_vocab.get_codebase_ids)
            if not exists and not registered:
                return False

            metadata = await self._storage.get_metadata(collection_id) if exists else None
            path = metadata.get("codebase_path", "") if metadata else ""
            async with self._write_intent(
                collection_id,
                str(path),
                "delete",
                recover_existing=False,
            ):
                if not await self._safe_unregister_vocab(collection_id):
                    raise RuntimeError(
                        f"Could not unregister vocabulary for {collection_id}; "
                        "the collection was left untouched"
                    )
                if exists:
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

        Prepare files once, then embed and commit one recoverable batch at a time.
        """
        start_time = time.time()
        if not files:
            await self._storage.store_metadata(
                col_name,
                codebase_path,
                indexing_in_progress=False,
            )
            return 0, 0, IndexingStats(files_indexed=0, chunks_indexed=0, languages={})

        # Preparation is read-only. Vocabulary contribution is committed per
        # batch only after that batch's dense vectors are ready.
        prepared_files, tokens_per_doc = await _run_sync(self._prepare_files, files)

        total_chunks = 0
        languages: dict[str, int] = {}
        new_tokens = 0
        doc_offset = 0

        for batch_start in range(0, len(prepared_files), INDEXING_BATCH_SIZE):
            batch_end = min(batch_start + INDEXING_BATCH_SIZE, len(prepared_files))
            batch = prepared_files[batch_start:batch_end]
            doc_count = sum(1 + len(prepared.chunks) for prepared in batch)
            batch_added = tokens_per_doc[doc_offset : doc_offset + doc_count]
            doc_offset += doc_count

            chunk_count, batch_new_tokens = await self._process_batch(
                batch,
                col_name,
                codebase_path,
                languages,
                added_tokens=batch_added,
                removed_tokens=[],
                net_doc_change=len(batch_added),
            )
            total_chunks += chunk_count
            new_tokens += batch_new_tokens

        del tokens_per_doc

        # Store codebase path metadata
        await self._storage.store_metadata(
            col_name,
            codebase_path,
            indexing_in_progress=False,
        )

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

        # Read the outgoing files' tokens. Nothing is removed yet: each file's
        # points are dropped only once its replacement is ready to be written.
        removed_by_path = await self._collect_removed_tokens(col_name, changes)
        removed_tokens = list(chain.from_iterable(removed_by_path.values()))

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
            paths = list(removed_by_path)
            async with self._write_intent(col_name, codebase_path, "index"):
                await self._set_pending_paths(col_name, paths)
                await self._storage.delete_by_paths_batch(col_name, paths)
                await _run_sync(
                    self._global_vocab.update_codebase_incremental,
                    col_name,
                    added_tokens=[],
                    removed_tokens=removed_tokens,
                    net_doc_change=-len(removed_tokens),
                )
                await self._clear_pending_paths(col_name)
            stats = IndexingStats(
                files_indexed=0,
                chunks_indexed=0,
                languages={},
                was_incremental=True,
                files_deleted=len(changes.deleted),
            )
            return 0, 0, stats

        # Prepare file data and collect tokens for vocabulary update
        prepared_files, added_tokens = await _run_sync(self._prepare_files, files_to_index)

        total_chunks = 0
        languages: dict[str, int] = {}
        new_tokens = 0

        # A file that no longer exists has no replacement to wait for, so its
        # points and its share of the vocabulary go together, now.
        replacement_paths = {file_info.rel_path for file_info in changes.added + changes.modified}
        gone = [path for path in removed_by_path if path not in replacement_paths]
        if gone:
            gone_tokens = [t for p in gone for t in removed_by_path[p]]
            async with self._write_intent(col_name, codebase_path, "index"):
                await self._set_pending_paths(col_name, gone)
                await self._storage.delete_by_paths_batch(col_name, gone)
                await _run_sync(
                    self._global_vocab.update_codebase_incremental,
                    col_name,
                    added_tokens=[],
                    removed_tokens=gone_tokens,
                    net_doc_change=-len(gone_tokens),
                )
                await self._clear_pending_paths(col_name)

        # Each batch embeds first, then commits its vocabulary delta and Qdrant
        # replacement under one durable intent and the global writer lock.
        # Recovery reconstructs from the points that actually landed, so a
        # failure never unwinds earlier complete batches.
        doc_offset = 0
        for batch_start in range(0, len(prepared_files), INDEXING_BATCH_SIZE):
            batch_end = min(batch_start + INDEXING_BATCH_SIZE, len(prepared_files))
            batch = prepared_files[batch_start:batch_end]

            # tokens_per_doc holds one entry for a file's summary followed by one
            # per chunk, in file order, so a batch's share is a contiguous slice.
            doc_count = sum(1 + len(p.chunks) for p in batch)
            batch_added = added_tokens[doc_offset : doc_offset + doc_count]
            doc_offset += doc_count

            batch_stale = [
                p.file_info.rel_path for p in batch if p.file_info.rel_path in removed_by_path
            ]
            batch_removed = [t for p in batch_stale for t in removed_by_path[p]]

            chunk_count, batch_new_tokens = await self._process_batch(
                batch,
                col_name,
                codebase_path,
                languages,
                added_tokens=batch_added,
                removed_tokens=batch_removed,
                net_doc_change=len(batch_added) - len(batch_removed),
                stale_paths=batch_stale,
            )
            total_chunks += chunk_count
            new_tokens += batch_new_tokens

        # Clear token sets to free memory
        del added_tokens
        del removed_tokens

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
                chunk_vocabulary_texts = [
                    build_chunk_vocabulary_text(
                        truncate_chunk_content(chunk.content), chunk.imports
                    )
                    for chunk in chunks
                ]
            except Exception as e:
                # Chunking operates on arbitrary untrusted source; one pathological
                # file (malformed encoding, parser crash, etc.) must not abort the
                # whole indexing run. Log and skip.
                logger.warning(f"Failed to chunk {f.rel_path}: {type(e).__name__}: {e}")
                continue

            prepared_files.append(
                PreparedFile(
                    file_info=f,
                    chunks=chunks,
                    summary=summary,
                    chunk_embedding_texts=chunk_texts,
                    chunk_vocabulary_texts=chunk_vocabulary_texts,
                )
            )

            # Tokenize summary
            tokens_per_doc.append(set(self._global_vocab.tokenize(summary)))
            # Tokenize the canonical text persisted for each chunk
            for chunk_text in chunk_vocabulary_texts:
                tokens_per_doc.append(set(self._global_vocab.tokenize(chunk_text)))

        return prepared_files, tokens_per_doc

    async def _process_batch(
        self,
        batch: list[PreparedFile],
        col_name: str,
        codebase_path: str,
        languages: dict[str, int],
        *,
        added_tokens: list[set[str]],
        removed_tokens: list[set[str]],
        net_doc_change: int,
        stale_paths: list[str] | None = None,
    ) -> tuple[int, int]:
        """
        Process a batch of prepared files: generate embeddings and upsert to Qdrant.

        Args:
            batch: List of PreparedFile objects
            col_name: Collection name
            codebase_path: Canonical codebase root recorded in the intent
            languages: Dict to track language counts (mutated in place)
            added_tokens: New sparse token sets contributed by this batch
            removed_tokens: Sparse token sets superseded by this batch
            net_doc_change: Added document count minus removed document count
            stale_paths: Paths whose existing points this batch replaces. They
                are removed only once the new points are built and ready to be
                written, so that embedding -- by far the slowest and most
                failure-prone step here -- cannot leave a file with neither its
                old points nor its new ones.

        Returns:
            Tuple of chunks indexed and vocabulary tokens newly introduced
        """
        # Collect texts for this batch (using pre-computed chunk texts)
        batch_texts = []
        for prepared in batch:
            batch_texts.append(prepared.summary)
            batch_texts.extend(prepared.chunk_embedding_texts)

        # Generate embeddings for this batch
        dense_embeddings = await self._embedder.embed_all(batch_texts, role="document")

        # Dense inference is the slow and failure-prone step. It deliberately
        # finishes before the short global consistency window begins, so other
        # codebases remain searchable while this batch is being embedded.
        async with self._write_intent(col_name, codebase_path, "index"):
            new_tokens = await _run_sync(
                self._global_vocab.update_codebase_incremental,
                col_name,
                added_tokens=added_tokens,
                removed_tokens=removed_tokens,
                net_doc_change=net_doc_change,
            )

            # Sparse vectorization must follow the vocabulary update so every
            # newly introduced token already has its append-only index.
            points, chunk_count = await _run_sync(
                self._build_batch_points,
                batch,
                dense_embeddings,
                languages,
            )

            # This marker distinguishes a crash before Qdrant changed from an
            # ambiguous partial delete/upsert. Recovery clears these paths only
            # in the latter case, then rebuilds the contribution from Qdrant.
            paths = [prepared.file_info.rel_path for prepared in batch]
            await self._set_pending_paths(col_name, paths)
            if stale_paths:
                await self._storage.delete_by_paths_batch(col_name, stale_paths)
            chunk_points = [
                point for point in points if (point.payload or {}).get("type") == "chunk"
            ]
            file_points = [point for point in points if (point.payload or {}).get("type") == "file"]
            await self._storage.upsert_batch(col_name, chunk_points)
            # File points are fast-change-detection completion markers. A
            # separate final write preserves that ordering even when Qdrant
            # sub-batch concurrency is configured above one.
            await self._storage.upsert_batch(col_name, file_points)
            await self._clear_pending_paths(col_name)

        return chunk_count, new_tokens

    def _build_batch_points(
        self,
        batch: list[PreparedFile],
        dense_embeddings: list[list[float]],
        languages: dict[str, int],
    ) -> tuple[list[PointStruct], int]:
        """Build dense/sparse Qdrant points for one prepared batch."""
        points = []
        embed_idx = 0
        chunk_count = 0

        for prepared in batch:
            file_info = prepared.file_info
            languages[file_info.language] = languages.get(file_info.language, 0) + 1

            # File point
            file_dense_vec = dense_embeddings[embed_idx]
            file_sparse_vec = self._global_vocab.vectorize_document(prepared.summary)
            embed_idx += 1

            # Chunk points (using pre-computed chunk texts)
            for i, chunk in enumerate(prepared.chunks):
                dense_vec = dense_embeddings[embed_idx]
                sparse_vec = self._global_vocab.vectorize_document(
                    prepared.chunk_vocabulary_texts[i]
                )
                embed_idx += 1
                chunk_count += 1

                points.append(self._build_chunk_point(file_info, chunk, dense_vec, sparse_vec, i))

            # The file point is the completion marker used by fast change
            # detection. Write it after all of that file's chunks so a legacy
            # process crash cannot make a partial file appear complete.
            points.append(
                self._build_file_point(
                    file_info,
                    prepared.summary,
                    file_dense_vec,
                    file_sparse_vec,
                )
            )
        return points, chunk_count

    async def _collect_removed_tokens(
        self,
        col_name: str,
        changes: ChangeSet,
    ) -> dict[str, list[set[str]]]:
        """
        Collect exact sparse token sets for every removed or replacement path.

        Reads only. Removing a file's points is deferred to the point where its
        replacement is ready to take their place, so that a failure between here
        and there leaves the existing index serving queries.

        Keyed by path because the vocabulary delta is now committed per batch,
        and a batch may only account for the files it actually replaced.

        Returns:
            Mapping of path to the token sets of its stored documents
        """
        # Collect all paths to process
        # Added paths can still have orphan chunk points from an interrupted
        # legacy batch whose file point never landed. Treat every replacement
        # path as potentially outgoing so those chunks and their vocabulary
        # contribution are removed atomically with the complete replacement.
        all_paths = (
            list(changes.deleted)
            + [file_info.rel_path for file_info in changes.modified]
            + [file_info.rel_path for file_info in changes.added]
        )

        if not all_paths:
            return {}

        stored = await self._storage.get_stored_sparse_indices_by_paths(col_name, all_paths)
        by_path: dict[str, list[set[str]]] = {}
        for path, documents in stored.items():
            by_path[path] = await self._token_sets_from_indices(
                documents, source=f"Stored path {path!r}"
            )
        return by_path

    @staticmethod
    def _chunk_embedding_text(chunk: Chunk) -> str:
        """Build embedding text for a chunk, including imports if available."""
        return build_chunk_vocabulary_text(chunk.content, chunk.imports)

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
        ordinal: int,
    ) -> PointStruct:
        """Build a Qdrant point for a chunk-level entry.

        ``ordinal`` is the chunk's position within its file, and it is what
        keeps the point ID unique: several chunks of one file can share a start
        line. See ``QdrantStorage._point_id``.
        """
        return PointStruct(
            id=self._storage._point_id("chunk", file_info.rel_path, chunk.start_line, ordinal),
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
                "content": truncate_chunk_content(chunk.content),
                "context": chunk.context,
                "imports": chunk.imports or [],
            },
        )
