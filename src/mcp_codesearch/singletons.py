"""Singleton instances for mcp-codesearch services.

Provides async-safe singleton patterns for:
- QdrantStorage
- EmbeddingClient
- GlobalVocabulary
- IndexingService
- SearchService
"""

from __future__ import annotations

from vector_core import AsyncSingleton, EmbeddingClient, GlobalVocabulary

from mcp_codesearch.services import IndexingService, SearchService
from mcp_codesearch.settings import settings
from mcp_codesearch.storage.qdrant import QdrantStorage

# Global singletons using AsyncSingleton pattern from vector-core
# Provides thread-safe initialization with retry support and proper cleanup
_storage: AsyncSingleton[QdrantStorage] = AsyncSingleton("storage")
_embedder: AsyncSingleton[EmbeddingClient] = AsyncSingleton("embedder")
_global_vocab: AsyncSingleton[GlobalVocabulary] = AsyncSingleton("global_vocab")
_indexing_service: AsyncSingleton[IndexingService] = AsyncSingleton("indexing_service")
_search_service: AsyncSingleton[SearchService] = AsyncSingleton("search_service")


async def get_storage() -> QdrantStorage:
    """Get or create QdrantStorage instance (async-safe via AsyncSingleton)."""
    return await _storage.get(QdrantStorage)


async def get_embedder() -> EmbeddingClient:
    """Get or create EmbeddingClient instance (async-safe via AsyncSingleton)."""
    return await _embedder.get(EmbeddingClient)


async def get_global_vocab() -> GlobalVocabulary:
    """Get the global vocabulary instance (async-safe via AsyncSingleton).

    Uses isolated vocabulary database for codesearch (separate from notes/docs).
    """
    def _create_vocab() -> GlobalVocabulary:
        codesearch_vocab_db = settings.cache_dir / "codesearch_vocabulary.db"
        return GlobalVocabulary(db_path=codesearch_vocab_db)

    return await _global_vocab.get(_create_vocab)


async def get_indexing_service() -> IndexingService:
    """Get or create IndexingService instance."""
    async def _create_service() -> IndexingService:
        storage = await get_storage()
        embedder = await get_embedder()
        vocab = await get_global_vocab()
        return IndexingService(storage, embedder, vocab)

    return await _indexing_service.get(_create_service)


async def get_search_service() -> SearchService:
    """Get or create SearchService instance."""
    async def _create_service() -> SearchService:
        storage = await get_storage()
        embedder = await get_embedder()
        vocab = await get_global_vocab()
        return SearchService(storage, embedder, vocab)

    return await _search_service.get(_create_service)


async def _safe_embedder_close(embedder: EmbeddingClient) -> None:
    """Safely close embedder, suppressing event loop errors during shutdown.

    httpx clients are tied to the event loop they were created in. If cleanup
    runs from a different loop (e.g., via asyncio.run() in atexit handler),
    the close will fail with "Event loop is closed". This is harmless - the
    client will be garbage collected anyway.
    """
    import logging
    _logger = logging.getLogger(__name__)

    try:
        if hasattr(embedder, 'close'):
            await embedder.close()
    except (RuntimeError, OSError, ValueError, AttributeError) as e:
        # Suppress expected shutdown errors
        error_str = str(e).lower()
        expected_patterns = ["event loop", "closing transport", "reset", "closed"]
        if not any(p in error_str for p in expected_patterns):
            _logger.warning(f"Embedder close error: {type(e).__name__}: {e}")


async def cleanup_resources() -> None:
    """Cleanup async resources on shutdown using AsyncSingleton's cleanup."""
    # Close services first (they don't own the resources)
    await _search_service.close(lambda s: s.clear_cache())
    await _indexing_service.close(lambda _: None)

    # Close singletons using AsyncSingleton's proper cleanup
    # Note: These may silently fail if called from a different event loop than
    # where the resources were created (e.g., during pytest cleanup). This is
    # expected - the resources will be garbage collected anyway.
    await _storage.close(lambda s: s.close())
    await _embedder.close(_safe_embedder_close)
    await _global_vocab.close(lambda v: v.close())
