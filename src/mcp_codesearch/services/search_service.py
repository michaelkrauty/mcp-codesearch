"""Search service for mcp-codesearch.

Handles search orchestration, caching, and result formatting.
The actual search algorithm is in mcp_codesearch.search.query.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from vector_core import CacheConfig, TTLCache

from mcp_codesearch.search.preprocess import preprocess_query
from mcp_codesearch.search.query import format_results, search_codebase
from mcp_codesearch.settings import settings
from mcp_codesearch.storage.qdrant import collection_name

if TYPE_CHECKING:
    from vector_core import EmbeddingClient, GlobalVocabulary

    from mcp_codesearch.search.query import SearchResult
    from mcp_codesearch.storage.qdrant import QdrantStorage

logger = logging.getLogger(__name__)


@dataclass
class SearchQuery:
    """Parameters for a search request."""

    query: str
    path: str
    mode: Literal["file", "chunk", "both"] = "both"
    limit: int = 10
    language: str | None = None
    path_prefix: str | None = None
    exclude_paths: list[str] | None = None
    output_format: Literal["text", "json", "markdown"] = "text"


@dataclass
class SearchResponse:
    """Response from a search operation."""

    # Formatted results
    formatted_output: str

    # Metadata
    was_cached: bool = False
    index_message: str = ""
    language_hint: str = ""
    results_count: int = 0

    # Raw results (optional, for programmatic access)
    raw_results: list[SearchResult] = field(default_factory=list)

    def to_output(self) -> str:
        """Build the complete output string."""
        parts = []
        if self.index_message:
            parts.append(self.index_message)
        if self.language_hint:
            parts.append(self.language_hint)
        parts.append(self.formatted_output)
        return "".join(parts)


class SearchService:
    """Orchestrates search operations with caching.

    This service wraps the search algorithm from mcp_codesearch.search.query
    and adds:
    - Result caching with TTL and LRU eviction
    - Cache invalidation on index updates
    - Language inference hints
    - Integration with auto-indexing
    """

    def __init__(
        self,
        storage: QdrantStorage,
        embedder: EmbeddingClient,
        global_vocab: GlobalVocabulary,
        cache_config: CacheConfig | None = None,
    ):
        """Initialize the search service.

        Args:
            storage: Qdrant storage instance
            embedder: Embedding client instance
            global_vocab: Global vocabulary for sparse vectors
            cache_config: Optional custom cache configuration
        """
        self._storage = storage
        self._embedder = embedder
        self._global_vocab = global_vocab

        # Initialize cache with provided or default config
        if cache_config is None:
            cache_config = CacheConfig(
                max_size=settings.search_cache_max_size,
                ttl_seconds=settings.search_cache_ttl_seconds,
                eviction_ratio=settings.search_cache_eviction_ratio,
            )
        self._cache: TTLCache[str] = TTLCache(cache_config)

    @property
    def cache(self) -> TTLCache[str]:
        """Expose cache for testing/inspection."""
        return self._cache

    async def search(
        self,
        query: SearchQuery,
        skip_cache: bool = False,
    ) -> SearchResponse:
        """Execute a search query.

        Args:
            query: The search query parameters
            skip_cache: If True, bypass cache check (e.g., after indexing)

        Returns:
            SearchResponse with formatted results and metadata
        """
        abs_path = str(Path(query.path).resolve())
        collection_name(abs_path)  # Validate/normalize path

        # Check cache first (unless skip_cache is True)
        cache_key = self._cache_key(query, abs_path)
        if not skip_cache:
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                # Add language hint even for cached results
                lang_hint = self._get_language_hint(query.query, query.language)
                return SearchResponse(
                    formatted_output=cached_result,
                    was_cached=True,
                    language_hint=lang_hint,
                )

        # Execute search
        results = await search_codebase(
            query=query.query,
            codebase_path=abs_path,
            storage=self._storage,
            embedder=self._embedder,
            global_vocab=self._global_vocab,
            mode=query.mode,
            language=query.language,
            limit=query.limit,
            path_prefix=query.path_prefix,
            exclude_paths=query.exclude_paths,
        )

        # Format results
        formatted = format_results(results, output_format=query.output_format)

        # Cache the result
        self._cache.set(cache_key, formatted)

        # Get language hint
        lang_hint = self._get_language_hint(query.query, query.language)

        return SearchResponse(
            formatted_output=formatted,
            was_cached=False,
            language_hint=lang_hint,
            results_count=len(results),
            raw_results=results,
        )

    def invalidate_cache(self, path: str) -> int:
        """Invalidate cache entries for a specific codebase path.

        Called after indexing operations to ensure fresh results.

        Args:
            path: The codebase path to invalidate

        Returns:
            Number of cache entries removed
        """
        abs_path = str(Path(path).resolve())
        removed = self._cache.invalidate_prefix(f"{abs_path}|")
        if removed > 0:
            logger.debug(f"Invalidated {removed} cache entries for {abs_path}")
        return removed

    def clear_cache(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def _cache_key(self, query: SearchQuery, abs_path: str) -> str:
        """Generate deterministic cache key from search parameters.

        The key includes the path as a prefix (before hash) to enable
        path-specific invalidation.
        """
        # Serialize exclude_paths consistently (sorted for determinism)
        exclude_str = "|".join(sorted(query.exclude_paths or []))
        key_parts = (
            f"{query.query}|{abs_path}|{query.mode}|"
            f"{query.language or ''}|{query.limit}|{query.output_format}|"
            f"{query.path_prefix or ''}|{exclude_str}"
        )
        key_hash = hashlib.sha256(key_parts.encode()).hexdigest()[:16]
        # Include path prefix for targeted invalidation
        return f"{abs_path}|{key_hash}"

    def _get_language_hint(
        self,
        query: str,
        language: str | None,
    ) -> str:
        """Generate language inference hint if applicable.

        Returns a hint message if the query suggests a specific language
        but no language filter was provided.
        """
        if language:
            # User already specified a language, no hint needed
            return ""

        _, parsed = preprocess_query(query, expand=False)
        if parsed.inferred_language:
            inf_lang = parsed.inferred_language
            return (
                f"[Tip: Query suggests {inf_lang} code. "
                f"Add language={inf_lang!r} for targeted results.]\n\n"
            )
        return ""
