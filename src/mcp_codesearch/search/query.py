"""Search query execution with intelligent query planning."""

from __future__ import annotations

import fnmatch
import json
import logging
from enum import Enum, auto
from pathlib import Path
from typing import Literal

__all__ = ["search_codebase", "format_results", "QueryType", "QueryPlan"]

from pydantic import BaseModel, ConfigDict
from vector_core import EmbeddingClient, GlobalVocabulary
from vector_core.embeddings.client import CircuitBreakerOpenError

from mcp_codesearch.search.preprocess import ParsedQuery, preprocess_query
from mcp_codesearch.settings import PATH_BOOST_MAX, PATH_BOOST_PATTERNS
from mcp_codesearch.storage.qdrant import (
    QdrantStorage,
    SearchResult,
    collection_name,
)

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query classification for routing optimization."""
    NAME_ONLY = auto()  # fn:X, class:X with no semantic text
    NAME_SEMANTIC = auto()  # fn:X + additional text to search
    SEMANTIC = auto()  # Pure natural language query
    EXACT_PHRASE = auto()  # Quoted exact string


class QueryPlan(BaseModel):
    """
    Query execution plan for optimized search routing.

    Different query types benefit from different search strategies:
    - NAME_ONLY: Skip embedding, just do exact name lookup
    - NAME_SEMANTIC: Name lookup first, then semantic search
    - SEMANTIC: Hybrid search with exact match fallback
    - EXACT_PHRASE: Prioritize exact match, skip semantic
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_type: QueryType
    search_text: str  # Text for embedding/semantic search
    name_filter: str | None  # Function/class name to filter on
    skip_embedding: bool  # Can skip expensive embedding generation
    use_exact_first: bool  # Prioritize exact match search


def _plan_query(query: str, parsed: ParsedQuery) -> QueryPlan:
    """
    Analyze query and create optimal execution plan.

    Routes queries to minimize unnecessary work:
    - "fn:handleRequest" → NAME_ONLY (skip embedding)
    - "fn:handleRequest websocket" → NAME_SEMANTIC (name lookup + semantic)
    - "error handling code" → SEMANTIC (full hybrid search)
    - '"exact phrase"' → EXACT_PHRASE (skip semantic)
    """
    has_name_filter = bool(parsed.function_name or parsed.class_name)
    has_semantic_text = bool(parsed.text and parsed.text.strip())

    # Detect quoted exact phrases
    is_quoted = query.strip().startswith('"') and query.strip().endswith('"')

    if is_quoted:
        # Exact phrase search - skip semantic, use exact match
        return QueryPlan(
            query_type=QueryType.EXACT_PHRASE,
            search_text=query.strip('" '),
            name_filter=None,
            skip_embedding=True,
            use_exact_first=True,
        )

    if has_name_filter and not has_semantic_text:
        # Pure name lookup - no need for embedding
        name = parsed.function_name or parsed.class_name
        return QueryPlan(
            query_type=QueryType.NAME_ONLY,
            search_text=name or "",  # Use name for any text matching
            name_filter=name,
            skip_embedding=True,
            use_exact_first=True,
        )

    if has_name_filter and has_semantic_text:
        # Name filter + semantic context
        return QueryPlan(
            query_type=QueryType.NAME_SEMANTIC,
            search_text=parsed.text or query,
            name_filter=parsed.function_name or parsed.class_name,
            skip_embedding=False,
            use_exact_first=True,  # Name lookup first
        )

    # Default: pure semantic search
    return QueryPlan(
        query_type=QueryType.SEMANTIC,
        search_text=parsed.text or query,
        name_filter=None,
        skip_embedding=False,
        use_exact_first=False,
    )


def _apply_path_boost(results: list[SearchResult]) -> list[SearchResult]:
    """Apply path-based score boosting to results.

    Uses additive adjustments (not multiplicative) to prevent score inversion.
    Total adjustment is capped at PATH_BOOST_MAX in either direction.

    Scores are NOT clamped to an upper bound here: exact-match search
    deliberately produces tiered scores above 1.0 (name=3.0, summary=2.0,
    content=1.0), and clamping would collapse those tiers into a tie before
    the sort below. _normalize_scores maps everything to [0, 1] afterwards.
    """
    for result in results:
        path_lower = result.path.lower()
        adjustment = 0.0

        for pattern, delta in PATH_BOOST_PATTERNS.items():
            if pattern in path_lower:
                adjustment += delta

        # Cap adjustment to prevent extreme score changes
        adjustment = max(-PATH_BOOST_MAX, min(PATH_BOOST_MAX, adjustment))

        # Apply additive adjustment, keeping scores non-negative
        result.score = max(0.0, result.score + adjustment)

    # Re-sort by adjusted score
    return sorted(results, key=lambda r: r.score, reverse=True)


def _normalize_scores(results: list[SearchResult]) -> list[SearchResult]:
    """
    Normalize all scores to 0-1 range.

    This ensures consistent score interpretation across different search methods:
    - RRF fusion typically produces scores in 0.2-0.5 range
    - Exact match produces scores in 1.0-3.0 range
    - Path boosting can further skew scores

    After normalization, scores are directly comparable and 1.0 = best match.
    """
    if not results:
        return results

    max_score = max(r.score for r in results)
    if max_score <= 0:
        return results

    for r in results:
        r.score = r.score / max_score

    return results


def _filter_by_parsed_query(  # noqa: PLR0912
    results: list[SearchResult],
    parsed: ParsedQuery,
) -> list[SearchResult]:
    """Filter results based on parsed query constraints.

    Applies the following filters when present in the parsed query:
    - path_prefix (path:X) - path component matching
    - exclude_paths (-path:X) - path substring exclusion
    - file_pattern (file:X) - filename glob matching (case-insensitive)
    - function_name/class_name (fn:X / class:X) - chunk name matching
    - scope (scope:X) - chunk type filtering
    """
    filtered = results

    # Filter by path prefix (matches path components anywhere in the path)
    # Examples:
    #   path:src         → matches "src/foo.py", "src/bar/baz.py"
    #   path:embeddings  → matches "src/mcp_codesearch/embeddings/foo.py"
    #   path:mcp         → matches "src/mcp_codesearch/foo.py" (substring of component)
    if parsed.path_prefix:
        prefix = parsed.path_prefix.strip('/')

        def matches_path(path: str) -> bool:
            # Fast exit if prefix not in path at all
            if prefix not in path:
                return False

            # Find component boundaries once (O(n) instead of O(n²))
            component_starts = [0] + [i + 1 for i, c in enumerate(path) if c == '/']

            for start in component_starts[:-1]:  # Exclude filename
                # Check if path from this component starts with prefix
                if path[start:].startswith(prefix) or path[start:].startswith(prefix + '/'):
                    return True
                # Find component end for substring match
                end = path.find('/', start)
                if end == -1:
                    end = len(path)
                # Substring match within single component
                if prefix in path[start:end]:
                    return True
            return False

        filtered = [r for r in filtered if matches_path(r.path)]

    # Exclude paths - use regex for large lists (O(n) vs O(n*m))
    if parsed.exclude_paths:
        if len(parsed.exclude_paths) <= 3:
            # Small list: simple iteration is fast enough
            filtered = [
                r for r in filtered
                if not any(exclude in r.path for exclude in parsed.exclude_paths)
            ]
        else:
            # Large list: use cached compiled regex for single-pass matching
            exclude_re = parsed.get_exclude_paths_regex()
            if exclude_re:
                filtered = [r for r in filtered if not exclude_re.search(r.path)]

    # Filter by filename pattern (file:db.py, file:*.sql)
    # Matches the filename component only, case-insensitive.
    # Patterns without wildcards match the filename exactly (fnmatch semantics).
    if parsed.file_pattern:
        pattern = parsed.file_pattern.lower()
        filtered = [
            r for r in filtered
            if fnmatch.fnmatch(r.path.rsplit('/', 1)[-1].lower(), pattern)
        ]

    # Filter by function/class name - strict filtering, not just boosting
    if parsed.function_name:
        target = parsed.function_name.lower()
        filtered = [r for r in filtered if r.name and target in r.name.lower()]
    if parsed.class_name:
        target = parsed.class_name.lower()
        filtered = [r for r in filtered if r.name and target in r.name.lower()]

    # Filter by scope (chunk type)
    if parsed.scope:
        if parsed.scope == "test":
            # Test scope: include only results from test files or with test-like names
            filtered = [
                r for r in filtered
                if "test" in r.path.lower()
                or (r.name and "test" in r.name.lower())
            ]
        elif parsed.scope == "impl":
            # Implementation scope: exclude test files
            filtered = [
                r for r in filtered
                if "test" not in r.path.lower()
                and (not r.name or "test" not in r.name.lower())
            ]
        elif parsed.scope == "function":
            # Only function/method chunks
            filtered = [
                r for r in filtered
                if r.chunk_type and r.chunk_type in ("function", "method")
            ]
        elif parsed.scope == "class":
            # Only class/struct/interface chunks
            class_types = ("class", "class_overview", "struct", "interface", "impl")
            filtered = [
                r for r in filtered
                if r.chunk_type and r.chunk_type in class_types
            ]

    return filtered


def _merge_results(
    primary: list[SearchResult],
    secondary: list[SearchResult],
) -> list[SearchResult]:
    """Merge two result lists, deduplicating by path+line."""
    seen: set[tuple[str, int | None]] = set()
    merged: list[SearchResult] = []

    for r in primary:
        key = (r.path, r.start_line)  # Tuples are hashable, no string ops
        if key not in seen:
            seen.add(key)
            merged.append(r)

    for r in secondary:
        key = (r.path, r.start_line)
        if key not in seen:
            seen.add(key)
            merged.append(r)

    return merged


async def search_codebase(
    query: str,
    codebase_path: str | Path,
    storage: QdrantStorage,
    embedder: EmbeddingClient,
    global_vocab: GlobalVocabulary,
    mode: Literal["file", "chunk", "both"] = "both",
    language: str | None = None,
    limit: int = 10,
    path_prefix: str | None = None,
    exclude_paths: list[str] | None = None,
) -> list[SearchResult]:
    """
    Search indexed codebase with intelligent query planning.

    The query planner optimizes search by routing different query types:
    - NAME_ONLY (fn:X): Skip embedding, use exact name lookup
    - NAME_SEMANTIC (fn:X + text): Name lookup first, then semantic
    - SEMANTIC (natural language): Hybrid search with fallback
    - EXACT_PHRASE ("quoted"): Skip semantic, use exact match

    Args:
        query: Natural language search query
        codebase_path: Root path of codebase
        storage: Qdrant storage instance
        embedder: embedding client instance
        global_vocab: Global vocabulary for sparse vectorization
        mode: Search mode (file/chunk/both)
        language: Optional language filter
        limit: Max results
        path_prefix: Only return results from paths starting with this prefix
        exclude_paths: Exclude results containing these path substrings

    Returns:
        List of SearchResult
    """
    abs_path = str(Path(codebase_path).resolve())
    col_name = collection_name(abs_path)

    # Preprocess query (synonyms + structured syntax parsing)
    processed_query, parsed = preprocess_query(query)

    # Merge explicit path params with parsed ones
    if path_prefix:
        parsed.path_prefix = path_prefix
    if exclude_paths:
        parsed.exclude_paths.extend(exclude_paths)

    # Create query execution plan
    plan = _plan_query(query, parsed)

    # Adjust fetch limit based on filtering needs
    if plan.name_filter or parsed.path_prefix or parsed.exclude_paths or parsed.file_pattern:
        fetch_limit = limit * 10  # Need more candidates for filtering
    else:
        fetch_limit = limit * 2

    results: list[SearchResult] = []

    # Execute based on query plan
    if plan.query_type == QueryType.NAME_ONLY:
        # Pure name lookup - skip embedding entirely
        results = await storage.exact_match_search(
            collection=col_name,
            query=plan.name_filter or "",
            mode=mode,
            language=language,
            limit=fetch_limit,
        )

    elif plan.query_type == QueryType.EXACT_PHRASE:
        # Exact phrase - skip semantic search
        results = await storage.exact_match_search(
            collection=col_name,
            query=plan.search_text,
            mode=mode,
            language=language,
            limit=fetch_limit,
        )

    elif plan.query_type == QueryType.NAME_SEMANTIC:
        # Name filter + semantic: do both and merge
        # 1. Exact name lookup first (high priority)
        name_results = await storage.exact_match_search(
            collection=col_name,
            query=plan.name_filter or "",
            mode=mode,
            language=language,
            limit=limit * 2,
        )

        # 2. Semantic search for context (with graceful degradation)
        try:
            dense_query = await embedder.embed_single_cached(processed_query or plan.search_text)
            sparse_query = global_vocab.vectorize_query(processed_query or plan.search_text)

            semantic_results = await storage.hybrid_search(
                collection=col_name,
                dense_query=dense_query,
                sparse_query=sparse_query,
                mode=mode,
                language=language,
                limit=fetch_limit,
            )
        except CircuitBreakerOpenError as e:
            # Embedding service unavailable - fall back to sparse-only search
            logger.warning(f"Embedding service unavailable, falling back to sparse-only search: {e}")
            sparse_query = global_vocab.vectorize_query(processed_query or plan.search_text)
            semantic_results = await storage.sparse_only_search(
                collection=col_name,
                sparse_query=sparse_query,
                mode=mode,
                language=language,
                limit=fetch_limit,
            )

        # Merge: name results first (higher priority)
        results = _merge_results(name_results, semantic_results)

    else:  # QueryType.SEMANTIC
        # Full semantic search with exact match fallback (with graceful degradation)
        try:
            dense_query = await embedder.embed_single_cached(processed_query or plan.search_text)
            sparse_query = global_vocab.vectorize_query(processed_query or plan.search_text)

            results = await storage.hybrid_search(
                collection=col_name,
                dense_query=dense_query,
                sparse_query=sparse_query,
                mode=mode,
                language=language,
                limit=fetch_limit,
            )
        except CircuitBreakerOpenError as e:
            # Embedding service unavailable - fall back to sparse-only search
            logger.warning(f"Embedding service unavailable, falling back to sparse-only search: {e}")
            sparse_query = global_vocab.vectorize_query(processed_query or plan.search_text)
            results = await storage.sparse_only_search(
                collection=col_name,
                sparse_query=sparse_query,
                mode=mode,
                language=language,
                limit=fetch_limit,
            )

        # Fallback to exact match if semantic scores are low
        EXACT_MATCH_THRESHOLD = 0.3
        if not results or (results and results[0].score < EXACT_MATCH_THRESHOLD):
            exact_results = await storage.exact_match_search(
                collection=col_name,
                query=parsed.text or query,
                mode=mode,
                language=language,
                limit=fetch_limit,
            )
            if exact_results:
                # Merge: exact matches first
                results = _merge_results(exact_results, results)

    # Apply path-based boosting
    results = _apply_path_boost(results)

    # Apply parsed query filters
    results = _filter_by_parsed_query(results, parsed)

    # Normalize scores to 0-1 range for consistent interpretation
    results = _normalize_scores(results[:limit])

    return results


def format_results(  # noqa: PLR0912
    results: list[SearchResult],
    output_format: str = "text",
    summary_length: int = 150,
    content_length: int = 300,
) -> str:
    """
    Format search results for display.

    Args:
        results: List of SearchResult objects
        output_format: "text" (default), "json", or "markdown"
        summary_length: Max chars for file summaries
        content_length: Max chars for chunk content previews
    """
    if not results:
        return "No results found." if output_format != "json" else "[]"

    if output_format == "json":
        return json.dumps([
            {
                "path": r.path,
                "score": round(r.score, 4),
                "type": r.point_type,
                "language": r.language,
                "summary": r.summary,
                "line_count": r.line_count,
                "chunk_type": r.chunk_type,
                "name": r.name,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "content": r.content,
                "degraded": r.degraded,
            }
            for r in results
        ], indent=2)

    if output_format == "markdown":
        lines = []
        for i, r in enumerate(results, 1):
            if r.point_type == "file":
                lines.append(f"### {i}. `{r.path}` [{r.language}]")
                if r.summary:
                    lines.append(f"> {r.summary[:summary_length]}...")
                lines.append(f"*{r.line_count} lines, score: {r.score:.3f}*")
            else:
                name = r.name or "unnamed"
                lines.append(f"### {i}. `{r.path}:{r.start_line}-{r.end_line}` [{r.language}]")
                lines.append(f"**{r.chunk_type}**: `{name}`")
                if r.content:
                    preview = r.content[:content_length].replace("\n", "\n> ")
                    lines.append(f"```{r.language}\n{preview}\n```")
                lines.append(f"*score: {r.score:.3f}*")
            lines.append("")
        return "\n".join(lines)

    # Default: text format
    lines = []
    for i, r in enumerate(results, 1):
        if r.point_type == "file":
            lines.append(f"{i}. [{r.language}] {r.path}")
            if r.summary:
                lines.append(f"   {r.summary[:summary_length]}...")
            lines.append(f"   ({r.line_count} lines, score: {r.score:.3f})")
        else:
            name = r.name or "unnamed"
            lines.append(f"{i}. [{r.language}] {r.path}:{r.start_line}-{r.end_line}")
            lines.append(f"   {r.chunk_type}: {name}")
            if r.content:
                preview = r.content[:content_length].replace("\n", " ")
                lines.append(f"   {preview}...")
            lines.append(f"   (score: {r.score:.3f})")
        lines.append("")

    return "\n".join(lines)
