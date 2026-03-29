"""Similarity and reference search operations.

Tools:
- find_similar: Find code similar to a provided snippet
- find_references: Find all usages of a symbol
"""

from __future__ import annotations

import logging
from typing import Literal

from vector_core import (
    EmbeddingServiceError,
    validate_directory_path,
    validate_limit,
)
from vector_core.embeddings.client import CircuitBreakerOpenError
from vector_core.errors import format_error

from mcp_codesearch.app import mcp
from mcp_codesearch.helpers import auto_index
from mcp_codesearch.search.query import format_results
from mcp_codesearch.singletons import (
    get_embedder,
    get_global_vocab,
    get_storage,
)
from mcp_codesearch.storage.qdrant import collection_name

logger = logging.getLogger(__name__)


@mcp.tool()
async def find_similar(
    code: str,
    path: str = ".",
    limit: int = 10,
    language: str | None = None,
    exclude_self: bool = True,
    output_format: Literal["text", "json", "markdown"] = "text",
) -> str:
    """
    Find code similar to the provided snippet.

    Args:
        code: Code snippet to find similar code for
        path: Root path of codebase to search (defaults to current directory)
        limit: Max results to return (default 10)
        language: Filter by language (python, typescript, etc.)
        exclude_self: If True, excludes exact matches of the input code (default True)
        output_format: Output format - "text" (default), "json", or "markdown"

    Returns:
        Similar code snippets ranked by similarity score
    """
    if not code or not code.strip():
        return "Error: Code snippet cannot be empty."

    limit = validate_limit(limit, default=10)

    result = validate_directory_path(path)
    if isinstance(result, dict):
        return format_error(result)
    abs_path = str(result)
    col_name = collection_name(abs_path)

    # Auto-index if needed
    files_indexed, chunks_indexed, stats, error = await auto_index(abs_path)
    if error:
        return error

    index_msg = ""
    if files_indexed > 0 and stats:
        index_msg = f"[Indexed {files_indexed} files, {chunks_indexed} chunks]\n\n"

    # Embed the input code snippet
    storage = await get_storage()
    embedder = await get_embedder()
    global_vocab = await get_global_vocab()

    # Search for similar code with graceful degradation
    sparse_query = global_vocab.vectorize_query(code)
    fetch_limit = limit + (5 if exclude_self else 0)

    try:
        dense_query = await embedder.embed_single(code)
        # Full hybrid search with dense + sparse
        results = await storage.hybrid_search(
            collection=col_name,
            dense_query=dense_query,
            sparse_query=sparse_query,
            mode="chunk",
            language=language,
            limit=fetch_limit,
        )
    except (EmbeddingServiceError, CircuitBreakerOpenError) as e:
        # Embedding service unavailable - fall back to sparse-only search
        logger.warning(f"Embedding service unavailable, falling back to sparse-only search: {e}")
        results = await storage.sparse_only_search(
            collection=col_name,
            sparse_query=sparse_query,
            mode="chunk",
            language=language,
            limit=fetch_limit,
        )

    # Optionally exclude exact matches
    if exclude_self:
        code_normalized = code.strip().lower()
        results = [
            r for r in results
            if not r.content or r.content.strip().lower() != code_normalized
        ][:limit]

    formatted = format_results(results, output_format=output_format)
    return index_msg + formatted


@mcp.tool()
async def find_references(
    symbol: str,
    path: str = ".",
    limit: int = 20,
    include_definition: bool = False,
    output_format: Literal["text", "json", "markdown"] = "text",
) -> str:
    """
    Find all usages/references of a symbol (function, class, variable).

    Args:
        symbol: Name of the function, class, or variable to find references for
        path: Root path of codebase to search (defaults to current directory)
        limit: Max results to return (default 20)
        include_definition: If True, includes the symbol's definition in results
        output_format: Output format - "text" (default), "json", or "markdown"

    Returns:
        List of code locations where the symbol is referenced
    """
    if not symbol or not symbol.strip():
        return "Error: Symbol name cannot be empty."

    symbol = symbol.strip()
    limit = validate_limit(limit, default=20)

    result = validate_directory_path(path)
    if isinstance(result, dict):
        return format_error(result)
    abs_path = str(result)
    col_name = collection_name(abs_path)

    # Auto-index if needed
    files_indexed, chunks_indexed, stats, error = await auto_index(abs_path)
    if error:
        return error

    index_msg = ""
    if files_indexed > 0 and stats:
        index_msg = f"[Indexed {files_indexed} files, {chunks_indexed} chunks]\n\n"

    storage = await get_storage()

    # Use exact match search
    results = await storage.exact_match_search(
        collection=col_name,
        query=symbol,
        mode="chunk",
        limit=limit * 3,
    )

    # Filter results
    filtered_results = []
    symbol_lower = symbol.lower()

    for r in results:
        content_lower = (r.content or "").lower()
        name_lower = (r.name or "").lower()

        is_definition = name_lower == symbol_lower

        if is_definition and not include_definition:
            continue

        if symbol_lower in content_lower or symbol_lower in name_lower:
            if is_definition:
                r.chunk_type = f"[DEFINITION] {r.chunk_type or 'unknown'}"
            filtered_results.append(r)

        if len(filtered_results) >= limit:
            break

    if not filtered_results:
        return (
            index_msg + f"No references found for '{symbol}'.\n\n"
            "Tip: Check the symbol name spelling, or try searching with "
            "code_search for semantic matches."
        )

    header = f"Found {len(filtered_results)} reference(s) to '{symbol}':\n\n"
    formatted = format_results(filtered_results, output_format=output_format)
    return index_msg + header + formatted
