"""Core search operations.

Tools:
- code_search: Semantic code search with auto-indexing
- search_multiple: Search across multiple codebases
- search_changed: Search only in files changed since a commit
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from vector_core import (
    EmbeddingServiceError,
    validate_directory_path,
    validate_limit,
)
from vector_core.errors import format_error
from vector_core.search import reciprocal_rank_fusion

from mcp_codesearch.app import mcp
from mcp_codesearch.helpers import (
    auto_index,
    format_index_message,
    to_abs_path,
    validate_git_since,
)
from mcp_codesearch.search.query import format_results, search_codebase
from mcp_codesearch.services import SearchQuery
from mcp_codesearch.singletons import (
    get_embedder,
    get_global_vocab,
    get_search_service,
    get_storage,
)
from mcp_codesearch.tools._errors import tool_error_handler

if TYPE_CHECKING:
    from vector_core import EmbeddingClient, GlobalVocabulary
    from vector_core.search import RankFusionResult

    from mcp_codesearch.search.query import SearchResult
    from mcp_codesearch.services.search_service import SearchService
    from mcp_codesearch.storage.qdrant import QdrantStorage

logger = logging.getLogger(__name__)


@mcp.tool()
@tool_error_handler
async def code_search(  # noqa: PLR0911
    query: str,
    path: str = ".",
    mode: Literal["file", "chunk", "both"] = "both",
    limit: int = 10,
    language: str | None = None,
    path_prefix: str | None = None,
    exclude_paths: list[str] | None = None,
    output_format: Literal["text", "json", "markdown"] = "text",
) -> str:
    """
    Semantic code search. Auto-indexes on first use, incrementally updates thereafter.

    Args:
        query: Natural language description of what you're looking for.
               Supports special syntax:
               - function:name or fn:name - search for specific function
               - class:name or cls:name - search for specific class
               - struct:name - search for specific struct (Rust, C, Go)
               - path:prefix - filter to paths starting with prefix
               - -path:pattern - exclude paths containing pattern
        path: Root path of codebase (defaults to current directory)
        mode: "file", "chunk" (function/class level), or "both" (combined)
        limit: Max results to return (default 10)
        language: Filter by language (python, typescript, etc.)
        path_prefix: Only return results from paths starting with prefix (e.g., "src/")
        exclude_paths: Exclude paths containing these strings (e.g., ["test", "vendor"])
        output_format: Output format - "text", "json", or "markdown"

    Returns:
        Formatted search results with file paths and relevant code
    """
    # Validate query
    if not query or not query.strip():
        return """Error: Query cannot be empty.

Examples of valid queries:
  • Natural language: "websocket connection handling"
  • Function search: "function:handleRequest" or "fn:parse"
  • Class search: "class:UserService" or "cls:Config"
  • Struct search: "struct:Message" (Rust, C, Go)
  • Path filtering: "error handling path:src/" or "api -path:test"
  • Combined: "fn:validate path:src/auth -path:test"

Tip: Use natural language to describe what you're looking for semantically,
or use the special syntax above for targeted searches."""

    # Validate limit
    limit = validate_limit(limit, default=10)

    # Validate path
    result = validate_directory_path(path)
    if isinstance(result, dict):
        return format_error(result)
    abs_path = str(result)

    # Auto-index if needed
    files_indexed, chunks_indexed, stats, error = await auto_index(abs_path)
    if error:
        return error

    # Build index message and invalidate cache if needed
    index_msg = format_index_message(files_indexed, chunks_indexed, stats)

    # Use search service
    search_svc = await get_search_service()

    # Invalidate cache if any indexing activity occurred (including deletions)
    files_deleted = getattr(stats, "files_deleted", 0) if stats else 0
    index_changed = files_indexed > 0 or files_deleted > 0
    if index_changed:
        search_svc.invalidate_cache(abs_path)

    try:
        response = await search_svc.search(
            SearchQuery(
                query=query,
                path=abs_path,
                mode=mode,
                limit=limit,
                language=language,
                path_prefix=path_prefix,
                exclude_paths=exclude_paths,
                output_format=output_format,
            ),
            skip_cache=index_changed,
        )
    except EmbeddingServiceError as e:
        return f"""Error: Embedding service unavailable during search.

{e}

The codebase is indexed but semantic search requires the embedding service.
Consider using Grep/Glob for exact text searches until the service is restored."""

    return index_msg + response.to_output()


@mcp.tool()
@tool_error_handler
async def search_multiple(
    query: str,
    paths: list[str],
    mode: Literal["file", "chunk", "both"] = "both",
    limit: int = 10,
    language: str | None = None,
    output_format: Literal["text", "json", "markdown"] = "text",
    global_ranking: bool = False,
) -> str:
    """
    Search across multiple codebases concurrently.

    Each codebase is indexed (incrementally, when needed) and searched in parallel, so
    overall latency is bounded by the slowest codebase rather than the sum of them all.

    Args:
        query: Natural language description of what you're looking for
        paths: List of codebase paths to search (e.g., ["./repo1", "./repo2"])
        mode: "file" for file-level, "chunk" for function/class level, "both" for combined
        limit: Max results per codebase (also the cap on fused results when
            global_ranking is True)
        language: Filter by language (python, typescript, etc.)
        output_format: Output format - "text", "json", or "markdown"
        global_ranking: When False (default), results are grouped under one
            "=== path ===" section per codebase. When True, results from every codebase
            are merged into a single list ranked across codebases with Reciprocal Rank
            Fusion and tagged by their source codebase — answering "across all my repos,
            where is the best match?". RRF fuses by rank position, so it is robust to the
            fact that raw similarity scores from different collections are not directly
            comparable.

    Returns:
        Results grouped per codebase (default) or a single globally-ranked list.
    """
    if not query or not query.strip():
        return "Error: Query cannot be empty."

    limit = validate_limit(limit, default=10)

    if not paths:
        return "Error: paths list cannot be empty."

    # Validate all paths exist
    invalid_paths = []
    for path in paths:
        resolved = Path(path).resolve()
        if not resolved.exists():
            invalid_paths.append(f"{path} (does not exist)")
        elif not resolved.is_dir():
            invalid_paths.append(f"{path} (not a directory)")
    if invalid_paths:
        return "Error: Invalid paths:\n  • " + "\n  • ".join(invalid_paths)

    if global_ranking:
        return await _search_multiple_global(
            query, paths, mode, limit, language, output_format
        )
    return await _search_multiple_grouped(
        query, paths, mode, limit, language, output_format
    )


async def _grouped_section(
    path: str,
    query: str,
    mode: Literal["file", "chunk", "both"],
    limit: int,
    language: str | None,
    output_format: Literal["text", "json", "markdown"],
    search_svc: SearchService,
) -> str:
    """Index and search one codebase, returning its ``=== path ===`` section.

    Never raises: any failure is captured and rendered as an error section so that one
    failing codebase cannot abort the others when these run under ``asyncio.gather``.
    """
    abs_path = to_abs_path(path)
    try:
        files_indexed, chunks_indexed, stats, error = await auto_index(abs_path)
        if error:
            return f"=== {path} ===\n{error}\n"

        files_deleted = getattr(stats, "files_deleted", 0) if stats else 0
        index_changed = files_indexed > 0 or files_deleted > 0

        response = await search_svc.search(
            SearchQuery(
                query=query,
                path=abs_path,
                mode=mode,
                limit=limit,
                language=language,
                output_format=output_format,
            ),
            skip_cache=index_changed,
        )

        section = f"=== {path} ==="
        if index_changed and stats:
            section += (
                f" [Indexed {files_indexed} files, {chunks_indexed} chunks "
                f"in {stats.indexing_time_ms}ms]"
            )
        section += "\n"
        # ``formatted_output`` already reads "No results found." for an empty result set
        # and — unlike ``results_count`` — is populated for cache hits too, so it is the
        # source of truth for what to display (a cache hit leaves results_count at 0).
        section += response.formatted_output
        return section
    except Exception as e:
        # Log full details but don't leak to user (could contain sensitive paths)
        logger.error(f"Search failed for {path}: {type(e).__name__}: {e}")
        return f"=== {path} ===\nError: Search failed. Check server logs for details.\n"


async def _search_multiple_grouped(
    query: str,
    paths: list[str],
    mode: Literal["file", "chunk", "both"],
    limit: int,
    language: str | None,
    output_format: Literal["text", "json", "markdown"],
) -> str:
    """Search every codebase concurrently, grouping results per codebase (input order)."""
    search_svc = await get_search_service()
    sections = await asyncio.gather(
        *(
            _grouped_section(path, query, mode, limit, language, output_format, search_svc)
            for path in paths
        )
    )
    return "\n\n".join(sections)


@dataclass
class _SourcedResult:
    """A search result paired with the codebase it came from (for global ranking)."""

    source: str  # Codebase path exactly as supplied by the caller (for display)
    abs_source: str  # Resolved codebase root, used to derive a stable file identity
    result: SearchResult


def _global_key(sourced: _SourcedResult) -> tuple[str, str, int, int]:
    """Identity for de-duplicating a result across codebases.

    A nested repository indexed both on its own and as part of its parent yields the same
    file under two collections; keying on the absolute file location and span collapses
    those into a single fused entry.
    """
    r = sourced.result
    abs_file = str(Path(sourced.abs_source) / r.path)
    return (abs_file, r.point_type, r.start_line or -1, r.end_line or -1)


def _first_line(text: str) -> str:
    """First non-empty line of a (possibly multi-line) message, for compact summaries."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


async def _index_one(path: str) -> tuple[str, str, str]:
    """Auto-index one codebase, returning ``(path, abs_path, error_reason)``.

    ``error_reason`` is empty on success. Never raises (see :func:`_grouped_section`).
    """
    abs_path = to_abs_path(path)
    try:
        _files, _chunks, _stats, error = await auto_index(abs_path)
        return path, abs_path, _first_line(error) if error else ""
    except Exception as e:
        logger.error(f"Global index failed for {path}: {type(e).__name__}: {e}")
        return path, abs_path, "indexing failed (see server logs)"


async def _search_one(
    path: str,
    abs_path: str,
    query: str,
    mode: Literal["file", "chunk", "both"],
    limit: int,
    language: str | None,
    storage: QdrantStorage,
    embedder: EmbeddingClient,
    global_vocab: GlobalVocabulary,
) -> tuple[str, str, list[SearchResult], str]:
    """Search one already-indexed codebase for raw ranked results.

    Returns ``(path, abs_path, results, error_reason)``. Failures are logged in full and
    reduced to a generic reason so internal detail never reaches the caller.
    """
    try:
        results = await search_codebase(
            query=query,
            codebase_path=abs_path,
            storage=storage,
            embedder=embedder,
            global_vocab=global_vocab,
            mode=mode,
            language=language,
            limit=limit,
        )
        return path, abs_path, results, ""
    except EmbeddingServiceError as e:
        logger.error(f"Global search failed for {path}: embedding service: {e}")
        return path, abs_path, [], "embedding service unavailable"
    except Exception as e:
        logger.error(f"Global search failed for {path}: {type(e).__name__}: {e}")
        return path, abs_path, [], "search failed (see server logs)"


async def _search_multiple_global(
    query: str,
    paths: list[str],
    mode: Literal["file", "chunk", "both"],
    limit: int,
    language: str | None,
    output_format: Literal["text", "json", "markdown"],
) -> str:
    """Fuse results from every codebase into a single global ranking.

    Indexing and searching run in two phases on purpose: every codebase is indexed
    first, then all searches run against the now-settled GlobalVocabulary. The
    vocabulary supplies the IDF weights shared across collections, so scoring every
    codebase against the same snapshot keeps the cross-codebase ordering consistent
    rather than letting a fast codebase rank against a vocabulary another codebase is
    still updating.
    """
    storage = await get_storage()
    embedder = await get_embedder()
    global_vocab = await get_global_vocab()

    # Phase 1 — index everything (concurrent; the vocabulary's cross-process lock
    # serializes the actual writes).
    indexed = await asyncio.gather(*(_index_one(path) for path in paths))

    # Phase 2 — search the successfully-indexed codebases against the now-settled
    # vocabulary. ``searchable`` keeps each codebase's index into ``paths`` so the
    # outcomes can be reassembled in input order regardless of which phase failed.
    searchable = [(i, p, ap) for i, (p, ap, reason) in enumerate(indexed) if not reason]
    searched = await asyncio.gather(
        *(
            _search_one(p, ap, query, mode, limit, language, storage, embedder, global_vocab)
            for _i, p, ap in searchable
        )
    )
    search_by_index = {searchable[k][0]: searched[k] for k in range(len(searched))}

    # Reassemble in input order: errors and result lists both follow ``paths``.
    errors: list[tuple[str, str]] = []
    sourced_lists: list[list[_SourcedResult]] = []
    for i, (path, abs_path, reason) in enumerate(indexed):
        if reason:
            errors.append((path, reason))
            continue
        _p, _ap, results, search_error = search_by_index[i]
        if search_error:
            errors.append((path, search_error))
        elif results:
            sourced_lists.append(
                [_SourcedResult(source=path, abs_source=abs_path, result=r) for r in results]
            )

    fused = reciprocal_rank_fusion(sourced_lists, key=_global_key, limit=limit)
    return _format_global(fused, output_format, errors)


def _format_global(
    fused: list[RankFusionResult[_SourcedResult, tuple[str, str, int, int]]],
    output_format: str,
    errors: list[tuple[str, str]],
) -> str:
    """Render globally-ranked results, each tagged with its source codebase."""
    if output_format == "json":
        return _format_global_json(fused, errors)

    if not fused:
        body = "No results found."
    elif output_format == "markdown":
        body = _format_global_markdown(fused)
    else:
        body = _format_global_text(fused)

    if errors:
        skipped = "\n".join(f"  • {path}: {reason}" for path, reason in errors)
        body += f"\n\n[Skipped {len(errors)} codebase(s):\n{skipped}\n]"
    return body


def _format_global_json(
    fused: list[RankFusionResult[_SourcedResult, tuple[str, str, int, int]]],
    errors: list[tuple[str, str]],
) -> str:
    payload: dict[str, object] = {
        "results": [
            {
                "rank": rank,
                "source": f.item.source,
                "path": f.item.result.path,
                "score": round(f.item.result.score, 4),
                "rrf_score": round(f.score, 6),
                "type": f.item.result.point_type,
                "language": f.item.result.language,
                "name": f.item.result.name,
                "start_line": f.item.result.start_line,
                "end_line": f.item.result.end_line,
                "content": f.item.result.content,
            }
            for rank, f in enumerate(fused, 1)
        ]
    }
    if errors:
        payload["skipped"] = [{"path": path, "error": reason} for path, reason in errors]
    return json.dumps(payload, indent=2)


def _format_global_text(
    fused: list[RankFusionResult[_SourcedResult, tuple[str, str, int, int]]],
    summary_length: int = 150,
    content_length: int = 300,
) -> str:
    lines: list[str] = []
    for rank, f in enumerate(fused, 1):
        r = f.item.result
        src = f.item.source
        if r.point_type == "file":
            lines.append(f"{rank}. [{r.language}] ({src}) {r.path}")
            if r.summary:
                lines.append(f"   {r.summary[:summary_length]}...")
            if r.line_count is not None:
                lines.append(f"   ({r.line_count} lines)")
        else:
            name = r.name or "unnamed"
            lines.append(
                f"{rank}. [{r.language}] ({src}) {r.path}:{r.start_line}-{r.end_line}"
            )
            lines.append(f"   {r.chunk_type}: {name}")
            if r.content:
                preview = r.content[:content_length].replace("\n", " ")
                lines.append(f"   {preview}...")
        lines.append("")
    return "\n".join(lines)


def _format_global_markdown(
    fused: list[RankFusionResult[_SourcedResult, tuple[str, str, int, int]]],
    summary_length: int = 150,
    content_length: int = 300,
) -> str:
    lines: list[str] = []
    for rank, f in enumerate(fused, 1):
        r = f.item.result
        src = f.item.source
        if r.point_type == "file":
            lines.append(f"### {rank}. `{r.path}` [{r.language}] — _{src}_")
            if r.summary:
                lines.append(f"> {r.summary[:summary_length]}...")
            if r.line_count is not None:
                lines.append(f"*{r.line_count} lines*")
        else:
            name = r.name or "unnamed"
            lines.append(
                f"### {rank}. `{r.path}:{r.start_line}-{r.end_line}` [{r.language}] — _{src}_"
            )
            lines.append(f"**{r.chunk_type}**: `{name}`")
            if r.content:
                preview = r.content[:content_length].replace("\n", "\n> ")
                lines.append(f"```{r.language}\n{preview}\n```")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
@tool_error_handler
async def search_changed(  # noqa: PLR0911
    query: str,
    path: str = ".",
    since: str = "HEAD~10",
    limit: int = 10,
    output_format: Literal["text", "json", "markdown"] = "text",
) -> str:
    """
    Search only in files that have changed since a given commit or time.

    Note: this runs a normal ranked search over the whole codebase and
    intersects the top-ranked candidates with the set of changed files.
    A match inside a changed file that ranks below the candidate pool
    (top limit*20, capped at 200) will not appear. A true filter pushdown
    into the search layer is future work.

    Args:
        query: Natural language description of what you're looking for
        path: Root path of git repository (defaults to current directory)
        since: Git revision or time to compare against (e.g., "HEAD~10", "main", "3.days.ago")
        limit: Max results to return (default 10)
        output_format: Output format - "text" (default), "json", or "markdown"

    Returns:
        Search results filtered to changed files
    """
    if not query or not query.strip():
        return "Error: Query cannot be empty."

    limit = validate_limit(limit, default=10)

    path_result = validate_directory_path(path)
    if isinstance(path_result, dict):
        return format_error(path_result)
    abs_path = str(path_result)

    # Find git repository root (supports subdirectories of a repo)
    try:
        git_root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=False, cwd=abs_path,
            capture_output=True, text=True, timeout=10,
        )
        if git_root_result.returncode != 0:
            return f"Error: {path} is not within a git repository"
        git_root = git_root_result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return f"Error: {path} is not within a git repository (git not found or timed out)"

    # Validate since parameter (returns transformed value for .ago patterns)
    is_valid, validated_result = validate_git_since(since)
    if not is_valid:
        return validated_result

    # Get list of changed files from git
    try:
        # Use transformed value if provided, otherwise use original
        git_since = validated_result if validated_result else since

        git_result = subprocess.run(
            ["git", "diff", "--name-only", git_since],
            check=False, cwd=git_root,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if git_result.returncode != 0:
            git_result = subprocess.run(
                ["git", "log", "--since", git_since, "--name-only", "--pretty=format:"],
                check=False, cwd=git_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if git_result.returncode != 0:
                return (
                    f"Error: Git command failed. Invalid revision or time: '{since}'\n\n"
                    "Examples:\n  - HEAD~10 (last 10 commits)\n"
                    "  - main (since diverging from main)\n  - 3.days.ago (relative time)"
                )

        # Git returns paths relative to repo root. Convert to be relative to
        # the indexed codebase path so they match search result paths.
        rel_prefix = os.path.relpath(abs_path, git_root)
        raw_files = {f.strip() for f in git_result.stdout.strip().split("\n") if f.strip()}
        if rel_prefix == ".":
            # Indexed path IS the repo root — paths already match
            changed_files = raw_files
        else:
            # Indexed path is a subdirectory — strip the prefix
            prefix = rel_prefix + "/"
            changed_files = {
                f[len(prefix):] for f in raw_files if f.startswith(prefix)
            }

        if not changed_files:
            return f"No files changed since '{since}'."

    except subprocess.TimeoutExpired:
        return "Error: Git command timed out."
    except FileNotFoundError:
        return "Error: git command not found. Ensure git is installed and in PATH."

    # Auto-index if needed
    files_indexed, chunks_indexed, stats, error = await auto_index(abs_path)
    if error:
        return error

    index_msg = ""
    if files_indexed > 0 and stats:
        index_msg = f"[Indexed {files_indexed} files, {chunks_indexed} chunks]\n\n"

    # Search with higher limit to filter
    storage = await get_storage()
    embedder = await get_embedder()
    global_vocab = await get_global_vocab()

    # Fetch a large candidate pool: results are post-filtered by changed
    # paths, so matches ranking below this pool are invisible (see docstring).
    candidate_pool = min(limit * 20, 200)
    try:
        results = await search_codebase(
            query=query,
            codebase_path=abs_path,
            storage=storage,
            embedder=embedder,
            global_vocab=global_vocab,
            mode="both",
            limit=candidate_pool,
        )
    except EmbeddingServiceError as e:
        return (
            f"Error during search: {e}\n\n"
            "Ensure the embedding service is running and accessible."
        )

    # Filter to only changed files
    filtered_results = [r for r in results if r.path in changed_files][:limit]

    if not filtered_results:
        return (
            index_msg + f"No matches for this query among the top "
            f"{len(results)} search results within the {len(changed_files)} "
            f"files changed since '{since}'.\n\n"
            "A weaker match in a changed file may rank below this candidate "
            "pool. Try a narrower query, a different time range, or "
            "force_reindex if the index may be stale."
        )

    header = (
        f"Found {len(filtered_results)} result(s) in {len(changed_files)} "
        f"changed files (since '{since}'):\n\n"
    )
    formatted = format_results(filtered_results, output_format=output_format)
    return index_msg + header + formatted
