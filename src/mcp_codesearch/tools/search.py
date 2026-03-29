"""Core search operations.

Tools:
- code_search: Semantic code search with auto-indexing
- search_multiple: Search across multiple codebases
- search_changed: Search only in files changed since a commit
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

from vector_core import (
    EmbeddingServiceError,
    validate_directory_path,
    validate_limit,
)
from vector_core.errors import format_error

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


@mcp.tool()
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
async def search_multiple(
    query: str,
    paths: list[str],
    mode: Literal["file", "chunk", "both"] = "both",
    limit: int = 10,
    language: str | None = None,
    output_format: Literal["text", "json", "markdown"] = "text",
) -> str:
    """
    Search across multiple codebases simultaneously.

    Args:
        query: Natural language description of what you're looking for
        paths: List of codebase paths to search (e.g., ["./repo1", "./repo2"])
        mode: "file" for file-level, "chunk" for function/class level, "both" for combined
        limit: Max results per codebase (default 10)
        language: Filter by language (python, typescript, etc.)
        output_format: Output format - "text", "json", or "markdown"

    Returns:
        Combined results from all codebases, grouped by codebase
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

    search_svc = await get_search_service()
    all_results = []

    for path in paths:
        abs_path = to_abs_path(path)

        try:
            # Auto-index if needed
            files_indexed, chunks_indexed, stats, error = await auto_index(abs_path)
            if error:
                all_results.append(f"=== {path} ===\n{error}\n")
                continue

            # Check if any indexing activity occurred (including deletions)
            files_deleted = getattr(stats, "files_deleted", 0) if stats else 0
            index_changed = files_indexed > 0 or files_deleted > 0

            # Search
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

            # Format results for this codebase
            section = f"=== {path} ==="
            if index_changed and stats:
                section += (
                    f" [Indexed {files_indexed} files, {chunks_indexed} chunks "
                    f"in {stats.indexing_time_ms}ms]"
                )
            section += "\n"

            if response.results_count > 0:
                section += response.formatted_output
            else:
                section += "No results found.\n"

        except Exception as e:
            # Log full details but don't leak to user (could contain sensitive paths)
            logger.error(f"Search failed for {path}: {type(e).__name__}: {e}")
            section = f"=== {path} ===\nError: Search failed. Check server logs for details.\n"

        all_results.append(section)

    return "\n\n".join(all_results)


@mcp.tool()
async def search_changed(  # noqa: PLR0911
    query: str,
    path: str = ".",
    since: str = "HEAD~10",
    limit: int = 10,
    output_format: Literal["text", "json", "markdown"] = "text",
) -> str:
    """
    Search only in files that have changed since a given commit or time.

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

    try:
        results = await search_codebase(
            query=query,
            codebase_path=abs_path,
            storage=storage,
            embedder=embedder,
            global_vocab=global_vocab,
            mode="both",
            limit=limit * 5,
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
            index_msg + f"No matches found in changed files "
            f"(searched {len(changed_files)} changed files since '{since}').\n\n"
            "Try broadening your query or checking a different time range."
        )

    header = (
        f"Found {len(filtered_results)} result(s) in {len(changed_files)} "
        f"changed files (since '{since}'):\n\n"
    )
    formatted = format_results(filtered_results, output_format=output_format)
    return index_msg + header + formatted
