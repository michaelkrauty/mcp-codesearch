"""Index management operations.

Tools:
- index_status: Check indexing status for a codebase
- force_reindex: Force complete re-indexing of a codebase
- preview_index: Preview what would be indexed without actually indexing
"""

from __future__ import annotations

from pathlib import Path

from vector_core import (
    EmbeddingServiceError,
    format_error,
    validate_directory_path,
    validate_limit,
)

from mcp_codesearch.app import mcp
from mcp_codesearch.indexer.discovery import EXTENSION_TO_LANGUAGE, scan_file_metadata
from mcp_codesearch.singletons import get_indexing_service, get_search_service


@mcp.tool()
async def index_status(path: str = ".") -> str:
    """
    Check indexing status for a codebase.

    Args:
        path: Root path of codebase

    Returns:
        Status info: file count, last indexed, pending changes
    """
    result = validate_directory_path(path)
    if isinstance(result, dict):
        return format_error(result)

    indexing_svc = await get_indexing_service()
    status = await indexing_svc.get_status(str(result))

    if not status.get("indexed"):
        return f"Not indexed: {status['path']}\nRun code_search to auto-index."

    lines = [
        f"Codebase: {status['path']}",
        f"Collection: {status['collection']}",
        f"Indexed files: {status['files_indexed']}",
        f"Last updated: {status['last_updated']}",
        "",
        "Pending changes:",
        f"  Added: {status['pending_changes']['added']}",
        f"  Modified: {status['pending_changes']['modified']}",
        f"  Deleted: {status['pending_changes']['deleted']}",
        "",
        "Global vocabulary status:",
        f"  Total tokens: {status['vocabulary']['total_tokens']}",
        f"  Total docs across all codebases: {status['vocabulary']['total_docs']}",
        f"  Docs from this codebase: {status['vocabulary']['codebase_docs']}",
    ]

    return "\n".join(lines)


@mcp.tool()
async def force_reindex(path: str = ".") -> str:
    """
    Force complete re-indexing of a codebase.

    Args:
        path: Root path of codebase

    Returns:
        Indexing result summary
    """
    result = validate_directory_path(path)
    if isinstance(result, dict):
        return format_error(result)
    abs_path = str(result)

    try:
        indexing_svc = await get_indexing_service()
        files_indexed, chunks_indexed, stats = await indexing_svc.index(abs_path, force=True)
    except EmbeddingServiceError as e:
        return f"""Error: Embedding service unavailable.

{e}

Cannot re-index without the embedding service. Ensure your OpenAI-compatible embedding server is running."""

    # Invalidate search cache
    search_svc = await get_search_service()
    search_svc.invalidate_cache(abs_path)

    if stats:
        langs = sorted(stats.languages.items())
        lang_lines = "\n".join(f"    {lang}: {count} files" for lang, count in langs)
        return (
            f"Re-indexed {abs_path}:\n"
            f"  Files: {files_indexed}\n"
            f"  Chunks: {chunks_indexed}\n"
            f"  Time: {stats.indexing_time_ms}ms\n"
            f"  Languages:\n{lang_lines}"
        )
    return f"Re-indexed {abs_path}:\n  Files: {files_indexed}\n  Chunks: {chunks_indexed}"


@mcp.tool()
async def preview_index(
    path: str = ".",
    show_files: bool = False,
    limit: int = 50,
) -> str:
    """
    Preview what would be indexed without actually indexing.

    Args:
        path: Root path of codebase
        show_files: If True, list individual file paths
        limit: Max files to show when show_files=True

    Returns:
        Summary of what would be indexed
    """
    limit = validate_limit(limit, default=50)

    result = validate_directory_path(path)
    if isinstance(result, dict):
        return format_error(result)
    abs_path = str(result)

    # Discover files without reading content (fast)
    files_by_language: dict[str, list[str]] = {}
    total_size = 0
    file_count = 0

    for rel_path, _mtime, size in scan_file_metadata(abs_path):
        file_count += 1
        total_size += size

        ext = Path(rel_path).suffix.lower()
        lang = EXTENSION_TO_LANGUAGE.get(ext, "unknown")

        if lang not in files_by_language:
            files_by_language[lang] = []
        files_by_language[lang].append(rel_path)

    lines = [
        f"Preview for: {abs_path}",
        "",
        f"Total files: {file_count}",
        f"Total size: {total_size / 1024:.1f} KB",
        "",
        "Files by language:",
    ]

    for lang, files in sorted(files_by_language.items(), key=lambda x: len(x[1]), reverse=True):
        lines.append(f"  {lang}: {len(files)}")

    if show_files:
        lines.append("")
        lines.append(f"File list (first {limit}):")
        all_files = [f for files in files_by_language.values() for f in files]
        for f in sorted(all_files)[:limit]:
            lines.append(f"  {f}")
        if len(all_files) > limit:
            lines.append(f"  ... and {len(all_files) - limit} more")

    return "\n".join(lines)
