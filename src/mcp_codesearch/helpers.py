"""Shared helper functions for mcp-codesearch tools."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from vector_core import EmbeddingServiceError
from vector_core.storage.qdrant import QdrantConnectionError

from mcp_codesearch.singletons import get_embedder, get_indexing_service
from mcp_codesearch.storage.qdrant import (
    EmbeddingDimMismatchError,
    EmbeddingModelMismatchError,
)

if TYPE_CHECKING:
    from mcp_codesearch.services.indexing_service import IndexingStats

logger = logging.getLogger(__name__)


def validate_path_containment(file_path: Path, root_path: Path) -> bool:
    """Verify file_path is contained within root_path.

    Prevents path traversal attacks where relative paths like "../../../etc/passwd"
    could escape the intended directory.

    Args:
        file_path: Path to validate (can be relative or absolute)
        root_path: Root directory that should contain file_path

    Returns:
        True if file_path resolves to a location within root_path, False otherwise
    """
    try:
        file_path.resolve().relative_to(root_path.resolve())
        return True
    except ValueError:
        return False


def to_abs_path(path: str | Path) -> str:
    """Convert path to absolute string with symlinks resolved.

    This is a common pattern used throughout the codebase for normalizing
    paths before passing to storage/collection functions.
    """
    return str(Path(path).resolve())


# Pattern for validating git 'since' parameter
# Allows: HEAD~N, branch names, commit hashes, relative times (N.unit.ago)
_GIT_SINCE_PATTERN = re.compile(
    r"^("
    r"HEAD(~\d+|@\{\d+\})?|"  # HEAD, HEAD~10, HEAD@{1}
    r"[a-zA-Z][a-zA-Z0-9._/-]*|"  # Branch/tag names (must start with letter)
    r"[a-fA-F0-9]{4,40}|"  # Commit hashes (4-40 hex chars)
    r"\d+\.(second|minute|hour|day|week|month|year)s?\.ago"  # Relative time
    r")$"
)

# Pattern for valid .ago formats only (used for safe transformation)
_GIT_SINCE_AGO_PATTERN = re.compile(
    r"^\d+\.(second|minute|hour|day|week|month|year)s?\.ago$"
)


def validate_git_since(since: str) -> tuple[bool, str]:
    """Validate git 'since' parameter to prevent command injection.

    Returns:
        Tuple of (is_valid, result) where result is:
        - Empty string if valid and no transformation needed
        - Transformed value if valid and transformation applied (e.g., "3.days.ago" -> "3 days ago")
        - Error message if invalid
    """
    if not since or not since.strip():
        return False, "Error: 'since' parameter cannot be empty."

    since = since.strip()

    # Block dangerous patterns that could affect git behavior
    if since.startswith("-"):
        return False, "Error: 'since' parameter cannot start with '-' (looks like a git option)."

    if ".." in since and not since.endswith(".ago"):
        return False, "Error: Revision ranges ('..') are not supported. Use a single revision."

    # Validate against allowed patterns
    if not _GIT_SINCE_PATTERN.match(since):
        return False, (
            f"Error: Invalid 'since' format: '{since}'\n\n"
            "Allowed formats:\n"
            "  - HEAD~N (e.g., HEAD~10)\n"
            "  - Branch/tag name (e.g., main, v1.0)\n"
            "  - Commit hash (e.g., abc123)\n"
            "  - Relative time (e.g., 3.days.ago, 1.week.ago)"
        )

    # Transform .ago patterns and validate the result
    # This prevents "feature.branch.ago" from passing validation then
    # being transformed to ambiguous "feature branch ago"
    if since.endswith(".ago"):
        if not _GIT_SINCE_AGO_PATTERN.match(since):
            return False, f"Error: Invalid relative time format: '{since}'"
        # Return transformed value (e.g., "3.days.ago" -> "3 days ago")
        return True, since.replace(".", " ")

    return True, ""


def format_index_message(
    files_indexed: int,
    chunks_indexed: int,
    stats: IndexingStats | None,
) -> str:
    """Format index operation message."""
    if stats is None:
        return ""

    # Show message if any indexing activity occurred (add/modify/delete)
    files_deleted = getattr(stats, "files_deleted", 0)
    if files_indexed == 0 and files_deleted == 0:
        return ""

    langs = sorted(stats.languages.items())
    lang_summary = ", ".join(f"{lang}: {count}" for lang, count in langs) if langs else "none"

    if stats.was_incremental:
        return (
            f"[Updated: +{stats.files_added} added, ~{stats.files_modified} modified, "
            f"-{stats.files_deleted} deleted | {stats.chunks_indexed} chunks | "
            f"{lang_summary} | {stats.indexing_time_ms}ms]\n\n"
        )
    return (
        f"[Indexed {files_indexed} files, {chunks_indexed} chunks | "
        f"{lang_summary} | {stats.indexing_time_ms}ms]\n\n"
    )


async def auto_index(path: str) -> tuple[int, int, IndexingStats | None, str]:
    """Auto-index a codebase if needed.

    Returns:
        Tuple of (files_indexed, chunks_indexed, stats, error_message)
        If error_message is non-empty, indexing failed.
    """
    try:
        indexing_svc = await get_indexing_service()
        files, chunks, stats = await indexing_svc.index(path)
        return files, chunks, stats, ""
    except EmbeddingDimMismatchError as e:
        return 0, 0, None, f"""Error: this codebase was indexed with a different embedding model.

{e}.

The stored vectors can no longer be searched with the current model, so indexing
was skipped to avoid Qdrant errors and meaningless results. Rebuild this
codebase's index with the current model:

  force_reindex(path="{path}")

Only this codebase's collection is affected; other indexed codebases are left as-is."""
    except EmbeddingModelMismatchError as e:
        return 0, 0, None, f"""Error: this codebase was indexed with a different embedding model.

{e}.

The stored vectors share the current model's dimension, so searches would appear
to work — but query and stored vectors come from incompatible embedding spaces,
and the results would be meaningless. Indexing was skipped instead. Rebuild this
codebase's index with the current model:

  force_reindex(path="{path}")

Only this codebase's collection is affected; other indexed codebases are left as-is."""
    except QdrantConnectionError as e:
        return 0, 0, None, f"""Error: Qdrant vector database unavailable.

{e}

To fix this:
1. Ensure Qdrant is running (typically on port 6333)
2. Check VECTOR_QDRANT_URL environment variable if using non-default address
3. Try again once Qdrant is ready"""
    except EmbeddingServiceError as e:
        embedder = await get_embedder()
        return 0, 0, None, f"""Error: Embedding service unavailable.

{e}

To fix this:
1. Start an OpenAI-compatible embedding server
2. Ensure it's running on {embedder.base_url}
3. Try again once the service is ready"""
