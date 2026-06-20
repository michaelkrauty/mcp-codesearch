"""Collection management operations.

Tools:
- list_collections: List all indexed codebases
- delete_collection: Delete index for a codebase
- cleanup_orphans: Find and delete orphaned collections
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from mcp_codesearch.app import mcp
from mcp_codesearch.helpers import to_abs_path
from mcp_codesearch.singletons import (
    get_indexing_service,
    get_search_service,
    get_storage,
)
from mcp_codesearch.tools._errors import tool_error_handler

if TYPE_CHECKING:
    from mcp_codesearch.storage.qdrant import QdrantStorage

logger = logging.getLogger(__name__)

# Strict regex for collection ID validation (prevents injection attacks)
_COLLECTION_ID_PATTERN = re.compile(r"^codesearch_[a-f0-9]{12}$")


@mcp.tool()
@tool_error_handler
async def list_collections() -> str:
    """
    List all indexed codebases.

    Returns:
        List of collection names and their codebase paths
    """
    storage = await get_storage()
    collections = await storage.list_collections()

    if not collections:
        return "No codebases indexed yet."

    valid_count = 0
    orphan_count = 0
    missing_count = 0

    lines = ["Indexed codebases:"]
    for col_name in collections:
        metadata = await storage.get_metadata(col_name)
        path = metadata.get("codebase_path") if metadata else None

        if not path:
            path = await storage.infer_codebase_path(col_name)

        if path:
            if Path(path).exists():
                lines.append(f"  {col_name}: {path}")
                valid_count += 1
            else:
                lines.append(f"  {col_name}: {path} [MISSING - path deleted]")
                missing_count += 1
        else:
            lines.append(f"  {col_name}: [ORPHAN - unknown path]")
            orphan_count += 1

    if orphan_count > 0 or missing_count > 0:
        lines.append("")
        lines.append(
            f"Summary: {valid_count} valid, {missing_count} missing, {orphan_count} orphaned"
        )
        lines.append("Tip: Run cleanup_orphans() to remove stale collections.")

    return "\n".join(lines)


@mcp.tool()
@tool_error_handler
async def delete_collection(path: str = "", collection_id: str = "") -> str:
    """
    Delete index for a codebase.

    Args:
        path: Root path of codebase to remove from index
        collection_id: Direct collection ID (e.g., "codesearch_abc123") for orphan cleanup

    Returns:
        Confirmation message

    Note:
        Use collection_id to delete orphaned collections that show as "unknown" in list_collections.
        Either path or collection_id must be provided, but not both.
    """
    if path and collection_id:
        return "Error: Provide either 'path' or 'collection_id', not both."
    if not path and not collection_id:
        return "Error: Must provide either 'path' (codebase root) or 'collection_id' (for orphans)."

    indexing_svc = await get_indexing_service()
    search_svc = await get_search_service()

    if collection_id:
        if not _COLLECTION_ID_PATTERN.match(collection_id):
            return (
                f"Error: Invalid collection ID format. "
                f"Expected 'codesearch_<12 hex chars>' but got '{collection_id}'"
            )

        deleted = await indexing_svc.delete_by_collection_id(collection_id)
        if not deleted:
            return f"Collection not found: {collection_id}"
        return f"Deleted collection: {collection_id}"
    else:
        abs_path = to_abs_path(path)
        deleted = await indexing_svc.delete(abs_path)
        if not deleted:
            return f"No index found for: {abs_path}"

        # Invalidate search cache
        search_svc.invalidate_cache(abs_path)
        return f"Deleted index for: {abs_path}"


async def _classify_for_cleanup(storage: QdrantStorage, col_name: str) -> str:
    """Classify a collection for cleanup as "valid", "orphan", or "skip".

    Returns "orphan" only on positive confirmation that the codebase is gone:
    a path was determined and is confirmed absent on disk. Any uncertainty (an
    undeterminable path, a backend error reading it, or an inaccessible
    location) returns "skip", so a still-valid index is never deleted.
    """
    try:
        metadata = await storage.get_metadata(col_name)
        path = metadata.get("codebase_path") if metadata else None
        if not path:
            path = await storage.infer_codebase_path(col_name)
    except Exception as e:
        logger.warning(
            "cleanup_orphans: skipping %s, could not read its codebase path: %s",
            col_name, e,
        )
        return "skip"

    if not path:
        # No stored path and none inferable: we cannot confirm the codebase is
        # gone, so keep the collection rather than delete it blindly.
        return "skip"

    try:
        present = Path(path).exists()
    except OSError as e:
        # The path is on an inaccessible location (a stale NFS handle, a
        # permission error). We cannot confirm absence, so keep it.
        logger.warning(
            "cleanup_orphans: skipping %s, path %s is inaccessible: %s",
            col_name, path, e,
        )
        return "skip"

    return "valid" if present else "orphan"


@mcp.tool()
@tool_error_handler
async def cleanup_orphans() -> str:
    """
    Find and delete orphaned collections whose codebase directory is gone.

    A collection is deleted only when its codebase path is known and confirmed
    absent on disk. Collections whose path cannot be determined (no stored or
    inferable path, or a backend error reading it) or whose path is currently
    inaccessible (for example on an unmounted removable or network volume) are
    kept and reported as skipped, never deleted, so a temporarily unavailable
    codebase is not mistaken for a deleted one. Run this while any removable or
    network volumes holding indexed codebases are mounted.

    Returns:
        Summary of cleaned up, skipped, and remaining collections
    """
    storage = await get_storage()
    collections = await storage.list_collections()

    if not collections:
        return "No collections found."

    orphans: list[str] = []
    valid: list[str] = []
    skipped: list[str] = []
    buckets = {"valid": valid, "orphan": orphans, "skip": skipped}

    for col_name in collections:
        verdict = await _classify_for_cleanup(storage, col_name)
        buckets[verdict].append(col_name)

    if not orphans:
        msg = f"No orphaned collections found. {len(valid)} collection(s) have valid paths."
        if skipped:
            msg += (
                f" {len(skipped)} collection(s) skipped "
                "(path undeterminable or inaccessible; not deleted)."
            )
        return msg

    # Delete orphans using the service
    indexing_svc = await get_indexing_service()
    deleted = []
    failed = []

    for col_name in orphans:
        try:
            success = await indexing_svc.delete_by_collection_id(col_name)
            if success:
                deleted.append(col_name)
            else:
                failed.append(f"{col_name}: not found")
        except TimeoutError:
            failed.append(f"{col_name}: lock timeout (another process may be using it)")
        except Exception as e:
            failed.append(f"{col_name}: {e}")

    lines = [f"Cleaned up {len(deleted)} orphaned collection(s):"]
    for col in deleted:
        lines.append(f"  ✓ {col}")

    if failed:
        lines.append(f"\nFailed to delete {len(failed)} collection(s):")
        for msg in failed:
            lines.append(f"  ✗ {msg}")

    lines.append(f"\nRemaining valid collections: {len(valid)}")
    if skipped:
        lines.append(
            f"Skipped {len(skipped)} collection(s) "
            "(path undeterminable or inaccessible; not deleted)."
        )

    return "\n".join(lines)
