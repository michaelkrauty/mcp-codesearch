"""Collection management operations.

Tools:
- list_collections: List all indexed codebases
- delete_collection: Delete index for a codebase
- cleanup_orphans: Find and delete orphaned collections
"""

from __future__ import annotations

import re
from pathlib import Path

from mcp_codesearch.app import mcp

# Strict regex for collection ID validation (prevents injection attacks)
_COLLECTION_ID_PATTERN = re.compile(r"^codesearch_[a-f0-9]{12}$")
from mcp_codesearch.helpers import to_abs_path
from mcp_codesearch.singletons import (
    get_indexing_service,
    get_search_service,
    get_storage,
)


@mcp.tool()
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


@mcp.tool()
async def cleanup_orphans() -> str:
    """
    Find and delete orphaned collections (where codebase path is unknown/deleted).

    Returns:
        Summary of cleaned up collections
    """
    storage = await get_storage()
    collections = await storage.list_collections()

    if not collections:
        return "No collections found."

    orphans: list[str] = []
    valid: list[tuple[str, str]] = []

    for col_name in collections:
        metadata = await storage.get_metadata(col_name)
        path = metadata.get("codebase_path") if metadata else None

        if not path:
            path = await storage.infer_codebase_path(col_name)

        if path:
            if Path(path).exists():
                valid.append((col_name, path))
            else:
                orphans.append(col_name)
        else:
            orphans.append(col_name)

    if not orphans:
        return f"No orphaned collections found. All {len(valid)} collections have valid paths."

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

    return "\n".join(lines)
