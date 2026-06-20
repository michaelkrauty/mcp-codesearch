"""Unit tests for cleanup_orphans fail-safe classification (no live Qdrant).

cleanup_orphans must only delete a collection when it can positively confirm
the codebase is gone (a path was determined and is confirmed absent on disk).
Any uncertainty -- an undeterminable path, a transient backend error, or an
inaccessible/unmounted location -- must keep the collection, never delete it.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import mcp_codesearch.tools.collections as collections_mod
from mcp_codesearch.tools.collections import cleanup_orphans

GONE = "/nonexistent/definitely-gone-xyz"


def _wire(monkeypatch, *, names, get_metadata, infer=None):
    storage = MagicMock()
    storage.list_collections = AsyncMock(return_value=names)
    storage.get_metadata = get_metadata
    storage.infer_codebase_path = infer or AsyncMock(return_value=None)
    svc = MagicMock()
    svc.delete_by_collection_id = AsyncMock(return_value=True)
    monkeypatch.setattr(collections_mod, "get_storage", AsyncMock(return_value=storage))
    monkeypatch.setattr(
        collections_mod, "get_indexing_service", AsyncMock(return_value=svc)
    )
    return storage, svc


async def test_transient_path_read_error_skips_and_continues(monkeypatch):
    # First collection errors while reading its path; the second is a genuine
    # orphan. The error must not abort the whole run, and must not delete the
    # erroring collection.
    async def get_metadata(col):
        if col == "codesearch_err":
            raise RuntimeError("transient qdrant error")
        return {"codebase_path": GONE}

    _storage, svc = _wire(
        monkeypatch, names=["codesearch_err", "codesearch_orphan"],
        get_metadata=AsyncMock(side_effect=get_metadata),
    )
    result = await cleanup_orphans()
    # The genuine orphan is still deleted (run continued past the error)...
    svc.delete_by_collection_id.assert_awaited_once_with("codesearch_orphan")
    # ...and the erroring collection was skipped, not deleted.
    assert "skip" in result.lower()


async def test_undeterminable_path_is_skipped_not_deleted(monkeypatch):
    # No stored path and none inferable: keep it, do not delete.
    _storage, svc = _wire(
        monkeypatch, names=["codesearch_nopath"],
        get_metadata=AsyncMock(return_value=None),
        infer=AsyncMock(return_value=None),
    )
    result = await cleanup_orphans()
    svc.delete_by_collection_id.assert_not_awaited()
    assert "skip" in result.lower()


async def test_inaccessible_path_skips_and_continues(monkeypatch):
    real_exists = Path.exists

    def fake_exists(self):
        if str(self) == "/mnt/unmounted/proj":
            raise OSError("Stale file handle")
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)

    async def get_metadata(col):
        if col == "codesearch_mount":
            return {"codebase_path": "/mnt/unmounted/proj"}
        return {"codebase_path": GONE}

    _storage, svc = _wire(
        monkeypatch, names=["codesearch_mount", "codesearch_orphan"],
        get_metadata=AsyncMock(side_effect=get_metadata),
    )
    result = await cleanup_orphans()
    # genuine orphan still deleted; inaccessible-mount collection skipped
    svc.delete_by_collection_id.assert_awaited_once_with("codesearch_orphan")
    assert "skip" in result.lower()


async def test_confirmed_absent_path_is_deleted(monkeypatch):
    # Regression guard: a real orphan (path determined, confirmed absent) is
    # still cleaned up.
    _storage, svc = _wire(
        monkeypatch, names=["codesearch_orphan"],
        get_metadata=AsyncMock(return_value={"codebase_path": GONE}),
    )
    await cleanup_orphans()
    svc.delete_by_collection_id.assert_awaited_once_with("codesearch_orphan")


async def test_valid_path_is_kept(monkeypatch, tmp_path):
    # Regression guard: an existing codebase path is never deleted.
    _storage, svc = _wire(
        monkeypatch, names=["codesearch_valid"],
        get_metadata=AsyncMock(return_value={"codebase_path": str(tmp_path)}),
    )
    await cleanup_orphans()
    svc.delete_by_collection_id.assert_not_awaited()
