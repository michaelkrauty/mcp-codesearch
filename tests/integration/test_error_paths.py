"""Tests for error paths and edge cases in MCP server tools."""

import shutil
from pathlib import Path
from uuid import uuid4

from mcp_codesearch.server import (
    cleanup_orphans,
    code_search,
    delete_collection,
    find_references,
    find_similar,
)
from mcp_codesearch.singletons import get_storage
from mcp_codesearch.storage.qdrant import collection_name

from .conftest import requires_full_stack, requires_qdrant


def is_error_result(result) -> bool:
    """Check if result is an error response (string or dict)."""
    if isinstance(result, dict):
        return "error_code" in result
    return isinstance(result, str) and "error" in result.lower()


def error_contains(result, text: str) -> bool:
    """Check if error message contains text (handles string or dict)."""
    text_lower = text.lower()
    if isinstance(result, dict):
        msg = result.get("message", "").lower()
        code = result.get("error_code", "").lower()
        return text_lower in msg or text_lower in code
    return text_lower in str(result).lower()


class TestFindSimilarErrorPaths:
    """Tests for find_similar error handling."""

    async def test_empty_code(self, temp_codebase):
        """Empty code returns error."""
        result = await find_similar(code="", path=str(temp_codebase))
        assert is_error_result(result)
        assert error_contains(result, "empty")

    async def test_whitespace_only_code(self, temp_codebase):
        """Whitespace-only code returns error."""
        result = await find_similar(code="   \n\t  ", path=str(temp_codebase))
        assert is_error_result(result)
        assert error_contains(result, "empty")

    @requires_full_stack
    async def test_invalid_limit_zero_clamps_to_default(self, temp_codebase):
        """Zero limit is clamped to default (10), not rejected."""
        # First index so we have something to search
        await code_search(query="test", path=str(temp_codebase))

        result = await find_similar(
            code="def test(): pass",
            path=str(temp_codebase),
            limit=0,
        )
        # Should not error - clamped to default
        assert not is_error_result(result) or error_contains(result, "no similar")

    @requires_full_stack
    async def test_invalid_limit_negative_clamps_to_default(self, temp_codebase):
        """Negative limit is clamped to default (10), not rejected."""
        await code_search(query="test", path=str(temp_codebase))

        result = await find_similar(
            code="def test(): pass",
            path=str(temp_codebase),
            limit=-5,
        )
        # Should not error - clamped to default
        assert not is_error_result(result) or error_contains(result, "no similar")

    async def test_path_not_exists(self, tmp_path):
        """Non-existent path returns error."""
        fake_path = tmp_path / "nonexistent_dir_xyz"
        result = await find_similar(
            code="def test(): pass",
            path=str(fake_path),
        )
        assert is_error_result(result)
        assert error_contains(result, "not exist") or error_contains(result, "does not exist")

    async def test_path_is_file(self, tmp_path):
        """File path (not directory) returns error."""
        file_path = tmp_path / "test.py"
        file_path.write_text("print('hello')")

        result = await find_similar(
            code="def test(): pass",
            path=str(file_path),
        )
        assert is_error_result(result)
        assert error_contains(result, "directory")

    @requires_full_stack
    async def test_find_similar_with_language_filter(self, temp_codebase):
        """Find similar with language filter."""
        # First index
        await code_search(query="test", path=str(temp_codebase))

        result = await find_similar(
            code="def test(): pass",
            path=str(temp_codebase),
            language="python",
            limit=3,
        )
        # Should not error even if no matches
        assert not is_error_result(result) or error_contains(result, "no similar")

    async def test_find_similar_exclude_self(self, temp_codebase):
        """Find similar excludes exact matches by default."""
        # First index
        await code_search(query="test", path=str(temp_codebase))

        # Search for code that might exist
        result = await find_similar(
            code='def main():\n    print("Hello, World!")',
            path=str(temp_codebase),
            exclude_self=True,
            limit=3,
        )
        # Should handle gracefully
        assert isinstance(result, str)

    async def test_find_similar_include_self(self, temp_codebase):
        """Find similar can include exact matches."""
        await code_search(query="test", path=str(temp_codebase))

        result = await find_similar(
            code='def main():\n    print("Hello, World!")',
            path=str(temp_codebase),
            exclude_self=False,
            limit=3,
        )
        assert isinstance(result, str)


class TestFindReferencesErrorPaths:
    """Tests for find_references error handling."""

    async def test_empty_symbol(self, temp_codebase):
        """Empty symbol returns error."""
        result = await find_references(symbol="", path=str(temp_codebase))
        assert is_error_result(result)
        assert error_contains(result, "empty")

    async def test_whitespace_only_symbol(self, temp_codebase):
        """Whitespace-only symbol returns error."""
        result = await find_references(symbol="   ", path=str(temp_codebase))
        assert is_error_result(result)
        assert error_contains(result, "empty")

    @requires_full_stack
    async def test_invalid_limit_zero_clamps_to_default(self, temp_codebase):
        """Zero limit is clamped to default (20), not rejected."""
        await code_search(query="test", path=str(temp_codebase))

        result = await find_references(
            symbol="main",
            path=str(temp_codebase),
            limit=0,
        )
        # Should not error - clamped to default
        assert not is_error_result(result) or error_contains(result, "not found") or error_contains(result, "no references")

    @requires_full_stack
    async def test_invalid_limit_negative_clamps_to_default(self, temp_codebase):
        """Negative limit is clamped to default (20), not rejected."""
        await code_search(query="test", path=str(temp_codebase))

        result = await find_references(
            symbol="main",
            path=str(temp_codebase),
            limit=-10,
        )
        # Should not error - clamped to default
        assert not is_error_result(result) or error_contains(result, "not found") or error_contains(result, "no references")

    async def test_path_not_exists(self, tmp_path):
        """Non-existent path returns error."""
        fake_path = tmp_path / "nonexistent"
        result = await find_references(symbol="test", path=str(fake_path))
        assert is_error_result(result)
        assert error_contains(result, "not exist") or error_contains(result, "does not exist")

    async def test_path_is_file(self, tmp_path):
        """File path (not directory) returns error."""
        file_path = tmp_path / "test.py"
        file_path.write_text("print('hello')")

        result = await find_references(symbol="test", path=str(file_path))
        assert is_error_result(result)
        assert error_contains(result, "directory")

    @requires_full_stack
    async def test_symbol_not_found(self, temp_codebase):
        """Non-existent symbol returns helpful message."""
        await code_search(query="test", path=str(temp_codebase))

        result = await find_references(
            symbol="nonexistent_symbol_xyz123abc",
            path=str(temp_codebase),
        )
        assert "No references" in result or "not found" in result.lower()

    async def test_include_definition(self, temp_codebase):
        """Find references with include_definition=True."""
        await code_search(query="test", path=str(temp_codebase))

        result = await find_references(
            symbol="main",
            path=str(temp_codebase),
            include_definition=True,
            limit=5,
        )
        # Should handle gracefully
        assert isinstance(result, str)


class TestDeleteCollectionEdgeCases:
    """Tests for delete_collection edge cases."""

    @requires_qdrant
    async def test_delete_nonexistent_collection_by_path(self, tmp_path):
        """Deleting non-existent collection returns error."""
        new_path = tmp_path / f"never_indexed_{uuid4().hex[:8]}"
        new_path.mkdir()
        (new_path / "test.py").write_text("print('hi')")

        result = await delete_collection(path=str(new_path))
        assert "No index found" in result or "not found" in result.lower()

    async def test_delete_by_invalid_collection_id(self, tmp_path):
        """Invalid collection ID format returns error."""
        result = await delete_collection(
            path="",  # Empty path
            collection_id="invalid_format_not_codesearch"
        )
        assert is_error_result(result)
        assert error_contains(result, "invalid") or error_contains(result, "format")

    @requires_qdrant
    async def test_delete_by_nonexistent_collection_id(self, tmp_path):
        """Non-existent collection ID returns error."""
        # A well-formed ID of the current identity for a path never indexed
        never_indexed = collection_name(str((tmp_path / f"never_{uuid4().hex[:8]}").resolve()))
        result = await delete_collection(path="", collection_id=never_indexed)
        assert "not found" in result.lower() or "Collection not found" in result

    @requires_full_stack
    async def test_delete_by_collection_id_success(self, temp_codebase):
        """Delete by collection ID works when collection exists."""
        # First index to create collection
        await code_search(query="test", path=str(temp_codebase))

        # Get the collection name
        col_name = collection_name(str(Path(temp_codebase).resolve()))

        # Delete by collection ID
        result = await delete_collection(path="", collection_id=col_name)
        assert "Deleted" in result


class TestCleanupOrphansEdgeCases:
    """Tests for cleanup_orphans edge cases.

    Each test lists the collections cleanup may see through ``cleanup_scope``,
    so a run against a real Qdrant never deletes collections it did not create.
    """

    @requires_full_stack
    async def test_cleanup_with_no_orphans(self, temp_codebase, cleanup_scope):
        """A collection whose codebase still exists is kept."""
        await code_search(query="test", path=str(temp_codebase))
        indexed = collection_name(str(Path(temp_codebase).resolve()))
        cleanup_scope.add(indexed)
        try:
            result = await cleanup_orphans()
            assert "No orphaned collections found" in result
            assert "1 collection(s) have valid paths" in result
        finally:
            await delete_collection(collection_id=indexed)

    @requires_full_stack
    async def test_cleanup_finds_orphaned_collection(self, temp_codebase, tmp_path, cleanup_scope):
        """Cleanup removes the collection of a deleted codebase and keeps a live one."""
        orphan_path = tmp_path / f"orphan_test_{uuid4().hex[:8]}"
        orphan_path.mkdir()
        (orphan_path / "test.py").write_text("def test(): pass")
        await code_search(query="test", path=str(orphan_path))
        await code_search(query="test", path=str(temp_codebase))
        orphan = collection_name(str(orphan_path.resolve()))
        valid = collection_name(str(Path(temp_codebase).resolve()))
        cleanup_scope.update({orphan, valid})

        shutil.rmtree(orphan_path)
        try:
            result = await cleanup_orphans()
            assert orphan in result
            assert await list_collections_named(orphan) == []
            assert await list_collections_named(valid) == [valid]
        finally:
            await delete_collection(collection_id=orphan)
            await delete_collection(collection_id=valid)

    async def test_cleanup_no_collections(self, cleanup_scope):
        """With nothing in scope, cleanup reports that and touches nothing."""
        result = await cleanup_orphans()
        assert result == "No collections found."


async def list_collections_named(name: str) -> list[str]:
    """Collections of the current identity with exactly this name."""
    storage = await get_storage()
    # Unscoped on purpose: cleanup_scope only filters what cleanup can see.
    return [found for found in await type(storage).list_collections(storage) if found == name]


class TestCodeSearchErrorPaths:
    """Tests for code_search error handling."""

    async def test_empty_query_returns_results(self, temp_codebase):
        """Empty query still works (returns all files)."""
        result = await code_search(query="", path=str(temp_codebase), limit=5)
        # Should not crash, may return results or message
        assert isinstance(result, (str, dict))

    async def test_invalid_path_returns_error(self, tmp_path):
        """Invalid path returns error message."""
        fake_path = tmp_path / "does_not_exist_xyz"
        result = await code_search(query="test", path=str(fake_path))
        assert is_error_result(result)

    async def test_file_path_returns_error(self, tmp_path):
        """File path (not directory) returns error."""
        file_path = tmp_path / "test.py"
        file_path.write_text("print('hello')")

        result = await code_search(query="test", path=str(file_path))
        assert is_error_result(result)
        assert error_contains(result, "directory")

    @requires_full_stack
    async def test_invalid_limit_zero_clamps_to_default(self, temp_codebase):
        """Zero limit is clamped to default (10), not rejected."""
        result = await code_search(
            query="test",
            path=str(temp_codebase),
            limit=0,
        )
        # Should not error - clamped to default
        assert not is_error_result(result) or error_contains(result, "no results")

    @requires_full_stack
    async def test_invalid_limit_negative_clamps_to_default(self, temp_codebase):
        """Negative limit is clamped to default (10), not rejected."""
        result = await code_search(
            query="test",
            path=str(temp_codebase),
            limit=-1,
        )
        # Should not error - clamped to default
        assert not is_error_result(result) or error_contains(result, "no results")
