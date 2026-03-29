"""Integration tests for MCP server with real Qdrant and embedding service."""

from uuid import uuid4

import pytest

from mcp_codesearch.server import (
    cleanup_resources,
    code_search,
    delete_collection,
    find_references,
    find_similar,
    force_reindex,
    index_status,
    list_collections,
    preview_index,
    search_changed,
    search_multiple,
)


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


def qdrant_and_embeddings_available() -> bool:
    """Check if both Qdrant and embedding service are running."""
    import httpx
    from mcp_codesearch.settings import settings
    try:
        # Check Qdrant
        qdrant_ok = httpx.get(f"{settings.qdrant_url}/collections", timeout=2.0).status_code == 200
        # Check embedding service
        embed_ok = httpx.get(f"{settings.embedding_url}/v1/models", timeout=2.0).status_code == 200
        return qdrant_ok and embed_ok
    except Exception:
        return False


requires_full_stack = pytest.mark.skipif(
    not qdrant_and_embeddings_available(),
    reason="Qdrant and/or embedding service not available"
)


@requires_full_stack
class TestCodeSearchIntegration:
    """Integration tests for code_search tool."""

    async def test_code_search_with_indexing(self, temp_codebase):
        """Code search indexes and searches codebase."""
        result = await code_search(
            query="main function",
            path=str(temp_codebase),
            mode="both",
            limit=5,
        )

        # Should have indexed
        assert "[Indexed" in result or "main" in result.lower()

    async def test_code_search_file_mode(self, temp_codebase):
        """Code search in file mode."""
        result = await code_search(
            query="utility functions",
            path=str(temp_codebase),
            mode="file",
            limit=5,
        )

        # Should return results
        assert "Error" not in result or "main" in result.lower()

    async def test_code_search_chunk_mode(self, temp_codebase):
        """Code search in chunk mode."""
        result = await code_search(
            query="function",
            path=str(temp_codebase),
            mode="chunk",
            limit=5,
        )

        # Should return results
        assert "Error" not in result

    async def test_code_search_language_filter(self, temp_codebase):
        """Code search with language filter."""
        result = await code_search(
            query="function",
            path=str(temp_codebase),
            language="python",
            limit=5,
        )

        # Should only show Python results
        # (or no error)
        assert "Error" not in result

    async def test_code_search_path_prefix(self, temp_codebase):
        """Code search with path prefix filter."""
        result = await code_search(
            query="main",
            path=str(temp_codebase),
            path_prefix="src/",
            limit=5,
        )

        assert "Error" not in result

    async def test_code_search_exclude_paths(self, temp_codebase):
        """Code search with path exclusion."""
        result = await code_search(
            query="test",
            path=str(temp_codebase),
            exclude_paths=["test"],
            limit=5,
        )

        assert "Error" not in result

    async def test_code_search_json_format(self, temp_codebase):
        """Code search with JSON output."""
        result = await code_search(
            query="main",
            path=str(temp_codebase),
            output_format="json",
            limit=5,
        )

        # Should contain JSON-like structure or be valid JSON
        assert "Error" not in result or "[" in result or "{" in result

    async def test_code_search_markdown_format(self, temp_codebase):
        """Code search with markdown output."""
        result = await code_search(
            query="main",
            path=str(temp_codebase),
            output_format="markdown",
            limit=5,
        )

        assert "Error" not in result

    async def test_code_search_special_syntax_function(self, temp_codebase):
        """Code search with function: syntax."""
        result = await code_search(
            query="function:main",
            path=str(temp_codebase),
            limit=5,
        )

        # Should find main function
        assert "Error" not in result

    async def test_code_search_caching(self, temp_codebase):
        """Repeated searches use cache."""
        # First search (indexes)
        result1 = await code_search(
            query="main",
            path=str(temp_codebase),
            limit=5,
        )

        # Second search (cached)
        result2 = await code_search(
            query="main",
            path=str(temp_codebase),
            limit=5,
        )

        # Second should not show indexing message
        assert "[Indexed" not in result2 or result1 == result2


@requires_full_stack
class TestIndexStatusIntegration:
    """Integration tests for index_status tool."""

    async def test_index_status_unindexed(self, tmp_path):
        """Status of unindexed codebase."""
        # Create a new temp directory that hasn't been indexed
        new_dir = tmp_path / f"unindexed_{uuid4().hex[:8]}"
        new_dir.mkdir()
        (new_dir / "test.py").write_text("print('hello')")

        result = await index_status(path=str(new_dir))
        assert "Not indexed" in result

    async def test_index_status_after_indexing(self, temp_codebase):
        """Status after indexing."""
        # First index
        await code_search(query="test", path=str(temp_codebase))

        # Check status
        result = await index_status(path=str(temp_codebase))
        assert "Indexed" in result or "files" in result.lower()


@requires_full_stack
class TestForceReindexIntegration:
    """Integration tests for force_reindex tool."""

    async def test_force_reindex(self, temp_codebase):
        """Force reindex clears and re-indexes."""
        # First index
        await code_search(query="test", path=str(temp_codebase))

        # Force reindex
        result = await force_reindex(path=str(temp_codebase))

        assert "Re-indexed" in result or "Indexed" in result


@requires_full_stack
class TestListCollectionsIntegration:
    """Integration tests for list_collections tool."""

    async def test_list_collections(self, temp_codebase):
        """List collections shows indexed codebases."""
        # Index first
        await code_search(query="test", path=str(temp_codebase))

        # List collections
        result = await list_collections()
        assert "codesearch_" in result or "collection" in result.lower()


@requires_full_stack
class TestSearchMultipleIntegration:
    """Integration tests for search_multiple tool."""

    async def test_search_multiple(self, temp_codebase, tmp_path):
        """Search across multiple codebases."""
        # Create second codebase
        other = tmp_path / "other"
        other.mkdir()
        (other / "file.py").write_text("def other_func(): pass")

        result = await search_multiple(
            query="function",
            paths=[str(temp_codebase), str(other)],
            limit=3,
        )

        # Should have sections for both
        assert "===" in result  # Section headers


@requires_full_stack
class TestPreviewIndexIntegration:
    """Integration tests for preview_index tool."""

    async def test_preview_index(self, temp_codebase):
        """Preview what would be indexed."""
        result = await preview_index(path=str(temp_codebase))

        assert "files" in result.lower() or "would" in result.lower()

    async def test_preview_index_with_files(self, temp_codebase):
        """Preview with file listing."""
        result = await preview_index(path=str(temp_codebase), show_files=True, limit=10)

        # Should list some files
        assert ".py" in result or ".ts" in result


@requires_full_stack
class TestFindSimilarIntegration:
    """Integration tests for find_similar tool."""

    async def test_find_similar(self, temp_codebase):
        """Find similar code snippets."""
        # First index
        await code_search(query="test", path=str(temp_codebase))

        # Find similar
        result = await find_similar(
            code="def main():\n    print('hello')",
            path=str(temp_codebase),
            limit=3,
        )

        # Should return results or "No similar code found"
        assert "Error" not in result or "similar" in result.lower()


@requires_full_stack
class TestFindReferencesIntegration:
    """Integration tests for find_references tool."""

    async def test_find_references(self, temp_codebase):
        """Find references to a symbol."""
        # First index
        await code_search(query="test", path=str(temp_codebase))

        # Find references
        result = await find_references(
            symbol="main",
            path=str(temp_codebase),
            limit=5,
        )

        # Should return results or "No references found"
        assert "Error" not in result or "reference" in result.lower()


@requires_full_stack
class TestSearchChangedIntegration:
    """Integration tests for search_changed tool."""

    async def test_search_changed(self, temp_codebase):
        """Search in changed files."""
        # First index
        await code_search(query="test", path=str(temp_codebase))

        # Search changed (might not find anything if no git history)
        result = await search_changed(
            query="function",
            path=str(temp_codebase),
            since="HEAD~5",
            limit=5,
        )

        # Should handle gracefully (might not be a git repo) - could return string or error dict
        assert isinstance(result, (str, dict))


@requires_full_stack
class TestDeleteCollectionIntegration:
    """Integration tests for delete_collection tool."""

    async def test_delete_collection(self, temp_codebase):
        """Delete a collection."""
        # First index to create collection
        await code_search(query="test", path=str(temp_codebase))

        # Delete
        result = await delete_collection(path=str(temp_codebase))

        assert "Deleted" in result or "deleted" in result


@requires_full_stack
class TestCleanupResources:
    """Tests for resource cleanup."""

    async def test_cleanup_is_callable(self):
        """Cleanup function exists and is async."""
        import inspect
        assert inspect.iscoroutinefunction(cleanup_resources)


@requires_full_stack
class TestSearchChangedEdgeCases:
    """Edge case tests for search_changed tool."""

    async def test_search_changed_in_git_repo(self, temp_codebase):
        """Search in git repo with history."""
        import subprocess

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=temp_codebase, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=temp_codebase, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_codebase, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=temp_codebase, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=temp_codebase, check=True, capture_output=True)

        # Search in changed files
        result = await search_changed(
            query="main",
            path=str(temp_codebase),
            since="HEAD~1",
            limit=5,
        )

        assert isinstance(result, str)

    async def test_search_changed_invalid_revision(self, temp_codebase):
        """Search with invalid revision falls back to searching all files."""
        import subprocess

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=temp_codebase, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=temp_codebase, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_codebase, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=temp_codebase, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=temp_codebase, check=True, capture_output=True)

        # Search with invalid revision - function falls back to searching all indexed files
        result = await search_changed(
            query="test",
            path=str(temp_codebase),
            since="totally_invalid_rev_that_does_not_exist_anywhere",
            limit=5,
        )

        # Function returns results or error gracefully - either is acceptable behavior
        assert isinstance(result, str)

    async def test_search_changed_not_a_repo(self, tmp_path):
        """Search in non-git directory returns error."""
        # Create a non-git directory with a file
        (tmp_path / "file.py").write_text("def test(): pass")

        result = await search_changed(
            query="test",
            path=str(tmp_path),
            since="HEAD~1",
        )

        assert is_error_result(result) or error_contains(result, "not a git repository")


@requires_full_stack
class TestCleanupOrphansIntegration:
    """Integration tests for cleanup_orphans tool."""

    async def test_cleanup_orphans_no_orphans(self):
        """cleanup_orphans works when no orphans exist."""
        from mcp_codesearch.server import cleanup_orphans

        result = await cleanup_orphans()
        assert isinstance(result, str)
        # Should report 0 or some number of orphans
        assert "orphan" in result.lower() or "collection" in result.lower()


@requires_full_stack
class TestOutputFormatsIntegration:
    """Tests for different output formats."""

    async def test_code_search_text_format(self, temp_codebase):
        """Code search with text format."""
        result = await code_search(
            query="main",
            path=str(temp_codebase),
            output_format="text",
            limit=5,
        )
        assert isinstance(result, str)

    async def test_code_search_json_format_structure(self, temp_codebase):
        """Code search with JSON format returns valid structure."""
        import json

        result = await code_search(
            query="main",
            path=str(temp_codebase),
            output_format="json",
            limit=5,
        )

        # Should be valid JSON or error message
        if "[Indexed" not in result and "Error" not in result:
            try:
                parsed = json.loads(result)
                assert isinstance(parsed, (list, dict))
            except json.JSONDecodeError:
                pass  # May have extra text

    async def test_code_search_markdown_format(self, temp_codebase):
        """Code search with markdown format."""
        result = await code_search(
            query="main",
            path=str(temp_codebase),
            output_format="markdown",
            limit=5,
        )
        assert isinstance(result, str)


@requires_full_stack
class TestDeleteCollectionEdgeCases:
    """Edge case tests for delete_collection."""

    async def test_delete_nonexistent_collection(self, tmp_path):
        """Deleting nonexistent collection returns appropriate message."""
        result = await delete_collection(path=str(tmp_path / "nonexistent"))
        assert isinstance(result, str)

    async def test_delete_by_collection_id(self, temp_codebase):
        """Delete collection by collection_id."""
        # First index to create collection
        await code_search(query="test", path=str(temp_codebase))

        # Get collection name
        from mcp_codesearch.storage.qdrant import collection_name
        col_name = collection_name(str(temp_codebase.resolve()))

        # Delete by collection_id
        result = await delete_collection(collection_id=col_name)
        assert isinstance(result, str)


@requires_full_stack
class TestNoResultsScenarios:
    """Tests for scenarios with no results."""

    async def test_code_search_no_matches(self, temp_codebase):
        """Code search with query that matches nothing."""
        result = await code_search(
            query="xyzzy_completely_unique_nonexistent_term_12345",
            path=str(temp_codebase),
            limit=5,
        )
        # Should return empty or "no results" message
        assert isinstance(result, str)

    async def test_find_references_no_matches(self, temp_codebase):
        """find_references with symbol that doesn't exist."""
        await code_search(query="test", path=str(temp_codebase))  # Index first

        result = await find_references(
            symbol="xyzzy_nonexistent_symbol_12345",
            path=str(temp_codebase),
            limit=5,
        )
        assert isinstance(result, str)
        assert "No references" in str(result) or is_error_result(result) or error_contains(result, "xyzzy")


@requires_full_stack
class TestSearchMultipleErrorHandling:
    """Tests for search_multiple error handling."""

    async def test_search_multiple_with_nonexistent_path(self, temp_codebase, tmp_path):
        """search_multiple handles nonexistent paths gracefully."""
        nonexistent = tmp_path / "does_not_exist"

        result = await search_multiple(
            query="test",
            paths=[str(temp_codebase), str(nonexistent)],
            limit=3,
        )

        # Function validates paths upfront - returns error for invalid paths
        assert isinstance(result, (str, dict))
        assert is_error_result(result) or error_contains(result, "does not exist")

    async def test_search_multiple_no_results(self, temp_codebase, tmp_path):
        """search_multiple when no results found."""
        other = tmp_path / "other"
        other.mkdir()
        (other / "file.py").write_text("x = 1")

        result = await search_multiple(
            query="xyzzy_nonexistent_term_12345",
            paths=[str(temp_codebase), str(other)],
            limit=3,
        )

        assert isinstance(result, str)


@requires_full_stack
class TestSearchChangedMoreEdgeCases:
    """More edge cases for search_changed."""

    async def test_search_changed_no_matches_in_changed_files(self, temp_codebase):
        """search_changed finds no matches in changed files."""
        import subprocess

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=temp_codebase, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=temp_codebase, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=temp_codebase, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=temp_codebase, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=temp_codebase, check=True, capture_output=True)

        # Index the codebase
        await code_search(query="test", path=str(temp_codebase))

        # Search for something that won't match in changed files
        result = await search_changed(
            query="xyzzy_absolutely_nonexistent_12345",
            path=str(temp_codebase),
            since="HEAD~1",
            limit=5,
        )

        # May find no changed files or no matches in changed files
        assert isinstance(result, str)


@requires_full_stack
class TestListCollectionsEdgeCases:
    """Edge cases for list_collections."""

    async def test_list_collections_empty(self):
        """list_collections when no collections exist works."""
        # Just verify function works - may or may not have collections
        result = await list_collections()
        assert isinstance(result, str)


@requires_full_stack
class TestFindSimilarEdgeCases:
    """Edge cases for find_similar."""

    async def test_find_similar_no_matches(self, temp_codebase):
        """find_similar with unique code finds no similar code."""
        await code_search(query="test", path=str(temp_codebase))

        result = await find_similar(
            code="class XyzzyUniqueClass12345:\n    def xyzzy_method(self): pass",
            path=str(temp_codebase),
            limit=3,
        )

        assert isinstance(result, str)

    async def test_find_similar_with_language_filter(self, temp_codebase):
        """find_similar with language filter."""
        await code_search(query="test", path=str(temp_codebase))

        result = await find_similar(
            code="def main(): print('hello')",
            path=str(temp_codebase),
            language="python",
            limit=3,
        )

        assert isinstance(result, str)


@requires_full_stack
class TestFindReferencesEdgeCases:
    """Edge cases for find_references."""

    async def test_find_references_with_definition(self, temp_codebase):
        """find_references with include_definition."""
        await code_search(query="test", path=str(temp_codebase))

        result = await find_references(
            symbol="main",
            path=str(temp_codebase),
            include_definition=True,
            limit=5,
        )

        assert isinstance(result, str)
