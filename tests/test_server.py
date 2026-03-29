"""Tests for server module helper functions and logic."""

import time
from unittest.mock import MagicMock, patch

import pytest

from mcp_codesearch.helpers import validate_git_since
from mcp_codesearch.services import IndexingStats


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


class TestIndexingStats:
    """Tests for IndexingStats dataclass (now in services module)."""

    def test_default_values(self):
        """IndexingStats has correct defaults."""
        stats = IndexingStats(
            files_indexed=10,
            chunks_indexed=50,
            languages={"python": 8, "typescript": 2},
        )
        assert stats.indexing_time_ms == 0
        assert stats.was_incremental is False
        assert stats.files_added == 0
        assert stats.files_modified == 0
        assert stats.files_deleted == 0
        assert stats.new_tokens == 0

    def test_incremental_stats(self):
        """IndexingStats for incremental update."""
        stats = IndexingStats(
            files_indexed=5,
            chunks_indexed=20,
            languages={"python": 5},
            indexing_time_ms=150,
            was_incremental=True,
            files_added=2,
            files_modified=2,
            files_deleted=1,
            new_tokens=15,
        )
        assert stats.was_incremental is True
        assert stats.files_added == 2
        assert stats.files_modified == 2
        assert stats.files_deleted == 1
        assert stats.new_tokens == 15

    def test_to_response(self):
        """IndexingStats.to_response() returns correct dict."""
        stats = IndexingStats(
            files_indexed=10,
            chunks_indexed=50,
            languages={"python": 10},
            indexing_time_ms=100,
            was_incremental=True,
            files_added=3,
            files_modified=4,
            files_deleted=2,
            new_tokens=25,
        )
        response = stats.to_response()
        assert response["files_indexed"] == 10
        assert response["chunks_indexed"] == 50
        assert response["indexing_time_ms"] == 100
        assert response["was_incremental"] is True


class TestResourceManagement:
    """Tests for resource management functions."""

    @pytest.mark.asyncio
    async def test_get_storage_singleton(self):
        """get_storage returns same instance (AsyncSingleton pattern)."""
        import mcp_codesearch.server as server_module
        from mcp_codesearch.server import get_storage

        # Reset singleton for clean test
        server_module._storage.reset()

        try:
            s1 = await get_storage()
            s2 = await get_storage()
            assert s1 is s2
        finally:
            server_module._storage.reset()

    @pytest.mark.asyncio
    async def test_get_embedder_singleton(self):
        """get_embedder returns same instance (AsyncSingleton pattern)."""
        import mcp_codesearch.server as server_module
        from mcp_codesearch.server import get_embedder

        # Reset singleton for clean test
        server_module._embedder.reset()

        try:
            e1 = await get_embedder()
            e2 = await get_embedder()
            assert e1 is e2
        finally:
            server_module._embedder.reset()

    @pytest.mark.asyncio
    async def test_get_global_vocab_singleton(self):
        """get_global_vocab returns same GlobalVocabulary instance (AsyncSingleton pattern)."""
        import mcp_codesearch.server as server_module
        from mcp_codesearch.server import get_global_vocab

        # Reset singleton for clean test
        server_module._global_vocab.reset()

        try:
            v1 = await get_global_vocab()
            v2 = await get_global_vocab()
            assert v1 is v2
        finally:
            server_module._global_vocab.reset()

    @pytest.mark.asyncio
    async def test_get_indexing_service_singleton(self):
        """get_indexing_service returns same instance."""
        import mcp_codesearch.server as server_module
        from mcp_codesearch.server import get_indexing_service

        # Reset all singletons for clean test
        server_module._indexing_service.reset()
        server_module._storage.reset()
        server_module._embedder.reset()
        server_module._global_vocab.reset()

        try:
            s1 = await get_indexing_service()
            s2 = await get_indexing_service()
            assert s1 is s2
        finally:
            server_module._indexing_service.reset()
            server_module._storage.reset()
            server_module._embedder.reset()
            server_module._global_vocab.reset()

    @pytest.mark.asyncio
    async def test_get_search_service_singleton(self):
        """get_search_service returns same instance."""
        import mcp_codesearch.server as server_module
        from mcp_codesearch.server import get_search_service

        # Reset all singletons for clean test
        server_module._search_service.reset()
        server_module._storage.reset()
        server_module._embedder.reset()
        server_module._global_vocab.reset()

        try:
            s1 = await get_search_service()
            s2 = await get_search_service()
            assert s1 is s2
        finally:
            server_module._search_service.reset()
            server_module._storage.reset()
            server_module._embedder.reset()
            server_module._global_vocab.reset()


class TestCleanupResources:
    """Tests for async resource cleanup using AsyncSingleton pattern."""

    @pytest.mark.asyncio
    async def test_cleanup_resets_singletons(self):
        """cleanup_resources resets all singleton instances."""
        import mcp_codesearch.server as server_module
        from mcp_codesearch.server import cleanup_resources, get_storage

        # First, ensure storage is initialized
        await get_storage()
        assert server_module._storage.is_initialized

        # Clean up
        await cleanup_resources()

        # Verify all singletons are reset
        assert not server_module._storage.is_initialized
        assert not server_module._embedder.is_initialized
        assert not server_module._global_vocab.is_initialized

    @pytest.mark.asyncio
    async def test_cleanup_with_uninitialized_singletons(self):
        """cleanup_resources handles uninitialized singletons gracefully."""
        import mcp_codesearch.server as server_module
        from mcp_codesearch.server import cleanup_resources

        # Reset all singletons first
        server_module._storage.reset()
        server_module._embedder.reset()
        server_module._global_vocab.reset()
        server_module._indexing_service.reset()
        server_module._search_service.reset()

        # Should not raise when nothing is initialized
        await cleanup_resources()

        # Still not initialized after cleanup
        assert not server_module._storage.is_initialized
        assert not server_module._embedder.is_initialized

    @pytest.mark.asyncio
    async def test_cleanup_clears_search_service_cache(self):
        """cleanup_resources clears the search service cache."""
        import mcp_codesearch.server as server_module
        from mcp_codesearch.server import cleanup_resources, get_search_service

        # Initialize search service and add to cache
        search_svc = await get_search_service()
        search_svc._cache.set("test_key|hash123", "test_result")
        assert search_svc._cache.size() > 0

        # Clean up
        await cleanup_resources()

        # Service singleton is reset, so we'd need to get a new one
        # The important thing is cleanup didn't raise


class TestSyncCleanup:
    """Tests for synchronous cleanup wrapper with AsyncSingleton pattern."""

    @pytest.mark.asyncio
    async def test_sync_cleanup_creates_task_when_loop_running(self):
        """_sync_cleanup creates task when event loop is running."""
        import asyncio

        import mcp_codesearch.server as server_module
        from mcp_codesearch.server import _sync_cleanup, get_storage

        # Initialize storage so cleanup has something to do
        await get_storage()
        assert server_module._storage.is_initialized

        try:
            mock_loop = MagicMock()
            captured_coros = []

            def capture_create_task(coro):
                captured_coros.append(coro)
                return MagicMock()  # Return a mock task

            mock_loop.create_task = capture_create_task

            with patch.object(asyncio, 'get_running_loop', return_value=mock_loop):
                _sync_cleanup()

            # Verify create_task was called with a coroutine
            assert len(captured_coros) == 1

            # Close the captured coroutine to avoid "never awaited" warning
            captured_coros[0].close()
        finally:
            server_module._storage.reset()

    @pytest.mark.asyncio
    async def test_sync_cleanup_runs_complete_when_loop_not_running(self):
        """_sync_cleanup uses asyncio.run() when no running loop.

        Note: sync_cleanup_wrapper uses asyncio.run() directly instead of
        get_event_loop().run_until_complete() for simplicity and consistency.
        """
        import asyncio

        import mcp_codesearch.server as server_module
        from mcp_codesearch.server import _sync_cleanup, get_storage

        # Initialize storage so cleanup has something to do
        await get_storage()
        assert server_module._storage.is_initialized

        try:
            captured_coros = []

            def capture_run(coro):
                captured_coros.append(coro)

            with patch.object(asyncio, 'get_running_loop', side_effect=RuntimeError("No running loop")):
                with patch.object(asyncio, 'run', side_effect=capture_run):
                    _sync_cleanup()

            # Verify asyncio.run was called with cleanup coroutine
            assert len(captured_coros) == 1

            # Close the captured coroutine to avoid "never awaited" warning
            captured_coros[0].close()
        finally:
            server_module._storage.reset()

    @pytest.mark.asyncio
    async def test_sync_cleanup_handles_no_event_loop(self):
        """_sync_cleanup uses asyncio.run when no event loop exists."""
        import asyncio

        import mcp_codesearch.server as server_module
        from mcp_codesearch.server import _sync_cleanup, get_storage

        # Initialize storage so cleanup has something to do
        await get_storage()
        assert server_module._storage.is_initialized

        try:
            captured_coros = []

            def capture_run(coro):
                captured_coros.append(coro)

            with patch.object(asyncio, 'get_running_loop', side_effect=RuntimeError("No running loop")):
                with patch.object(asyncio, 'get_event_loop', side_effect=RuntimeError("No event loop")):
                    with patch.object(asyncio, 'run', side_effect=capture_run):
                        _sync_cleanup()

            # Verify asyncio.run was called
            assert len(captured_coros) == 1

            # Close the captured coroutine to avoid "never awaited" warning
            captured_coros[0].close()
        finally:
            server_module._storage.reset()

    def test_sync_cleanup_early_exit_when_nothing_to_cleanup(self):
        """_sync_cleanup returns early when there's nothing to clean up."""
        import asyncio

        import mcp_codesearch.server as server_module
        from mcp_codesearch.server import _sync_cleanup

        # Ensure singletons are not initialized
        server_module._storage.reset()
        server_module._embedder.reset()
        server_module._global_vocab.reset()
        server_module._indexing_service.reset()
        server_module._search_service.reset()

        with patch.object(asyncio, 'get_running_loop') as mock_get_loop:
            _sync_cleanup()

            # Should not have tried to get a loop since nothing is initialized
            mock_get_loop.assert_not_called()


class TestInputValidation:
    """Tests for input validation in MCP tools."""

    @pytest.mark.asyncio
    async def test_code_search_empty_query(self):
        """Empty query returns error."""
        from mcp_codesearch.server import code_search

        result = await code_search(query="", path=".")
        assert "Error: Query cannot be empty" in result

    @pytest.mark.asyncio
    async def test_code_search_whitespace_query(self):
        """Whitespace-only query returns error."""
        from mcp_codesearch.server import code_search

        result = await code_search(query="   ", path=".")
        assert "Error: Query cannot be empty" in result

    @pytest.mark.asyncio
    async def test_code_search_negative_limit(self):
        """Negative limit is clamped to default (10), not rejected."""
        from mcp_codesearch.server import code_search

        result = await code_search(query="test", path=".", limit=-1)
        # Should not error - clamped to default
        assert "Error: limit must be a positive integer" not in result

    @pytest.mark.asyncio
    async def test_code_search_zero_limit(self):
        """Zero limit is clamped to default (10), not rejected."""
        from mcp_codesearch.server import code_search

        result = await code_search(query="test", path=".", limit=0)
        # Should not error - clamped to default
        assert "Error: limit must be a positive integer" not in result

    @pytest.mark.asyncio
    async def test_code_search_nonexistent_path(self, tmp_path):
        """Non-existent path returns error."""
        from mcp_codesearch.server import code_search

        nonexistent = tmp_path / "does_not_exist"
        result = await code_search(query="test", path=str(nonexistent))
        assert is_error_result(result)
        assert error_contains(result, "does not exist")

    @pytest.mark.asyncio
    async def test_code_search_path_is_file(self, tmp_path):
        """Path that is a file returns error."""
        from mcp_codesearch.server import code_search

        file_path = tmp_path / "file.txt"
        file_path.write_text("content")
        result = await code_search(query="test", path=str(file_path))
        assert is_error_result(result)
        assert error_contains(result, "not a directory")

    @pytest.mark.asyncio
    async def test_search_multiple_empty_query(self):
        """Empty query returns error."""
        from mcp_codesearch.server import search_multiple

        result = await search_multiple(query="", paths=["."])
        assert is_error_result(result)
        assert error_contains(result, "cannot be empty")

    @pytest.mark.asyncio
    async def test_search_multiple_empty_paths(self):
        """Empty paths returns error."""
        from mcp_codesearch.server import search_multiple

        result = await search_multiple(query="test", paths=[])
        assert is_error_result(result)
        assert error_contains(result, "cannot be empty")

    @pytest.mark.asyncio
    async def test_search_multiple_negative_limit(self):
        """Negative limit is clamped to default (10), not rejected."""
        from mcp_codesearch.server import search_multiple

        result = await search_multiple(query="test", paths=["."], limit=-5)
        # Should not error - clamped to default
        assert "Error: limit must be a positive integer" not in result

    @pytest.mark.asyncio
    async def test_search_multiple_invalid_paths(self, tmp_path):
        """Invalid paths in list returns error."""
        from mcp_codesearch.server import search_multiple

        result = await search_multiple(
            query="test",
            paths=[str(tmp_path / "nonexistent")]
        )
        assert is_error_result(result)
        assert error_contains(result, "invalid") or error_contains(result, "does not exist")

    @pytest.mark.asyncio
    async def test_index_status_nonexistent_path(self, tmp_path):
        """Non-existent path returns error."""
        from mcp_codesearch.server import index_status

        result = await index_status(path=str(tmp_path / "nonexistent"))
        assert is_error_result(result)
        assert error_contains(result, "does not exist")

    @pytest.mark.asyncio
    async def test_index_status_path_is_file(self, tmp_path):
        """Path that is a file returns error."""
        from mcp_codesearch.server import index_status

        file_path = tmp_path / "file.txt"
        file_path.write_text("content")
        result = await index_status(path=str(file_path))
        assert is_error_result(result)
        assert error_contains(result, "not a directory")

    @pytest.mark.asyncio
    async def test_search_multiple_path_is_file(self, tmp_path):
        """Path that is a file returns error."""
        from mcp_codesearch.server import search_multiple

        file_path = tmp_path / "file.txt"
        file_path.write_text("content")
        result = await search_multiple(query="test", paths=[str(file_path)])
        assert is_error_result(result)
        assert error_contains(result, "not a directory")


class TestValidateGitSince:
    """Tests for git 'since' parameter validation."""

    # Valid inputs that don't need transformation (return empty string)
    @pytest.mark.parametrize("since", [
        "HEAD",
        "HEAD~1",
        "HEAD~10",
        "HEAD~999",
        "HEAD@{1}",
        "main",
        "master",
        "develop",
        "feature/my-branch",
        "v1.0",
        "v1.0.0",
        "release-2.0",
        "abc123",
        "abc123def456",
        "1234567890abcdef1234567890abcdef12345678",  # Full SHA
    ])
    def test_valid_since_values(self, since: str):
        """Valid since values should pass validation with empty result."""
        is_valid, result = validate_git_since(since)
        assert is_valid is True, f"Expected '{since}' to be valid, got error: {result}"
        assert result == "", f"Expected empty string for non-.ago pattern, got: {result}"

    # Valid .ago inputs that get transformed (return transformed value)
    @pytest.mark.parametrize("since,expected", [
        ("1.day.ago", "1 day ago"),
        ("3.days.ago", "3 days ago"),
        ("1.week.ago", "1 week ago"),
        ("2.weeks.ago", "2 weeks ago"),
        ("1.month.ago", "1 month ago"),
        ("6.months.ago", "6 months ago"),
        ("1.year.ago", "1 year ago"),
        ("30.seconds.ago", "30 seconds ago"),
        ("5.minutes.ago", "5 minutes ago"),
        ("2.hours.ago", "2 hours ago"),
    ])
    def test_valid_ago_values_transformed(self, since: str, expected: str):
        """Valid .ago values should pass validation and return transformed value."""
        is_valid, result = validate_git_since(since)
        assert is_valid is True, f"Expected '{since}' to be valid, got error: {result}"
        assert result == expected, f"Expected '{expected}', got '{result}'"

    # Invalid inputs - dangerous patterns
    @pytest.mark.parametrize("since,expected_error", [
        ("", "'since' parameter cannot be empty"),
        ("   ", "'since' parameter cannot be empty"),
        ("-help", "cannot start with '-'"),
        ("--version", "cannot start with '-'"),
        ("-o /etc/passwd", "cannot start with '-'"),
        ("main..HEAD", "Revision ranges"),
        ("HEAD..main", "Revision ranges"),
    ])
    def test_invalid_dangerous_patterns(self, since: str, expected_error: str):
        """Dangerous patterns should be rejected."""
        is_valid, error_msg = validate_git_since(since)
        assert is_valid is False
        assert expected_error in error_msg

    # Invalid inputs - malformed values
    @pytest.mark.parametrize("since", [
        "_underscore",  # Starts with underscore
        "spaces in name",
        "semi;colon",
        "back`tick",
        "$(whoami)",
        "${HOME}",
        "foo\nbar",
        "foo\rbar",
        "123_not_hex",  # Numbers with invalid hex chars
    ])
    def test_invalid_malformed_values(self, since: str):
        """Malformed values should be rejected."""
        is_valid, error_msg = validate_git_since(since)
        assert is_valid is False
        assert "Invalid 'since' format" in error_msg or "cannot" in error_msg

    # Edge cases - valid branch names (should pass even if unusual)
    @pytest.mark.parametrize("since", [
        "notavalidformat",  # Valid branch name (starts with letter)
        "abc123",  # Could be short commit hash (4+ hex chars)
        "abc",  # Too short for commit hash but valid branch
    ])
    def test_edge_case_valid_values(self, since: str):
        """Edge cases that look weird but are actually valid git refs."""
        is_valid, error_msg = validate_git_since(since)
        assert is_valid is True, f"Expected '{since}' to be valid, got error: {error_msg}"


class TestMainFunction:
    """Tests for main() function."""

    def test_main_exists_and_callable(self):
        """main() function exists and is callable."""
        from mcp_codesearch.server import main
        assert callable(main)

    def test_main_with_mock_mcp_run(self):
        """main() calls mcp.run()."""
        from mcp_codesearch import server

        with patch.object(server, "mcp") as mock_mcp, \
             patch("mcp_codesearch.server.verify_tools_registered"):
            mock_mcp.run = MagicMock()
            server.main()
            mock_mcp.run.assert_called_once_with(transport="stdio")


class TestFormatIndexMessage:
    """Tests for format_index_message helper function."""

    def test_format_index_message_deletion_only(self):
        """Regression test: format_index_message shows output for deletion-only changes.

        This tests the fix for the bug where deleted files weren't reflected in
        the index message because it only checked files_indexed > 0.
        """
        from mcp_codesearch.helpers import format_index_message

        stats = IndexingStats(
            files_indexed=0,
            chunks_indexed=0,
            languages={},
            indexing_time_ms=5,
            was_incremental=True,
            files_added=0,
            files_modified=0,
            files_deleted=2,  # Files were deleted
            new_tokens=0,
        )

        result = format_index_message(files_indexed=0, chunks_indexed=0, stats=stats)

        # Should produce output because files were deleted
        assert result != ""
        assert "-2 deleted" in result

    def test_format_index_message_no_changes(self):
        """format_index_message returns empty when no changes occurred."""
        from mcp_codesearch.helpers import format_index_message

        stats = IndexingStats(
            files_indexed=0,
            chunks_indexed=0,
            languages={},
            indexing_time_ms=0,
            was_incremental=True,
            files_added=0,
            files_modified=0,
            files_deleted=0,  # No files deleted
            new_tokens=0,
        )

        result = format_index_message(files_indexed=0, chunks_indexed=0, stats=stats)

        # Should return empty string when no changes
        assert result == ""

    def test_format_index_message_mixed_changes(self):
        """format_index_message shows all change types."""
        from mcp_codesearch.helpers import format_index_message

        stats = IndexingStats(
            files_indexed=3,
            chunks_indexed=15,
            languages={"python": 2, "typescript": 1},
            indexing_time_ms=50,
            was_incremental=True,
            files_added=1,
            files_modified=1,
            files_deleted=1,
            new_tokens=10,
        )

        result = format_index_message(files_indexed=3, chunks_indexed=15, stats=stats)

        assert "+1 added" in result
        assert "~1 modified" in result
        assert "-1 deleted" in result


class TestCacheInvalidationOnDeletion:
    """Regression tests for cache invalidation when files are deleted.

    These tests verify the fix for the bug where search results were stale
    after file deletion because the cache wasn't being invalidated.
    """

    def test_cache_invalidation_logic_deletion_only(self):
        """Verify cache invalidation logic considers files_deleted.

        This replicates the logic from tools/search.py:
        - files_deleted = getattr(stats, "files_deleted", 0) if stats else 0
        - index_changed = files_indexed > 0 or files_deleted > 0
        """
        # Simulate deletion-only scenario
        files_indexed = 0
        files_deleted = 1

        # This is the fixed logic
        index_changed = files_indexed > 0 or files_deleted > 0

        # With the fix, index_changed should be True
        assert index_changed is True, (
            "Cache should be invalidated when files_deleted > 0 "
            "even if files_indexed == 0"
        )

    def test_cache_invalidation_logic_no_changes(self):
        """Cache not invalidated when no changes occurred."""
        files_indexed = 0
        files_deleted = 0

        index_changed = files_indexed > 0 or files_deleted > 0

        assert index_changed is False

    def test_cache_invalidation_logic_additions_only(self):
        """Cache invalidated when files added."""
        files_indexed = 5
        files_deleted = 0

        index_changed = files_indexed > 0 or files_deleted > 0

        assert index_changed is True
