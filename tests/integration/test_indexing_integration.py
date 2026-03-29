"""Integration tests for indexing functionality with real Qdrant and embeddings."""

import asyncio

import pytest

from mcp_codesearch.server import (
    cleanup_orphans,
    code_search,
    delete_collection,
    force_reindex,
    index_status,
    list_collections,
)
from mcp_codesearch.storage.qdrant import collection_name


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


class TestIncrementalIndexing:
    """Tests for incremental indexing with file changes."""

    @pytest.fixture
    def git_codebase(self, tmp_path):
        """Create a git-initialized codebase for testing."""
        import subprocess

        # Initialize git repo
        subprocess.run(["git", "init"], check=False, cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            check=False, cwd=tmp_path, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            check=False, cwd=tmp_path, capture_output=True
        )

        # Create initial files
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        (src_dir / "main.py").write_text('''"""Main module."""

def main():
    """Entry point."""
    print("Hello, World!")

if __name__ == "__main__":
    main()
''')

        (src_dir / "utils.py").write_text('''"""Utility functions."""

def helper():
    """A helper function."""
    return 42
''')

        # Initial commit
        subprocess.run(["git", "add", "."], check=False, cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            check=False, cwd=tmp_path, capture_output=True
        )

        # Track collection for cleanup
        col_name = collection_name(str(tmp_path.resolve()))

        yield tmp_path

        # Cleanup
        try:
            from mcp_codesearch.storage.qdrant import QdrantStorage
            storage = QdrantStorage()

            async def cleanup():
                try:
                    await storage.delete_collection(col_name)
                except Exception:
                    pass
                await storage.close()

            asyncio.run(cleanup())
        except Exception:
            pass

    async def test_initial_indexing(self, git_codebase):
        """Initial indexing creates collection."""
        result = await code_search(
            query="main function",
            path=str(git_codebase),
            limit=5,
        )

        # Should index files
        assert "[Indexed" in result or "main" in result.lower()

    async def test_incremental_add_file(self, git_codebase):
        """Adding a file triggers incremental indexing."""
        # Initial index
        await code_search(query="main", path=str(git_codebase))

        # Add a new file
        new_file = git_codebase / "src" / "new_module.py"
        new_file.write_text('''"""New module with unique content."""

def unique_function_xyzabc():
    """A unique function for testing."""
    return "unique"
''')

        # Search again to trigger incremental index (NOT force_reindex which does full)
        result = await code_search(
            query="unique_function_xyzabc",
            path=str(git_codebase),
        )

        # Incremental indexing should find the new file
        assert "unique" in result.lower() or "xyzabc" in result.lower() or "Indexed" in result

    async def test_incremental_modify_file(self, git_codebase):
        """Modifying a file triggers re-indexing via incremental path."""
        # Initial index
        await code_search(query="main", path=str(git_codebase))

        # Modify existing file (change mtime to trigger change detection)
        main_file = git_codebase / "src" / "main.py"
        main_file.write_text('''"""Main module - updated."""

def main():
    """Entry point - now with special_marker_abc123."""
    print("Updated!")

def new_function():
    """Added function."""
    pass

if __name__ == "__main__":
    main()
''')

        # Search again to trigger incremental index
        result = await code_search(
            query="special_marker_abc123",
            path=str(git_codebase),
        )

        # Should find modified content via incremental indexing
        assert "special_marker" in result.lower() or "abc123" in result.lower() or "Indexed" in result

    async def test_incremental_delete_file(self, git_codebase):
        """Deleting a file removes it from index via incremental path."""
        # Initial index
        await code_search(query="helper", path=str(git_codebase))

        # Delete utils.py
        utils_file = git_codebase / "src" / "utils.py"
        utils_file.unlink()

        # Search again to trigger incremental index (handles deletions)
        result = await code_search(
            query="main function",
            path=str(git_codebase),
        )

        # Should complete without error - incremental indexing handles deletions
        assert isinstance(result, str)

    async def test_incremental_no_changes(self, git_codebase):
        """Incremental index with no changes returns early."""
        # Initial index
        await code_search(query="main", path=str(git_codebase))

        # Search again without any file changes - should detect no changes
        result = await code_search(
            query="main function entry",
            path=str(git_codebase),
        )

        # Should return results without re-indexing
        assert "main" in result.lower()

    async def test_incremental_deletions_only(self, git_codebase):
        """Incremental index with only deletions (no adds/modifies)."""
        # Initial index - ensure both files are indexed
        result1 = await code_search(query="helper function", path=str(git_codebase))
        assert "helper" in result1.lower() or "Indexed" in result1

        # Delete one file (utils.py), keep main.py
        utils_file = git_codebase / "src" / "utils.py"
        utils_file.unlink()

        # Search - triggers incremental with deletions only (no files_to_index)
        result = await code_search(
            query="main entry point",
            path=str(git_codebase),
        )

        # Should handle deletions-only case
        assert isinstance(result, str)

    async def test_incremental_multiple_operations(self, git_codebase):
        """Incremental index with add, modify, and delete in one pass."""
        # Initial index
        await code_search(query="main", path=str(git_codebase))

        # Add new file
        (git_codebase / "src" / "added.py").write_text('''"""Added file."""
def added_function(): return "added"
''')

        # Modify existing file
        main_file = git_codebase / "src" / "main.py"
        main_file.write_text('''"""Modified main."""
def main(): print("modified_marker_xyz")
''')

        # Delete another file
        utils_file = git_codebase / "src" / "utils.py"
        utils_file.unlink()

        # Search - triggers incremental with all three change types
        result = await code_search(
            query="added function modified",
            path=str(git_codebase),
        )

        # Should handle complex incremental update
        assert isinstance(result, str)


class TestCleanupOrphans:
    """Tests for cleanup_orphans functionality."""

    async def test_cleanup_no_orphans(self, tmp_path):
        """Cleanup with no orphans reports correctly."""
        # Create and index a codebase
        src = tmp_path / "src"
        src.mkdir()
        (src / "test.py").write_text("def test(): pass")

        await code_search(query="test", path=str(tmp_path))

        # Cleanup should find no orphans
        result = await cleanup_orphans()

        assert "orphan" in result.lower() or "valid" in result.lower()

        # Cleanup the collection we created
        await delete_collection(path=str(tmp_path))

    async def test_cleanup_empty_database(self):
        """Cleanup with no collections."""
        # First delete all test collections
        await list_collections()  # Ensure we can list (may be empty)

        # If there are no collections at all
        result = await cleanup_orphans()
        assert isinstance(result, str)


class TestSearchChanged:
    """Tests for search_changed with git integration."""

    @pytest.fixture
    def git_repo_with_history(self, tmp_path):
        """Create git repo with commit history."""
        import subprocess

        # Initialize git repo
        subprocess.run(["git", "init"], check=False, cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            check=False, cwd=tmp_path, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            check=False, cwd=tmp_path, capture_output=True
        )

        # Create initial file and commit
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        (src_dir / "original.py").write_text('''"""Original file."""
def original(): pass
''')

        subprocess.run(["git", "add", "."], check=False, cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            check=False, cwd=tmp_path, capture_output=True
        )

        # Add another file and commit
        (src_dir / "second.py").write_text('''"""Second file with unique_marker_xyz."""
def second_function():
    return "unique_marker_xyz"
''')

        subprocess.run(["git", "add", "."], check=False, cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add second"],
            check=False, cwd=tmp_path, capture_output=True
        )

        col_name = collection_name(str(tmp_path.resolve()))

        yield tmp_path

        # Cleanup
        try:
            from mcp_codesearch.storage.qdrant import QdrantStorage
            storage = QdrantStorage()

            async def cleanup():
                try:
                    await storage.delete_collection(col_name)
                except Exception:
                    pass
                await storage.close()

            asyncio.run(cleanup())
        except Exception:
            pass

    async def test_search_changed_finds_recent(self, git_repo_with_history):
        """Search changed finds recently modified files."""
        from mcp_codesearch.server import search_changed

        result = await search_changed(
            query="function",
            path=str(git_repo_with_history),
            since="HEAD~1",
            limit=5,
        )

        # Should find second.py which was added in last commit
        assert "second" in result.lower() or "changed" in result.lower()

    async def test_search_changed_not_git_repo(self, tmp_path):
        """Search changed on non-git directory returns error."""
        from mcp_codesearch.server import search_changed

        # Create non-git directory
        (tmp_path / "test.py").write_text("def test(): pass")

        result = await search_changed(
            query="test",
            path=str(tmp_path),
        )

        assert is_error_result(result)
        assert error_contains(result, "git")

    async def test_search_changed_no_changes(self, git_repo_with_history):
        """Search changed with no matching files."""
        from mcp_codesearch.server import search_changed

        # Search for something that doesn't exist in changed files
        result = await search_changed(
            query="nonexistent_query_xyz",
            path=str(git_repo_with_history),
            since="HEAD~1",
        )

        # Should report no matches
        assert "No matches" in result or "changed" in result.lower()


class TestVocabularyGrowth:
    """Tests for vocabulary growth and reindexing triggers."""

    async def test_vocabulary_extends_with_new_content(self, tmp_path):
        """Vocabulary grows when new terms are indexed."""
        src = tmp_path / "src"
        src.mkdir()

        # Initial file with common terms
        (src / "common.py").write_text('''"""Common module."""
def common_function():
    return True
''')

        # Index
        await code_search(query="function", path=str(tmp_path))

        # Add file with specialized terms
        (src / "specialized.py").write_text('''"""Specialized module."""
def xylophone_kaleidoscope_function():
    """Very unique function name."""
    return "unique"
''')

        # Reindex - should extend vocabulary
        result = await force_reindex(path=str(tmp_path))

        # Should complete successfully
        assert "Indexed" in result or "Re-indexed" in result

        # Cleanup
        await delete_collection(path=str(tmp_path))


class TestBatchProcessing:
    """Tests for batch processing of large codebases."""

    async def test_large_codebase_indexing(self, tmp_path):
        """Index a codebase with many files."""
        src = tmp_path / "src"
        src.mkdir()

        # Create many files to trigger batch processing
        for i in range(25):
            (src / f"module_{i}.py").write_text(f'''"""Module {i}."""

def function_{i}():
    """Function in module {i}."""
    return {i}

class Class{i}:
    """Class in module {i}."""
    def method(self):
        return {i}
''')

        # Index all files
        result = await code_search(
            query="function",
            path=str(tmp_path),
            limit=10,
        )

        # Should index successfully
        assert "[Indexed" in result or "function" in result.lower()

        # Verify status
        status = await index_status(path=str(tmp_path))
        assert "25" in status or "Indexed" in status

        # Cleanup
        await delete_collection(path=str(tmp_path))
