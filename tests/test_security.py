"""Security regression tests for mcp-codesearch.

Tests for:
- Git since validation (prevents command injection via .ago patterns)
- TOCTOU safety (symlink handling during file reads)
- Path traversal prevention
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_codesearch.helpers import validate_git_since, validate_path_containment


class TestGitSinceValidation:
    """Tests for git 'since' parameter security validation."""

    def test_rejects_ambiguous_ago_patterns(self):
        """Invalid .ago patterns like 'feature.branch.ago' should be rejected.

        This prevents validation/transformation mismatch where 'feature.branch.ago'
        would pass validation then transform to ambiguous 'feature branch ago'.
        """
        # These should be rejected - they look like .ago but aren't valid time specs
        invalid_patterns = [
            "feature.branch.ago",
            "main.release.ago",
            "v1.0.0.ago",
            "some.random.thing.ago",
            "days.ago",  # Missing number
            ".days.ago",  # Starts with dot
            "3days.ago",  # Missing dot between number and unit
        ]
        for pattern in invalid_patterns:
            is_valid, result = validate_git_since(pattern)
            assert not is_valid, f"Expected '{pattern}' to be rejected"
            assert "Error" in result

    def test_transformation_matches_validation(self):
        """Valid .ago patterns should return the transformed value."""
        valid_patterns = [
            ("3.days.ago", "3 days ago"),
            ("1.day.ago", "1 day ago"),
            ("2.weeks.ago", "2 weeks ago"),
            ("1.week.ago", "1 week ago"),
            ("6.months.ago", "6 months ago"),
            ("1.month.ago", "1 month ago"),
            ("1.year.ago", "1 year ago"),
            ("30.seconds.ago", "30 seconds ago"),
            ("5.minutes.ago", "5 minutes ago"),
            ("2.hours.ago", "2 hours ago"),
        ]
        for input_val, expected_output in valid_patterns:
            is_valid, result = validate_git_since(input_val)
            assert is_valid, f"Expected '{input_val}' to be valid, got: {result}"
            assert result == expected_output, f"Expected '{expected_output}', got '{result}'"

    def test_non_ago_patterns_return_empty_string(self):
        """Non-.ago patterns should return empty string (no transformation)."""
        patterns = ["HEAD~10", "main", "abc123def456"]
        for pattern in patterns:
            is_valid, result = validate_git_since(pattern)
            assert is_valid
            assert result == "", f"Expected empty string for '{pattern}', got '{result}'"

    def test_command_injection_blocked(self):
        """Patterns that could be command injection should be rejected."""
        dangerous = [
            "-help",
            "--version",
            "-o /etc/passwd",
            "$(whoami)",
            "${HOME}",
            "; rm -rf /",
            "| cat /etc/passwd",
        ]
        for pattern in dangerous:
            is_valid, _ = validate_git_since(pattern)
            assert not is_valid, f"Expected '{pattern}' to be rejected"


class TestTOCTOUSafety:
    """Tests for TOCTOU-safe file reading."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require admin on Windows")
    def test_symlink_not_followed_during_read(self, tmp_path):
        """Symlinks should be rejected atomically during read.

        This tests the _safe_read_file function indirectly through read_specific_files.
        """
        from mcp_codesearch.indexer.discovery import _safe_read_file

        # Create a regular file
        regular_file = tmp_path / "regular.txt"
        regular_file.write_text("regular content")

        # Create a symlink
        symlink_file = tmp_path / "symlink.txt"
        symlink_file.symlink_to(regular_file)

        # Regular file should be readable
        content = _safe_read_file(regular_file)
        assert content == "regular content"

        # Symlink should return None (rejected)
        content = _safe_read_file(symlink_file)
        assert content is None

    def test_nonexistent_file_returns_none(self, tmp_path):
        """Non-existent files should return None without raising."""
        from mcp_codesearch.indexer.discovery import _safe_read_file

        nonexistent = tmp_path / "does_not_exist.txt"
        content = _safe_read_file(nonexistent)
        assert content is None

    def test_readable_file_returns_content(self, tmp_path):
        """Regular readable files should return their content."""
        from mcp_codesearch.indexer.discovery import _safe_read_file

        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        content = _safe_read_file(test_file)
        assert content == "print('hello')"


class TestPathTraversal:
    """Tests for path traversal prevention."""

    def test_blocks_path_traversal(self, tmp_path):
        """Path traversal attempts should be blocked."""
        root = tmp_path / "project"
        root.mkdir()

        # These should all be blocked (Unix-style paths only - Windows backslashes
        # are treated as literal characters on Unix, which is correct behavior)
        traversal_attempts = [
            "../../../etc/passwd",
            "subdir/../../etc/passwd",
            "foo/../../../etc/passwd",
        ]

        for attempt in traversal_attempts:
            result = validate_path_containment(root / attempt, root)
            assert not result, f"Path traversal should be blocked: {attempt}"

    def test_allows_valid_paths(self, tmp_path):
        """Valid paths within root should be allowed."""
        root = tmp_path / "project"
        root.mkdir()
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("pass")

        valid_paths = [
            "src/main.py",
            "./src/main.py",
            "src/../src/main.py",  # Resolves within root
        ]

        for path in valid_paths:
            result = validate_path_containment(root / path, root)
            assert result, f"Valid path should be allowed: {path}"

    def test_read_specific_files_blocks_traversal(self, tmp_path):
        """read_specific_files should block path traversal attempts."""
        from mcp_codesearch.indexer.discovery import read_specific_files

        # Create a project directory
        project = tmp_path / "project"
        project.mkdir()
        (project / "safe.py").write_text("print('safe')")

        # Create a "secret" file outside project
        secret = tmp_path / "secret.txt"
        secret.write_text("secret data")

        # Attempt to read files with path traversal
        results = list(read_specific_files(
            project,
            {"../secret.txt", "safe.py"}
        ))

        # Only safe.py should be readable, not the traversal attempt
        paths = [r.rel_path for r in results]
        assert "safe.py" in paths
        assert "../secret.txt" not in paths


class TestSymlinkDirectoryHandling:
    """Tests for symlink directory handling in discovery."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require admin on Windows")
    def test_symlinked_directories_skipped(self, tmp_path):
        """Symlinked directories should be skipped to prevent loops."""
        from mcp_codesearch.indexer.discovery import discover_files

        # Create a directory structure
        project = tmp_path / "project"
        project.mkdir()
        src = project / "src"
        src.mkdir()
        (src / "main.py").write_text("print('main')")

        # Create a symlink that would cause a loop
        loop_link = project / "loop"
        loop_link.symlink_to(project)

        # Discovery should complete without infinite loop
        files = list(discover_files(project))
        paths = [f.rel_path for f in files]

        # Should find main.py but not follow the loop
        assert "src/main.py" in paths
        # Should not have duplicates from following the symlink
        assert paths.count("src/main.py") == 1
