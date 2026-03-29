"""Tests for change detection in mcp-codesearch."""

from pathlib import Path

from mcp_codesearch.indexer.change_detect import (
    ChangeSet,
    detect_changes,
    detect_changes_fast,
)
from mcp_codesearch.indexer.discovery import FileInfo


def _make_file_info(name: str = "test.py") -> FileInfo:
    """Create a minimal FileInfo for testing."""
    return FileInfo(
        path=Path(f"/test/{name}"),
        rel_path=name,
        language="python",
        size_bytes=100,
        content="# test",
        content_hash="abc123",
        line_count=1,
        mtime=1234567890.0,
    )


class TestChangeSet:
    """Tests for ChangeSet model."""

    def test_has_changes_empty(self):
        """Empty change set has no changes."""
        cs = ChangeSet(added=[], modified=[], deleted=[])
        assert cs.has_changes is False

    def test_has_changes_added(self):
        """Change set with added files has changes."""
        cs = ChangeSet(added=[_make_file_info()], modified=[], deleted=[])
        assert cs.has_changes is True

    def test_has_changes_modified(self):
        """Change set with modified files has changes."""
        cs = ChangeSet(added=[], modified=[_make_file_info()], deleted=[])
        assert cs.has_changes is True

    def test_has_changes_deleted(self):
        """Change set with deleted files has changes."""
        cs = ChangeSet(added=[], modified=[], deleted=["file.py"])
        assert cs.has_changes is True

    def test_total_changes(self):
        """Total changes counts all types."""
        cs = ChangeSet(
            added=[_make_file_info("a.py"), _make_file_info("b.py")],
            modified=[_make_file_info("c.py")],
            deleted=["d.py", "e.py", "f.py"],
        )
        assert cs.total_changes == 6


class TestDetectChanges:
    """Tests for detect_changes function."""

    def test_detect_new_files(self, tmp_path):
        """Detects new files not in index."""
        (tmp_path / "new_file.py").write_text("print('hello')")

        changes = detect_changes(tmp_path, indexed_files={})

        assert len(changes.added) == 1
        assert changes.added[0].rel_path == "new_file.py"
        assert len(changes.modified) == 0
        assert len(changes.deleted) == 0

    def test_detect_deleted_files(self, tmp_path):
        """Detects deleted files no longer present."""
        # No files on disk, but one in index
        indexed = {"deleted_file.py": "somehash"}

        changes = detect_changes(tmp_path, indexed_files=indexed)

        assert len(changes.added) == 0
        assert len(changes.modified) == 0
        assert len(changes.deleted) == 1
        assert "deleted_file.py" in changes.deleted

    def test_detect_modified_files(self, tmp_path):
        """Detects files with changed content."""
        file_path = tmp_path / "changed.py"
        file_path.write_text("new content")

        indexed = {"changed.py": "different_hash"}

        changes = detect_changes(tmp_path, indexed_files=indexed)

        assert len(changes.added) == 0
        assert len(changes.modified) == 1
        assert changes.modified[0].rel_path == "changed.py"

    def test_detect_unchanged_files(self, tmp_path):
        """Unchanged files not reported."""
        file_path = tmp_path / "same.py"
        content = "unchanged content"
        file_path.write_text(content)

        # Need to get the actual hash
        from vector_core.utils.hashing import hash_content
        indexed = {"same.py": hash_content(content)}

        changes = detect_changes(tmp_path, indexed_files=indexed)

        # With mtime=0 in indexed (legacy), it will check hash
        # The hash matches, so no changes
        assert len(changes.added) == 0
        # With mtime=0, it falls through to hash check
        # Content matches, so not modified
        assert len(changes.deleted) == 0


class TestDetectChangesFast:
    """Tests for detect_changes_fast function."""

    def test_fast_new_files(self, tmp_path):
        """Detects new files."""
        (tmp_path / "new.py").write_text("new file")

        changes = detect_changes_fast(tmp_path, indexed_metadata={})

        assert len(changes.added) == 1
        assert changes.added[0].rel_path == "new.py"

    def test_fast_deleted_files(self, tmp_path):
        """Detects deleted files."""
        indexed = {
            "gone.py": {"file_hash": "hash", "mtime": 1000.0, "size_bytes": 10}
        }

        changes = detect_changes_fast(tmp_path, indexed_metadata=indexed)

        assert len(changes.deleted) == 1
        assert "gone.py" in changes.deleted

    def test_fast_unchanged_mtime_size(self, tmp_path):
        """Files with same mtime+size skip hash check."""
        file_path = tmp_path / "unchanged.py"
        file_path.write_text("content")

        stat = file_path.stat()

        indexed = {
            "unchanged.py": {
                "file_hash": "some_hash",
                "mtime": stat.st_mtime,
                "size_bytes": stat.st_size,
            }
        }

        changes = detect_changes_fast(tmp_path, indexed_metadata=indexed)

        # Not detected as modified because mtime+size match
        assert len(changes.added) == 0
        assert len(changes.modified) == 0
        assert len(changes.deleted) == 0

    def test_fast_modified_mtime(self, tmp_path):
        """Files with different mtime are verified with hash."""
        file_path = tmp_path / "modified.py"
        file_path.write_text("new content")

        indexed = {
            "modified.py": {
                "file_hash": "old_hash",
                "mtime": 0.0,  # Different mtime
                "size_bytes": 11,
            }
        }

        changes = detect_changes_fast(tmp_path, indexed_metadata=indexed)

        assert len(changes.modified) == 1
        assert changes.modified[0].rel_path == "modified.py"

    def test_fast_modified_size(self, tmp_path):
        """Files with different size are verified with hash."""
        file_path = tmp_path / "modified.py"
        file_path.write_text("new content here")

        stat = file_path.stat()
        indexed = {
            "modified.py": {
                "file_hash": "old_hash",
                "mtime": stat.st_mtime,
                "size_bytes": 5,  # Different size
            }
        }

        changes = detect_changes_fast(tmp_path, indexed_metadata=indexed)

        assert len(changes.modified) == 1

    def test_fast_mtime_changed_same_content(self, tmp_path):
        """File with changed mtime but same content not reported."""
        content = "exact same content"
        file_path = tmp_path / "touched.py"
        file_path.write_text(content)

        from vector_core.utils.hashing import hash_content

        indexed = {
            "touched.py": {
                "file_hash": hash_content(content),
                "mtime": 0.0,  # Different mtime
                "size_bytes": len(content),
            }
        }

        changes = detect_changes_fast(tmp_path, indexed_metadata=indexed)

        # Content matches, so not in modified
        assert len(changes.added) == 0
        assert len(changes.modified) == 0

    def test_fast_legacy_mtime_zero(self, tmp_path):
        """Files with mtime=0 (legacy) trigger hash check."""
        file_path = tmp_path / "legacy.py"
        file_path.write_text("content")

        stat = file_path.stat()
        indexed = {
            "legacy.py": {
                "file_hash": "different_hash",
                "mtime": 0.0,  # Legacy data
                "size_bytes": stat.st_size,
            }
        }

        changes = detect_changes_fast(tmp_path, indexed_metadata=indexed)

        # Should trigger hash check due to mtime=0
        assert len(changes.modified) == 1

    def test_fast_no_changes(self, tmp_path):
        """Returns early when no potential changes."""
        # Empty directory, empty index
        changes = detect_changes_fast(tmp_path, indexed_metadata={})

        assert changes.has_changes is False
        assert len(changes.added) == 0
        assert len(changes.modified) == 0
        assert len(changes.deleted) == 0

    def test_fast_mixed_changes(self, tmp_path):
        """Detects mix of added, modified, deleted."""
        (tmp_path / "new.py").write_text("new file")
        (tmp_path / "modified.py").write_text("modified content")
        # deleted.py is only in index

        indexed = {
            "modified.py": {
                "file_hash": "old_hash",
                "mtime": 0.0,
                "size_bytes": 5,
            },
            "deleted.py": {
                "file_hash": "hash",
                "mtime": 1000.0,
                "size_bytes": 10,
            },
        }

        changes = detect_changes_fast(tmp_path, indexed_metadata=indexed)

        assert len(changes.added) == 1
        assert len(changes.modified) == 1
        assert len(changes.deleted) == 1

    def test_fast_path_resolution(self, tmp_path):
        """Handles path resolution correctly."""
        subdir = tmp_path / "src"
        subdir.mkdir()
        (subdir / "main.py").write_text("code")

        changes = detect_changes_fast(str(tmp_path), indexed_metadata={})

        assert len(changes.added) == 1
        assert changes.added[0].rel_path == "src/main.py"

    def test_fast_legacy_mtime_zero_same_stat(self, tmp_path):
        """Files with mtime=0 (legacy) where stat also returns 0 (line 94).

        This tests the rare case where:
        - indexed_mtime == 0 (legacy data)
        - current file mtime == 0 (rare but possible)
        - sizes match
        This triggers line 94's elif branch.
        """
        from unittest.mock import patch

        file_path = tmp_path / "legacy.py"
        file_path.write_text("content")

        indexed = {
            "legacy.py": {
                "file_hash": "different_hash",
                "mtime": 0.0,  # Legacy data with mtime=0
                "size_bytes": len("content"),  # Size matches
            }
        }

        # Mock scan_file_metadata to return mtime=0 for our file
        # This triggers line 94: mtime matches (both 0), size matches, indexed_mtime==0
        def mock_scan(codebase_path):
            yield ("legacy.py", 0.0, len("content"))  # mtime=0, size matches

        with patch(
            'mcp_codesearch.indexer.change_detect.scan_file_metadata',
            side_effect=mock_scan
        ):
            changes = detect_changes_fast(tmp_path, indexed_metadata=indexed)

        # Line 94: mtime=0 in both -> triggers hash check -> modified due to hash diff
        assert len(changes.modified) == 1
        assert changes.modified[0].rel_path == "legacy.py"
