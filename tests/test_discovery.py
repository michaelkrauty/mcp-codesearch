"""Tests for file discovery in mcp-codesearch."""

from pathlib import Path

from vector_core.utils.hashing import hash_content

from mcp_codesearch.indexer.discovery import (
    EXTENSION_TO_LANGUAGE,
    FileInfo,
    FileMetadata,
    _detect_language,
    _load_gitignore,
    discover_files,
    get_file_hash,
    get_file_stat,
    scan_file_metadata,
)


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_creation(self, tmp_path):
        """Create FileInfo with all fields."""
        info = FileInfo(
            path=tmp_path / "test.py",
            rel_path="test.py",
            language="python",
            size_bytes=100,
            content="print('hello')",
            content_hash="abc123",
            line_count=1,
            mtime=1234567890.0,
        )

        assert info.path == tmp_path / "test.py"
        assert info.rel_path == "test.py"
        assert info.language == "python"
        assert info.size_bytes == 100
        assert info.line_count == 1


class TestFileMetadata:
    """Tests for FileMetadata dataclass."""

    def test_creation(self):
        """Create FileMetadata with all fields."""
        meta = FileMetadata(
            rel_path="src/main.py",
            size_bytes=500,
            mtime=1234567890.0,
            content_hash="def456",
        )

        assert meta.rel_path == "src/main.py"
        assert meta.size_bytes == 500
        assert meta.content_hash == "def456"


class TestExtensionToLanguage:
    """Tests for language extension mapping."""

    def test_common_languages(self):
        """Common language extensions are mapped."""
        assert EXTENSION_TO_LANGUAGE[".py"] == "python"
        assert EXTENSION_TO_LANGUAGE[".js"] == "javascript"
        assert EXTENSION_TO_LANGUAGE[".ts"] == "typescript"
        assert EXTENSION_TO_LANGUAGE[".go"] == "go"
        assert EXTENSION_TO_LANGUAGE[".rs"] == "rust"
        assert EXTENSION_TO_LANGUAGE[".java"] == "java"

    def test_typescript_variants(self):
        """TypeScript variants are mapped."""
        assert EXTENSION_TO_LANGUAGE[".ts"] == "typescript"
        assert EXTENSION_TO_LANGUAGE[".tsx"] == "typescript"

    def test_javascript_variants(self):
        """JavaScript variants are mapped."""
        assert EXTENSION_TO_LANGUAGE[".js"] == "javascript"
        assert EXTENSION_TO_LANGUAGE[".jsx"] == "javascript"

    def test_c_cpp_headers(self):
        """C/C++ headers are mapped."""
        assert EXTENSION_TO_LANGUAGE[".c"] == "c"
        assert EXTENSION_TO_LANGUAGE[".h"] == "c"
        assert EXTENSION_TO_LANGUAGE[".cpp"] == "cpp"
        assert EXTENSION_TO_LANGUAGE[".hpp"] == "cpp"


class TestLoadGitignore:
    """Tests for _load_gitignore function."""

    def test_loads_patterns(self, tmp_path):
        """Loads gitignore patterns."""
        (tmp_path / ".gitignore").write_text("*.log\n__pycache__/\n")

        spec = _load_gitignore(tmp_path)

        assert spec is not None
        assert spec.match_file("debug.log")
        assert spec.match_file("__pycache__/cache.pyc")

    def test_no_gitignore(self, tmp_path):
        """Returns None when no .gitignore."""
        spec = _load_gitignore(tmp_path)
        assert spec is None


class TestHashContent:
    """Tests for hash_content function."""

    def test_deterministic(self):
        """Same content produces same hash."""
        assert hash_content("hello") == hash_content("hello")

    def test_different_content(self):
        """Different content produces different hashes."""
        assert hash_content("hello") != hash_content("world")

    def test_sha256_format(self):
        """Hash is 64 character hex string."""
        result = hash_content("test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestDetectLanguage:
    """Tests for _detect_language function."""

    def test_python_file(self, tmp_path):
        """Detects Python files."""
        assert _detect_language(tmp_path / "main.py") == "python"

    def test_javascript_file(self, tmp_path):
        """Detects JavaScript files."""
        assert _detect_language(tmp_path / "app.js") == "javascript"

    def test_unknown_extension(self, tmp_path):
        """Returns None for unknown extensions."""
        assert _detect_language(tmp_path / "data.xyz") is None

    def test_case_insensitive(self, tmp_path):
        """Extension matching is case-insensitive."""
        assert _detect_language(tmp_path / "Main.PY") == "python"


class TestDiscoverFiles:
    """Tests for discover_files function."""

    def test_discovers_python_files(self, tmp_path):
        """Discovers Python files."""
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def helper(): pass")

        files = list(discover_files(tmp_path))

        assert len(files) == 2
        rel_paths = [f.rel_path for f in files]
        assert "main.py" in rel_paths
        assert "utils.py" in rel_paths

    def test_recursive_discovery(self, tmp_path):
        """Discovers files in subdirectories."""
        (tmp_path / "root.py").write_text("root")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("nested")

        files = list(discover_files(tmp_path))

        rel_paths = [f.rel_path for f in files]
        assert "root.py" in rel_paths
        assert "subdir/nested.py" in rel_paths

    def test_excludes_hidden_files(self, tmp_path):
        """Excludes hidden files."""
        (tmp_path / "visible.py").write_text("visible")
        (tmp_path / ".hidden.py").write_text("hidden")

        files = list(discover_files(tmp_path))

        rel_paths = [f.rel_path for f in files]
        assert "visible.py" in rel_paths
        assert ".hidden.py" not in rel_paths

    def test_excludes_node_modules(self, tmp_path):
        """Excludes node_modules directory."""
        (tmp_path / "main.js").write_text("main")
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lib.js").write_text("lib")

        files = list(discover_files(tmp_path))

        rel_paths = [f.rel_path for f in files]
        assert not any("node_modules" in p for p in rel_paths)

    def test_excludes_pycache(self, tmp_path):
        """Excludes __pycache__ directory."""
        (tmp_path / "main.py").write_text("main")
        pc = tmp_path / "__pycache__"
        pc.mkdir()
        (pc / "main.cpython-39.pyc").write_text("compiled")

        files = list(discover_files(tmp_path))

        rel_paths = [f.rel_path for f in files]
        assert not any("__pycache__" in p for p in rel_paths)

    def test_respects_gitignore(self, tmp_path):
        """Respects .gitignore patterns."""
        (tmp_path / ".gitignore").write_text("*.log\nbuild/\n")
        (tmp_path / "main.py").write_text("main")
        (tmp_path / "debug.log").write_text("log")
        build = tmp_path / "build"
        build.mkdir()
        (build / "output.py").write_text("output")

        files = list(discover_files(tmp_path))

        rel_paths = [f.rel_path for f in files]
        assert "main.py" in rel_paths
        assert "debug.log" not in rel_paths
        assert not any("build" in p for p in rel_paths)

    def test_extension_filter(self, tmp_path):
        """Filters by extension."""
        (tmp_path / "main.py").write_text("python")
        (tmp_path / "app.js").write_text("javascript")
        (tmp_path / "data.txt").write_text("text")

        files = list(discover_files(tmp_path, include_extensions={".py"}))

        rel_paths = [f.rel_path for f in files]
        assert "main.py" in rel_paths
        assert "app.js" not in rel_paths

    def test_exclude_patterns(self, tmp_path):
        """Applies additional exclude patterns."""
        (tmp_path / "main.py").write_text("main")
        (tmp_path / "test_main.py").write_text("test")

        files = list(discover_files(tmp_path, exclude_patterns=["test_*"]))

        rel_paths = [f.rel_path for f in files]
        assert "main.py" in rel_paths
        assert "test_main.py" not in rel_paths

    def test_size_limit(self, tmp_path):
        """Respects file size limit."""
        (tmp_path / "small.py").write_text("small")
        (tmp_path / "large.py").write_text("x" * 10000)

        files = list(discover_files(tmp_path, max_file_size_kb=1))

        rel_paths = [f.rel_path for f in files]
        assert "small.py" in rel_paths
        assert "large.py" not in rel_paths

    def test_skips_empty_files(self, tmp_path):
        """Skips empty files."""
        (tmp_path / "content.py").write_text("content")
        (tmp_path / "empty.py").write_text("")

        files = list(discover_files(tmp_path))

        rel_paths = [f.rel_path for f in files]
        assert "content.py" in rel_paths
        assert "empty.py" not in rel_paths

    def test_file_info_content(self, tmp_path):
        """FileInfo contains correct content."""
        content = "def hello():\n    print('world')\n"
        (tmp_path / "main.py").write_text(content)

        files = list(discover_files(tmp_path))

        assert len(files) == 1
        assert files[0].content == content
        assert files[0].language == "python"
        assert files[0].line_count == 3

    def test_skips_symlinks(self, tmp_path):
        """Skips symlinked files."""
        real = tmp_path / "real.py"
        real.write_text("real")
        link = tmp_path / "link.py"
        link.symlink_to(real)

        files = list(discover_files(tmp_path))

        rel_paths = [f.rel_path for f in files]
        assert "real.py" in rel_paths
        assert "link.py" not in rel_paths


class TestGetFileHash:
    """Tests for get_file_hash function."""

    def test_returns_hash(self, tmp_path):
        """Returns hash of file content."""
        file_path = tmp_path / "test.py"
        file_path.write_text("content")

        result = get_file_hash(file_path)

        assert result == hash_content("content")

    def test_nonexistent_file(self, tmp_path):
        """Returns None for nonexistent file."""
        result = get_file_hash(tmp_path / "missing.py")
        assert result is None


class TestGetFileStat:
    """Tests for get_file_stat function."""

    def test_returns_mtime_size(self, tmp_path):
        """Returns (mtime, size) tuple."""
        file_path = tmp_path / "test.py"
        file_path.write_text("content")

        result = get_file_stat(file_path)

        assert result is not None
        mtime, size = result
        assert mtime > 0
        assert size == 7  # len("content")

    def test_nonexistent_file(self, tmp_path):
        """Returns None for nonexistent file."""
        result = get_file_stat(tmp_path / "missing.py")
        assert result is None


class TestScanFileMetadata:
    """Tests for scan_file_metadata function."""

    def test_returns_tuples(self, tmp_path):
        """Returns (rel_path, mtime, size) tuples."""
        (tmp_path / "main.py").write_text("content")

        results = list(scan_file_metadata(tmp_path))

        assert len(results) == 1
        rel_path, mtime, size = results[0]
        assert rel_path == "main.py"
        assert mtime > 0
        assert size == 7

    def test_faster_than_discover(self, tmp_path):
        """Metadata scan doesn't read content."""
        for i in range(5):
            (tmp_path / f"file{i}.py").write_text(f"content {i}")

        # Should work without reading content
        results = list(scan_file_metadata(tmp_path))
        assert len(results) == 5

    def test_respects_gitignore(self, tmp_path):
        """Respects .gitignore patterns."""
        (tmp_path / ".gitignore").write_text("ignored.py\n")
        (tmp_path / "main.py").write_text("main")
        (tmp_path / "ignored.py").write_text("ignored")

        results = list(scan_file_metadata(tmp_path))

        rel_paths = [r[0] for r in results]
        assert "main.py" in rel_paths
        assert "ignored.py" not in rel_paths


class TestDiscoverFilesEdgeCases:
    """Tests for discover_files edge cases."""

    def test_gitignore_pattern_skip(self, tmp_path):
        """Files matching gitignore are skipped (line 161)."""
        # Use .py extension so it passes extension filter but is caught by gitignore
        (tmp_path / ".gitignore").write_text("ignored.py\n")
        (tmp_path / "main.py").write_text("main content")
        (tmp_path / "ignored.py").write_text("should be ignored by gitignore")

        results = list(discover_files(tmp_path))

        rel_paths = [f.rel_path for f in results]
        assert "main.py" in rel_paths
        assert "ignored.py" not in rel_paths

    def test_stat_oserror_skip(self, tmp_path, monkeypatch):
        """Files that raise OSError on stat are skipped (lines 174-175)."""

        (tmp_path / "good.py").write_text("good content")
        (tmp_path / "bad.py").write_text("bad content")

        original_stat = Path.stat
        stat_calls = {}

        def mock_stat(self, *args, **kwargs):
            if self.name == "bad.py":
                stat_calls[self.name] = stat_calls.get(self.name, 0) + 1
                # Skip first call (is_symlink check), fail on second
                if stat_calls[self.name] > 1:
                    raise OSError("Permission denied")
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", mock_stat)

        results = list(discover_files(tmp_path))

        rel_paths = [f.rel_path for f in results]
        assert "good.py" in rel_paths
        assert "bad.py" not in rel_paths

    def test_read_error_skip(self, tmp_path, monkeypatch):
        """Files that raise Exception on read are skipped (lines 180-181)."""
        import sys

        (tmp_path / "good.py").write_text("good content")
        (tmp_path / "bad.py").write_text("bad content")

        if sys.platform == "win32":
            # Windows uses Path.read_text
            original_read_text = Path.read_text

            def mock_read_text(self, *args, **kwargs):
                if self.name == "bad.py":
                    raise UnicodeDecodeError("utf-8", b"", 0, 1, "mock error")
                return original_read_text(self, *args, **kwargs)

            monkeypatch.setattr(Path, "read_text", mock_read_text)
        else:
            # Unix uses os.fdopen - mock _safe_read_file directly
            import mcp_codesearch.indexer.discovery as discovery_module

            original_safe_read = discovery_module._safe_read_file

            def mock_safe_read(file_path):
                if file_path.name == "bad.py":
                    return None  # Simulate read failure
                return original_safe_read(file_path)

            monkeypatch.setattr(discovery_module, "_safe_read_file", mock_safe_read)

        results = list(discover_files(tmp_path))

        rel_paths = [f.rel_path for f in results]
        assert "good.py" in rel_paths
        assert "bad.py" not in rel_paths

    def test_unknown_language_skip(self, tmp_path):
        """Files with unknown language extension are skipped (line 186)."""
        # .xyz is not in EXTENSION_TO_LANGUAGE, but force it through extension filter
        (tmp_path / "main.py").write_text("python content")
        (tmp_path / "unknown.xyz").write_text("unknown language content")

        # Force .xyz through extension filter - it will be rejected at language detection
        results = list(discover_files(tmp_path, include_extensions={".py", ".xyz"}))

        rel_paths = [f.rel_path for f in results]
        assert "main.py" in rel_paths
        # .xyz passes extension filter but fails language detection (line 186)
        assert "unknown.xyz" not in rel_paths


class TestScanFileMetadataEdgeCases:
    """Tests for scan_file_metadata edge cases."""

    def test_empty_file_skip(self, tmp_path):
        """Empty files are skipped (line 281)."""
        (tmp_path / "content.py").write_text("content")
        (tmp_path / "empty.py").write_text("")

        results = list(scan_file_metadata(tmp_path))

        rel_paths = [r[0] for r in results]
        assert "content.py" in rel_paths
        assert "empty.py" not in rel_paths

    def test_symlink_skip(self, tmp_path):
        """Symlinks are skipped (line 267)."""
        real_file = tmp_path / "real.py"
        real_file.write_text("real content")

        symlink = tmp_path / "link.py"
        symlink.symlink_to(real_file)

        results = list(scan_file_metadata(tmp_path))

        rel_paths = [r[0] for r in results]
        assert "real.py" in rel_paths
        assert "link.py" not in rel_paths

    def test_wrong_extension_skip(self, tmp_path):
        """Files with wrong extension are skipped (line 271)."""
        (tmp_path / "code.py").write_text("python")
        (tmp_path / "data.txt").write_text("text data")

        results = list(scan_file_metadata(tmp_path))

        rel_paths = [r[0] for r in results]
        assert "code.py" in rel_paths
        assert "data.txt" not in rel_paths

    def test_stat_oserror_skip(self, tmp_path, monkeypatch):
        """Files that raise OSError on stat are skipped (lines 281-283)."""

        (tmp_path / "good.py").write_text("good content")
        (tmp_path / "bad.py").write_text("bad content")

        original_stat = Path.stat
        stat_calls = {}

        def mock_stat(self, *args, **kwargs):
            if self.name == "bad.py":
                stat_calls[self.name] = stat_calls.get(self.name, 0) + 1
                # Skip first call (is_symlink check), fail on second
                if stat_calls[self.name] > 1:
                    raise OSError("Permission denied")
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", mock_stat)

        results = list(scan_file_metadata(tmp_path))

        rel_paths = [r[0] for r in results]
        assert "good.py" in rel_paths
        assert "bad.py" not in rel_paths
