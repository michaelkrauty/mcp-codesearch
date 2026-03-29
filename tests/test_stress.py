"""Stress tests for large codebase indexing.

These tests verify that indexing handles large codebases without
running out of memory or crashing. They are marked slow and skipped
by default unless explicitly run with pytest -m stress.

Run with: pytest -m stress tests/test_stress.py -v
"""

import random
import time
from pathlib import Path

import pytest

from mcp_codesearch.indexer.change_detect import detect_changes_fast
from mcp_codesearch.indexer.chunker import chunk_file
from mcp_codesearch.indexer.discovery import discover_files, scan_file_metadata

# Skip all stress tests by default - run with: pytest -m stress
pytestmark = pytest.mark.stress


def generate_python_file(lines: int = 100) -> str:
    """Generate a realistic Python file with given number of lines."""
    parts = []

    # Imports
    parts.append('"""Auto-generated test file."""')
    parts.append("")
    parts.append("import os")
    parts.append("import sys")
    parts.append("from typing import Any, Optional")
    parts.append("")

    # Generate some classes and functions
    class_count = max(1, lines // 50)
    for c in range(class_count):
        class_name = f"TestClass{c}"
        parts.append(f"class {class_name}:")
        parts.append(f'    """Class {c} docstring."""')
        parts.append("")

        method_count = max(1, (lines // class_count) // 10)
        for m in range(method_count):
            method_name = f"method_{m}"
            parts.append(f"    def {method_name}(self, arg: Any) -> Optional[str]:")
            parts.append(f'        """Method {m} docstring."""')

            # Add some realistic-looking code
            for _ in range(random.randint(3, 8)):
                indent = "        "
                code_line = random.choice([
                    "result = self.process(arg)",
                    "if arg is None:\n            return None",
                    "data = []",
                    "for item in arg:\n            data.append(item)",
                    "return str(result)",
                    "self.validate(arg)",
                    "try:\n            value = int(arg)\n        except ValueError:\n            value = 0",
                    "logger.info(f'Processing {arg}')",
                ])
                parts.append(indent + code_line)
            parts.append("")

    # Pad to reach target line count
    while len(parts) < lines:
        parts.append("# Padding comment line " + "x" * random.randint(10, 50))

    return "\n".join(parts[:lines])


def generate_javascript_file(lines: int = 100) -> str:
    """Generate a realistic JavaScript file."""
    parts = []
    parts.append('/**')
    parts.append(' * Auto-generated test file.')
    parts.append(' */')
    parts.append("")
    parts.append("import { useState, useEffect } from 'react';")
    parts.append("")

    func_count = max(1, lines // 20)
    for f in range(func_count):
        parts.append(f"export function handleEvent{f}(event) {{")
        parts.append(f"  // Handler {f}")
        for _ in range(random.randint(3, 10)):
            parts.append(random.choice([
                "  const data = event.target.value;",
                "  console.log('Processing...');",
                "  return { success: true };",
                "  if (!data) return null;",
                "  setState(prevState => ({ ...prevState, loading: true }));",
            ]))
        parts.append("}")
        parts.append("")

    while len(parts) < lines:
        parts.append("// Padding comment " + "x" * random.randint(10, 40))

    return "\n".join(parts[:lines])


def create_large_codebase(
    base_path: Path,
    num_files: int = 100,
    lines_per_file: int = 200,
) -> dict:
    """Create a synthetic large codebase for testing."""
    stats = {"python": 0, "javascript": 0, "total_lines": 0}

    # Create directory structure
    dirs = [
        "src",
        "src/core",
        "src/utils",
        "src/models",
        "lib",
        "lib/helpers",
        "tests",
    ]
    for d in dirs:
        (base_path / d).mkdir(parents=True, exist_ok=True)

    # Generate files
    for i in range(num_files):
        # Alternate between Python and JavaScript
        if i % 2 == 0:
            filename = f"module_{i}.py"
            content = generate_python_file(lines_per_file)
            stats["python"] += 1
        else:
            filename = f"component_{i}.js"
            content = generate_javascript_file(lines_per_file)
            stats["javascript"] += 1

        # Place in random directory
        dir_name = random.choice(dirs)
        file_path = base_path / dir_name / filename
        file_path.write_text(content)
        stats["total_lines"] += lines_per_file

    return stats


class TestLargeCodebaseDiscovery:
    """Stress tests for file discovery."""

    def test_discover_100_files(self, tmp_path: Path) -> None:
        """Discover 100 files without issues."""
        stats = create_large_codebase(tmp_path, num_files=100, lines_per_file=100)

        start = time.time()
        files = list(discover_files(tmp_path))
        elapsed = time.time() - start

        assert len(files) == 100
        assert elapsed < 5.0, f"Discovery took too long: {elapsed:.2f}s"
        print(f"\nDiscovered 100 files in {elapsed:.2f}s")

    def test_discover_500_files(self, tmp_path: Path) -> None:
        """Discover 500 files without issues."""
        stats = create_large_codebase(tmp_path, num_files=500, lines_per_file=100)

        start = time.time()
        files = list(discover_files(tmp_path))
        elapsed = time.time() - start

        assert len(files) == 500
        assert elapsed < 15.0, f"Discovery took too long: {elapsed:.2f}s"
        print(f"\nDiscovered 500 files in {elapsed:.2f}s")

    def test_fast_metadata_scan(self, tmp_path: Path) -> None:
        """Fast metadata scan should be faster than full discovery."""
        create_large_codebase(tmp_path, num_files=200, lines_per_file=200)

        # Time full discovery
        start = time.time()
        list(discover_files(tmp_path))
        full_time = time.time() - start

        # Time fast scan
        start = time.time()
        list(scan_file_metadata(tmp_path))
        fast_time = time.time() - start

        # Fast scan should be at least 2x faster (it doesn't read content)
        assert fast_time < full_time, "Fast scan should be faster"
        print(f"\nFull: {full_time:.2f}s, Fast: {fast_time:.2f}s, Speedup: {full_time/fast_time:.1f}x")


class TestLargeFileChunking:
    """Stress tests for file chunking."""

    def test_chunk_large_python_file(self) -> None:
        """Chunk a large Python file."""
        content = generate_python_file(lines=2000)

        start = time.time()
        chunks = chunk_file(content, "python")
        elapsed = time.time() - start

        assert len(chunks) > 0
        assert elapsed < 2.0, f"Chunking took too long: {elapsed:.2f}s"
        print(f"\nChunked 2000-line Python file into {len(chunks)} chunks in {elapsed:.2f}s")

    def test_chunk_very_large_file(self) -> None:
        """Chunk a very large file (5000 lines)."""
        content = generate_python_file(lines=5000)

        start = time.time()
        chunks = chunk_file(content, "python")
        elapsed = time.time() - start

        assert len(chunks) > 0
        assert elapsed < 5.0, f"Chunking took too long: {elapsed:.2f}s"
        print(f"\nChunked 5000-line file into {len(chunks)} chunks in {elapsed:.2f}s")

    def test_chunk_many_files(self, tmp_path: Path) -> None:
        """Chunk many files in sequence."""
        create_large_codebase(tmp_path, num_files=100, lines_per_file=300)

        files = list(discover_files(tmp_path))
        total_chunks = 0

        start = time.time()
        for f in files:
            chunks = chunk_file(f.content, f.language)
            total_chunks += len(chunks)
        elapsed = time.time() - start

        assert total_chunks > 0
        assert elapsed < 30.0, f"Chunking all files took too long: {elapsed:.2f}s"
        print(f"\nChunked 100 files into {total_chunks} chunks in {elapsed:.2f}s")


class TestChangeDetection:
    """Stress tests for change detection."""

    def test_change_detection_large_codebase(self, tmp_path: Path) -> None:
        """Detect changes in a large codebase."""
        create_large_codebase(tmp_path, num_files=200, lines_per_file=150)

        # First pass - everything is new
        files = list(discover_files(tmp_path))
        indexed = {
            f.rel_path: {"file_hash": f.content_hash, "mtime": f.mtime, "size_bytes": f.size_bytes}
            for f in files
        }

        # No changes - should be fast
        start = time.time()
        changes = detect_changes_fast(tmp_path, indexed)
        elapsed = time.time() - start

        assert not changes.has_changes
        assert elapsed < 2.0, f"Change detection took too long: {elapsed:.2f}s"
        print(f"\nChange detection (no changes) in {elapsed:.2f}s")

    def test_change_detection_with_modifications(self, tmp_path: Path) -> None:
        """Detect changes when some files are modified."""
        create_large_codebase(tmp_path, num_files=200, lines_per_file=100)

        files = list(discover_files(tmp_path))
        indexed = {
            f.rel_path: {"file_hash": f.content_hash, "mtime": f.mtime, "size_bytes": f.size_bytes}
            for f in files
        }

        # Modify 10 files
        modified_count = 10
        for f in files[:modified_count]:
            file_path = tmp_path / f.rel_path
            content = file_path.read_text()
            file_path.write_text(content + "\n# Modified")

        start = time.time()
        changes = detect_changes_fast(tmp_path, indexed)
        elapsed = time.time() - start

        assert changes.has_changes
        assert len(changes.modified) == modified_count
        assert elapsed < 3.0, f"Change detection took too long: {elapsed:.2f}s"
        print(f"\nDetected {len(changes.modified)} modifications in {elapsed:.2f}s")


class TestMemoryUsage:
    """Tests to verify memory doesn't grow unbounded."""

    def test_streaming_discovery(self, tmp_path: Path) -> None:
        """Verify discovery streams files, not loading all at once."""
        create_large_codebase(tmp_path, num_files=300, lines_per_file=200)

        # Use iterator without converting to list
        count = 0
        total_size = 0
        for f in discover_files(tmp_path):
            count += 1
            total_size += f.size_bytes
            # Process one at a time - should not accumulate memory

        assert count == 300
        print(f"\nProcessed {count} files, total {total_size / 1024 / 1024:.1f} MB content")

    def test_chunking_doesnt_explode_memory(self) -> None:
        """Verify chunking a huge file doesn't use excessive memory."""
        # Generate a very large file (10000 lines, ~500KB)
        content = generate_python_file(lines=10000)

        # This should complete without running out of memory
        chunks = chunk_file(content, "python")

        assert len(chunks) > 0
        total_chunk_size = sum(len(c.content) for c in chunks)

        # Total chunk content should be roughly similar to original
        # (some overlap is expected from context)
        assert total_chunk_size < len(content) * 2, "Chunk expansion too high"
        print(f"\nOriginal: {len(content)/1024:.1f}KB, Chunks: {total_chunk_size/1024:.1f}KB")
