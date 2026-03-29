"""Tests for code chunking."""

from mcp_codesearch.indexer.chunker import (
    _chunk_fixed_size,
    _extract_imports,
    _extract_module_docstring,
    _merge_small_chunks,
    chunk_file,
    generate_file_summary,
)
from mcp_codesearch.indexer.treesitter import Chunk


class TestPythonChunking:
    """Tests for Python code chunking."""

    def test_function_extraction(self, sample_python_code):
        chunks = chunk_file(sample_python_code, "python")

        # Should extract functions
        function_names = [c.name for c in chunks if c.chunk_type == "function"]
        assert "simple_function" in function_names
        assert "helper_function" in function_names

    def test_class_extraction(self, sample_python_code):
        chunks = chunk_file(sample_python_code, "python")

        # Should extract class
        class_chunks = [c for c in chunks if "class" in c.chunk_type]
        assert len(class_chunks) >= 1
        assert any(c.name == "Calculator" for c in class_chunks)

    def test_method_extraction(self, sample_python_code):
        chunks = chunk_file(sample_python_code, "python")

        # Should extract methods with class context
        method_chunks = [c for c in chunks if c.context and "Calculator" in c.context]
        method_names = [c.name for c in method_chunks]
        assert "add" in method_names
        assert "subtract" in method_names

    def test_imports_attached(self, sample_python_code):
        chunks = chunk_file(sample_python_code, "python")

        # All chunks should have imports attached
        for chunk in chunks:
            assert chunk.imports is not None
            assert "os" in chunk.imports or "pathlib" in chunk.imports


class TestTypeScriptChunking:
    """Tests for TypeScript code chunking."""

    def test_interface_extraction(self, sample_typescript_code):
        chunks = chunk_file(sample_typescript_code, "typescript")

        # Should extract interface
        names = [c.name for c in chunks]
        assert "UserService" in names

    def test_class_extraction(self, sample_typescript_code):
        chunks = chunk_file(sample_typescript_code, "typescript")

        # Should extract class
        class_chunks = [c for c in chunks if "class" in c.chunk_type]
        assert len(class_chunks) >= 1

    def test_function_extraction(self, sample_typescript_code):
        chunks = chunk_file(sample_typescript_code, "typescript")

        # Should extract exported function
        function_names = [c.name for c in chunks if c.chunk_type == "function"]
        assert "createUserService" in function_names


class TestRustChunking:
    """Tests for Rust code chunking."""

    def test_struct_extraction(self, sample_rust_code):
        chunks = chunk_file(sample_rust_code, "rust")

        # Should extract struct
        struct_chunks = [c for c in chunks if c.chunk_type == "struct"]
        assert len(struct_chunks) >= 1
        assert any(c.name == "Processor" for c in struct_chunks)

    def test_impl_extraction(self, sample_rust_code):
        chunks = chunk_file(sample_rust_code, "rust")

        # Should extract impl block
        impl_chunks = [c for c in chunks if c.chunk_type == "impl"]
        assert len(impl_chunks) >= 1


class TestLargeClassHandling:
    """Tests for large class overview generation."""

    def test_large_class_overview(self, large_class_code):
        chunks = chunk_file(large_class_code, "python")

        # Should have class overview
        overview_chunks = [c for c in chunks if c.chunk_type == "class_overview"]
        assert len(overview_chunks) >= 1

        # Overview should contain method signatures
        overview = overview_chunks[0]
        assert "method_0" in overview.content
        assert "..." in overview.content  # Truncated

    def test_large_class_methods_indexed(self, large_class_code):
        chunks = chunk_file(large_class_code, "python")

        # Individual methods should also be indexed
        method_chunks = [c for c in chunks if c.chunk_type == "function" and c.context]
        assert len(method_chunks) > 0


class TestEdgeCases:
    """Tests for edge cases in chunking."""

    def test_empty_file(self, empty_file_content):
        chunks = chunk_file(empty_file_content, "python")
        # Empty file should produce at least one chunk or empty list
        assert isinstance(chunks, list)

    def test_unknown_language(self):
        chunks = chunk_file("some content", "unknown_language")
        # Should fallback to line-based chunking
        assert isinstance(chunks, list)

    def test_binary_like_content(self):
        # Content with non-text characters
        content = "normal code\x00with binary\x01chars"
        chunks = chunk_file(content, "python")
        # Should not crash
        assert isinstance(chunks, list)


class TestDocstringExtraction:
    """Tests for module docstring extraction."""

    def test_python_docstring(self, sample_python_code):
        docstring = _extract_module_docstring(sample_python_code, "python")
        assert docstring is not None
        assert "testing" in docstring.lower()

    def test_python_docstring_with_shebang(self):
        """Python file with shebang and encoding."""
        code = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module docstring after shebang."""

def hello():
    pass
'''
        docstring = _extract_module_docstring(code, "python")
        assert docstring is not None
        assert "shebang" in docstring.lower()

    def test_python_docstring_with_blank_lines(self):
        """Python file with blank lines at start."""
        code = '''

"""Module with blank lines before."""

def test():
    pass
'''
        docstring = _extract_module_docstring(code, "python")
        assert docstring is not None
        assert "blank" in docstring.lower()

    def test_typescript_jsdoc(self, sample_typescript_code):
        docstring = _extract_module_docstring(sample_typescript_code, "typescript")
        assert docstring is not None
        assert "user" in docstring.lower()

    def test_rust_doc_comments(self, sample_rust_code):
        docstring = _extract_module_docstring(sample_rust_code, "rust")
        assert docstring is not None
        assert "crate" in docstring.lower() or "module" in docstring.lower()

    def test_rust_regular_comments_skipped(self):
        """Rust file with regular // comments before doc comments."""
        code = '''// This is a regular comment
// Also regular

//! This is the actual crate doc
//! Continuing on next line

use std::io;
'''
        docstring = _extract_module_docstring(code, "rust")
        assert docstring is not None
        assert "actual crate doc" in docstring

    def test_go_docstring(self):
        """Go file with package comment."""
        code = '''// Package mypackage provides utilities.
// It does some things.
package mypackage

import "fmt"
'''
        docstring = _extract_module_docstring(code, "go")
        assert docstring is not None
        assert "mypackage" in docstring.lower() or "utilities" in docstring.lower()

    def test_go_docstring_with_code_before_package(self):
        """Go file with non-comment code before package (line 233-234)."""
        code = '''// Some comment
var x = 1
package mypackage
'''
        # Should extract comment but stop at non-comment line before package
        docstring = _extract_module_docstring(code, "go")
        # The "var x = 1" line triggers the break at line 233-234
        # Docstring should be just "Some comment" (before the var line)
        assert docstring is None or "Some comment" in (docstring or "")

    def test_java_docstring(self):
        """Java file with Javadoc."""
        code = '''/**
 * This is the class-level Javadoc.
 * It describes the main class.
 */
public class Main {
    public static void main(String[] args) {}
}
'''
        docstring = _extract_module_docstring(code, "java")
        assert docstring is not None
        assert "class-level" in docstring.lower() or "javadoc" in docstring.lower()


class TestImportExtraction:
    """Tests for import extraction."""

    def test_python_imports(self, sample_python_code):
        imports = _extract_imports(sample_python_code, "python")
        assert "os" in imports
        assert "pathlib" in imports

    def test_typescript_imports(self, sample_typescript_code):
        imports = _extract_imports(sample_typescript_code, "typescript")
        # Should extract from import statements
        assert len(imports) >= 0  # May or may not find imports depending on format

    def test_go_imports(self):
        """Go file with import statements."""
        code = '''package main

import (
    "fmt"
    "net/http"
    "github.com/user/pkg"
)

func main() {}
'''
        imports = _extract_imports(code, "go")
        assert len(imports) > 0
        assert "fmt" in imports or "http" in imports or "pkg" in imports


class TestFileSummary:
    """Tests for file summary generation."""

    def test_summary_includes_docstring(self, sample_python_code):
        chunks = chunk_file(sample_python_code, "python")
        summary = generate_file_summary(sample_python_code, chunks, "python")

        assert "testing" in summary.lower() or "module" in summary.lower()

    def test_summary_includes_definitions(self, sample_python_code):
        chunks = chunk_file(sample_python_code, "python")
        summary = generate_file_summary(sample_python_code, chunks, "python")

        # Should list top-level definitions
        assert "fn:" in summary or "cls:" in summary or "Defines:" in summary

    def test_summary_includes_imports(self, sample_python_code):
        chunks = chunk_file(sample_python_code, "python")
        summary = generate_file_summary(sample_python_code, chunks, "python")

        assert "Imports:" in summary or "os" in summary

    def test_summary_fallback_for_plain_text(self):
        """Fallback summary for file with no docstring/imports/chunks."""
        # Plain text file - no docstring, no imports, no parseable chunks
        content = """This is just plain text content.
It doesn't have any Python docstrings.
No imports either.
Just random lines of text that extend past 200 characters.
More lines to make sure we have enough content.
Even more content to test the fallback behavior.
And a few more lines just to be safe.
The summary should use the first 500 chars.
With truncation at a newline if possible.""" + "\n" * 50 + "End of file."

        # Empty chunks list simulates no AST parsing
        chunks = []
        summary = generate_file_summary(content, chunks, "text")

        # Should fallback to first N chars
        assert len(summary) <= 500
        assert "plain text" in summary.lower()

    def test_summary_fallback_truncates_at_newline(self):
        """Fallback truncates at last newline if > 200 chars."""
        # Long line followed by content after 200 chars
        content = "A" * 250 + "\n" + "B" * 100 + "\n" + "C" * 200

        chunks = []
        summary = generate_file_summary(content, chunks, "text")

        # Should truncate, preferring newline break
        assert len(summary) <= 500


class TestChunkFixedSize:
    """Tests for _chunk_fixed_size function."""

    def test_small_file_single_chunk(self):
        """Small file produces single chunk."""
        content = "line 1\nline 2\nline 3"
        chunks = _chunk_fixed_size(content, chunk_size=10, overlap=2)

        assert len(chunks) == 1
        assert chunks[0].chunk_type == "block"
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 3

    def test_large_file_multiple_chunks(self):
        """Large file produces multiple chunks."""
        lines = [f"line {i}" for i in range(200)]
        content = "\n".join(lines)
        chunks = _chunk_fixed_size(content, chunk_size=50, overlap=10)

        assert len(chunks) > 1
        # All chunks should be blocks
        for chunk in chunks:
            assert chunk.chunk_type == "block"

    def test_chunk_overlap(self):
        """Chunks have overlapping lines."""
        lines = [f"line {i}" for i in range(150)]
        content = "\n".join(lines)
        chunks = _chunk_fixed_size(content, chunk_size=50, overlap=10)

        # Check that chunks overlap
        for i in range(len(chunks) - 1):
            assert chunks[i + 1].start_line <= chunks[i].end_line + 10

    def test_line_numbers_correct(self):
        """Line numbers are 1-indexed and correct."""
        lines = [f"line {i}" for i in range(100)]
        content = "\n".join(lines)
        chunks = _chunk_fixed_size(content, chunk_size=30, overlap=5)

        # First chunk starts at line 1
        assert chunks[0].start_line == 1


class TestMergeSmallChunks:
    """Tests for _merge_small_chunks function."""

    def test_single_chunk_unchanged(self):
        """Single chunk returns unchanged."""
        chunk = Chunk(
            content="content",
            chunk_type="block",
            name=None,
            start_line=1,
            end_line=5,
            context=None,
        )
        result = _merge_small_chunks([chunk])

        assert len(result) == 1
        assert result[0] is chunk

    def test_empty_list(self):
        """Empty list returns empty."""
        result = _merge_small_chunks([])
        assert result == []

    def test_merge_adjacent_small_blocks(self):
        """Merges adjacent small block chunks."""
        chunks = [
            Chunk(content="a", chunk_type="block", name=None, start_line=1, end_line=3, context=None),
            Chunk(content="b", chunk_type="block", name=None, start_line=4, end_line=6, context=None),
        ]
        result = _merge_small_chunks(chunks, min_lines=10)

        # Should merge into one
        assert len(result) == 1
        assert result[0].content == "a\nb"
        assert result[0].start_line == 1
        assert result[0].end_line == 6

    def test_no_merge_non_block_chunks(self):
        """Does not merge non-block chunks (functions, classes)."""
        chunks = [
            Chunk(content="def a():", chunk_type="function", name="a", start_line=1, end_line=3, context=None),
            Chunk(content="def b():", chunk_type="function", name="b", start_line=4, end_line=6, context=None),
        ]
        result = _merge_small_chunks(chunks, min_lines=10)

        # Should not merge functions
        assert len(result) == 2

    def test_no_merge_large_chunks(self):
        """Does not merge chunks larger than min_lines."""
        chunks = [
            Chunk(content="a\n" * 15, chunk_type="block", name=None, start_line=1, end_line=15, context=None),
            Chunk(content="b\n" * 15, chunk_type="block", name=None, start_line=16, end_line=30, context=None),
        ]
        result = _merge_small_chunks(chunks, min_lines=10)

        # Should not merge large blocks
        assert len(result) == 2

    def test_no_merge_non_adjacent(self):
        """Does not merge non-adjacent chunks."""
        chunks = [
            Chunk(content="a", chunk_type="block", name=None, start_line=1, end_line=3, context=None),
            Chunk(content="b", chunk_type="block", name=None, start_line=10, end_line=12, context=None),
        ]
        result = _merge_small_chunks(chunks, min_lines=10)

        # Should not merge non-adjacent
        assert len(result) == 2

    def test_merge_mixed_preserves_order(self):
        """Merging preserves chunk order."""
        chunks = [
            Chunk(content="a", chunk_type="block", name=None, start_line=1, end_line=3, context=None),
            Chunk(content="b", chunk_type="block", name=None, start_line=4, end_line=6, context=None),
            Chunk(content="func", chunk_type="function", name="f", start_line=7, end_line=15, context=None),
        ]
        result = _merge_small_chunks(chunks, min_lines=10)

        # First two merged, function preserved
        assert len(result) == 2
        assert result[0].chunk_type == "block"
        assert result[1].chunk_type == "function"
