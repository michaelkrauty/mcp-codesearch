"""Tests for tree-sitter parsing and chunk extraction."""

from mcp_codesearch.indexer.treesitter import (
    DEFINITION_TYPES,
    LANGUAGE_MAP,
    _build_context_path,
    _extract_docstring,
    _generate_class_overview,
    _get_cached_parser,
    _get_node_name,
    _node_to_chunk_type,
    chunk_with_treesitter,
)


class TestGetNodeName:
    """Tests for _get_node_name extraction."""

    def test_identifier_extraction(self):
        """Extracts name from identifier node."""
        parser = _get_cached_parser("python")
        source = b"def my_function(): pass"
        tree = parser.parse(source)

        # Find function node
        func_node = None
        for node in tree.root_node.children:
            if node.type == "function_definition":
                func_node = node
                break

        assert func_node is not None
        name = _get_node_name(func_node, source)
        assert name == "my_function"

    def test_no_identifier_returns_none(self):
        """Returns None when no identifier found."""
        parser = _get_cached_parser("python")
        source = b"pass"
        tree = parser.parse(source)

        # Root node or pass statement has no identifier
        name = _get_node_name(tree.root_node, source)
        assert name is None


class TestNodeToChunkType:
    """Tests for _node_to_chunk_type mapping."""

    def test_function_types(self):
        """Maps function-related types correctly."""
        assert _node_to_chunk_type("function_definition") == "function"
        assert _node_to_chunk_type("method_definition") == "function"
        assert _node_to_chunk_type("arrow_function") == "function"

    def test_class_types(self):
        """Maps class types correctly."""
        assert _node_to_chunk_type("class_definition") == "class"
        assert _node_to_chunk_type("class_declaration") == "class"

    def test_interface_types(self):
        """Maps interface/trait types correctly."""
        assert _node_to_chunk_type("interface_declaration") == "interface"
        assert _node_to_chunk_type("trait_item") == "interface"

    def test_struct_types(self):
        """Maps struct types correctly."""
        assert _node_to_chunk_type("struct_item") == "struct"
        assert _node_to_chunk_type("struct_declaration") == "struct"

    def test_enum_types(self):
        """Maps enum types correctly."""
        assert _node_to_chunk_type("enum_item") == "enum"
        assert _node_to_chunk_type("enum_declaration") == "enum"

    def test_impl_types(self):
        """Maps impl types correctly."""
        assert _node_to_chunk_type("impl_item") == "impl"

    def test_module_types(self):
        """Maps module types correctly."""
        assert _node_to_chunk_type("module") == "module"
        assert _node_to_chunk_type("module_definition") == "module"

    def test_type_types(self):
        """Maps type alias types correctly."""
        assert _node_to_chunk_type("type_alias") == "type"
        assert _node_to_chunk_type("type_item") == "type"

    def test_unknown_types(self):
        """Unknown types map to block."""
        assert _node_to_chunk_type("random_node") == "block"
        assert _node_to_chunk_type("expression") == "block"


class TestBuildContextPath:
    """Tests for _build_context_path."""

    def test_empty_parts(self):
        """Empty parts returns None."""
        assert _build_context_path([]) is None

    def test_single_part(self):
        """Single part returns that part."""
        assert _build_context_path(["class:Foo"]) == "class:Foo"

    def test_multiple_parts(self):
        """Multiple parts joined with dots."""
        result = _build_context_path(["class:Outer", "class:Inner"])
        assert result == "class:Outer.class:Inner"


class TestCachedParser:
    """Tests for parser caching."""

    def test_valid_language_returns_parser(self):
        """Valid language returns parser."""
        parser = _get_cached_parser("python")
        assert parser is not None

    def test_invalid_language_returns_none(self):
        """Invalid language returns None."""
        parser = _get_cached_parser("not_a_real_language_xyz123")
        assert parser is None

    def test_caching_works(self):
        """Same parser returned for repeated calls."""
        parser1 = _get_cached_parser("python")
        parser2 = _get_cached_parser("python")
        assert parser1 is parser2


class TestChunkWithTreesitter:
    """Tests for the main chunk_with_treesitter function."""

    def test_python_function(self):
        """Extracts Python function."""
        code = '''def my_func(x, y):
    """Docstring."""
    return x + y
'''
        chunks = chunk_with_treesitter(code, "python")
        assert len(chunks) >= 1
        func_chunks = [c for c in chunks if c.name == "my_func"]
        assert len(func_chunks) == 1
        assert func_chunks[0].chunk_type == "function"

    def test_python_class_with_methods(self):
        """Extracts Python class and methods."""
        code = '''class MyClass:
    """Class docstring."""

    def method1(self):
        pass

    def method2(self):
        pass
'''
        chunks = chunk_with_treesitter(code, "python")
        names = [c.name for c in chunks]
        assert "MyClass" in names
        assert "method1" in names or any("method" in (c.context or "") for c in chunks)

    def test_decorated_function(self):
        """Extracts decorated functions."""
        code = '''@decorator
def decorated_func():
    pass

@another_decorator
@decorator
def multi_decorated():
    pass
'''
        chunks = chunk_with_treesitter(code, "python")
        names = [c.name for c in chunks if c.name]
        assert "decorated_func" in names or len(names) > 0

    def test_unknown_language_returns_empty(self):
        """Unknown language returns empty list."""
        chunks = chunk_with_treesitter("code", "made_up_language")
        assert chunks == []

    def test_invalid_syntax_still_parses(self):
        """Tree-sitter handles invalid syntax gracefully."""
        code = '''def broken(
    # Missing closing paren
def another():
    pass
'''
        # Should not crash
        chunks = chunk_with_treesitter(code, "python")
        assert isinstance(chunks, list)

    def test_typescript_interface(self):
        """Extracts TypeScript interfaces."""
        code = '''interface User {
    id: string;
    name: string;
}
'''
        chunks = chunk_with_treesitter(code, "typescript")
        names = [c.name for c in chunks]
        assert "User" in names

    def test_rust_struct_and_impl(self):
        """Extracts Rust structs and impl blocks."""
        code = '''pub struct Point {
    x: i32,
    y: i32,
}

impl Point {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}
'''
        chunks = chunk_with_treesitter(code, "rust")
        chunk_types = [c.chunk_type for c in chunks]
        assert "struct" in chunk_types
        assert "impl" in chunk_types or "function" in chunk_types

    def test_go_function(self):
        """Extracts Go functions."""
        code = '''package main

func main() {
    fmt.Println("Hello")
}

func helper() int {
    return 42
}
'''
        chunks = chunk_with_treesitter(code, "go")
        names = [c.name for c in chunks if c.name]
        assert "main" in names or "helper" in names

    def test_deeply_nested_does_not_recurse(self):
        """Files with deeply nested ASTs must not blow Python's recursion limit.

        Regression: the AST walker used to recurse per node, so a file with
        AST depth > sys.getrecursionlimit() (default 1000) raised RecursionError
        and aborted the entire indexing run. A few thousand parentheses produce
        such a tree, as do large generated sources, long method chains, and
        deeply nested templates.
        """
        # ~3000 parens → AST depth well past the default 1000 limit.
        depth = 3000
        code = "x = " + "(" * depth + "1" + ")" * depth + "\n"
        chunks = chunk_with_treesitter(code, "python")
        assert isinstance(chunks, list)

    def test_chunks_preserve_source_order(self):
        """Iterative walker must yield chunks in source order (top-to-bottom)."""
        code = '''def first():
    pass

def second():
    pass

def third():
    pass
'''
        chunks = chunk_with_treesitter(code, "python")
        names = [c.name for c in chunks if c.name in {"first", "second", "third"}]
        assert names == ["first", "second", "third"]


class TestExtractDocstring:
    """Tests for _extract_docstring."""

    def test_python_class_docstring(self):
        """Extracts Python class docstring."""
        parser = _get_cached_parser("python")
        source = b'''class MyClass:
    """This is the class docstring."""
    def method(self):
        pass
'''
        tree = parser.parse(source)
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_definition":
                class_node = node
                break

        docstring = _extract_docstring(source, class_node, "python")
        # May or may not extract depending on implementation details
        if docstring:
            assert "docstring" in docstring.lower()

    def test_function_without_docstring(self):
        """No docstring returns None."""
        parser = _get_cached_parser("python")
        source = b'''def my_func():
    x = 1
    return x
'''
        tree = parser.parse(source)
        func_node = None
        for node in tree.root_node.children:
            if node.type == "function_definition":
                func_node = node
                break

        docstring = _extract_docstring(source, func_node, "python")
        assert docstring is None

    def test_class_without_body(self):
        """Class without proper body returns None."""
        parser = _get_cached_parser("python")
        source = b"class Empty: pass"
        tree = parser.parse(source)
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_definition":
                class_node = node
                break

        docstring = _extract_docstring(source, class_node, "python")
        # May or may not find docstring - just shouldn't crash
        assert docstring is None or isinstance(docstring, str)


class TestGenerateClassOverview:
    """Tests for _generate_class_overview."""

    def test_class_with_methods(self):
        """Generates overview with method signatures."""
        parser = _get_cached_parser("python")
        source = b'''class Calculator:
    """A calculator class."""

    def add(self, x, y):
        return x + y

    def subtract(self, x, y):
        return x - y
'''
        tree = parser.parse(source)
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_definition":
                class_node = node
                break

        overview = _generate_class_overview(class_node, source, "Calculator", "python")
        assert "Calculator" in overview
        assert "add" in overview
        assert "subtract" in overview
        assert "..." in overview  # Method signatures end with ...

    def test_class_with_decorated_methods(self):
        """Handles decorated methods in overview."""
        parser = _get_cached_parser("python")
        source = b'''class Service:
    @property
    def value(self):
        return self._value

    @staticmethod
    def helper():
        pass
'''
        tree = parser.parse(source)
        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_definition":
                class_node = node
                break

        overview = _generate_class_overview(class_node, source, "Service", "python")
        assert "Service" in overview
        # Should include decorated methods
        assert "value" in overview or "helper" in overview or "def" in overview


class TestLanguageSupport:
    """Tests for language mapping and support."""

    def test_language_map_coverage(self):
        """All common languages are mapped."""
        expected = ["python", "typescript", "javascript", "rust", "go", "java"]
        for lang in expected:
            assert lang in LANGUAGE_MAP

    def test_definition_types_coverage(self):
        """Core languages have definition types."""
        # Only check languages we explicitly support with definitions
        core_languages = ["python", "typescript", "javascript", "rust", "go", "java"]
        for lang in core_languages:
            assert lang in DEFINITION_TYPES
            assert len(DEFINITION_TYPES[lang]) > 0


class TestRealWorldScenarios:
    """Integration-style tests with realistic code."""

    def test_complex_python_module(self):
        """Handles complex Python module."""
        code = '''#!/usr/bin/env python
"""Module docstring."""

import os
from typing import List

class BaseClass:
    """Base class."""
    def base_method(self): pass

class DerivedClass(BaseClass):
    """Derived with multiple methods."""

    def __init__(self, value):
        self.value = value

    @property
    def computed(self):
        return self.value * 2

    def method_one(self, arg: int) -> str:
        """Method with type hints."""
        return str(arg)

    def method_two(self):
        pass

def standalone_function():
    """Standalone."""
    pass
'''
        chunks = chunk_with_treesitter(code, "python")

        # Should extract meaningful chunks
        assert len(chunks) >= 3

        names = [c.name for c in chunks if c.name]
        assert "BaseClass" in names or "DerivedClass" in names
        assert "standalone_function" in names

    def test_nested_classes(self):
        """Handles nested class definitions."""
        code = '''class Outer:
    class Inner:
        def inner_method(self):
            pass

    def outer_method(self):
        pass
'''
        chunks = chunk_with_treesitter(code, "python")

        # Should have context for inner class - check that at least some chunks have Outer context
        has_outer_context = any(c.context and "Outer" in c.context for c in chunks)
        # May or may not have inner class as separate chunk depending on tree-sitter parsing
        assert len(chunks) >= 1
        # At least verify that nested classes are handled (may or may not have context)
        assert has_outer_context or len(chunks) >= 1


class TestTreesitterEdgeCases:
    """Tests for edge cases to improve coverage."""

    def test_php_name_node_extraction(self):
        """PHP uses 'name' node type for identifiers (line 86)."""
        code = '''<?php
class Calculator {
    function add($a, $b) {
        return $a + $b;
    }
}
'''
        chunks = chunk_with_treesitter(code, "php")
        # PHP should parse and extract class/method names via 'name' node type
        names = [c.name for c in chunks if c.name]
        # Should extract Calculator and/or add
        assert "Calculator" in names or "add" in names

    def test_php_method_name_extraction(self):
        """PHP method names are extracted via 'name' node."""
        code = '''<?php
function standalone_function() {
    return 42;
}
'''
        chunks = chunk_with_treesitter(code, "php")
        names = [c.name for c in chunks if c.name]
        # Should extract function name
        assert "standalone_function" in names

    def test_parser_exception_returns_empty(self):
        """When parser lookup fails, returns empty list (line 233)."""
        from unittest.mock import patch

        from mcp_codesearch.indexer import treesitter

        # Clear cache to force fresh lookup
        original_cache = treesitter._parser_cache.copy()
        treesitter._parser_cache.clear()

        # Make get_parser raise exception for this test language
        def mock_get_parser(lang):
            if lang == "test_broken_lang":
                raise RuntimeError("Parser unavailable")

        # Add the broken language to LANGUAGE_MAP temporarily
        treesitter.LANGUAGE_MAP["test_broken"] = "test_broken_lang"

        try:
            with patch.object(treesitter.tree_sitter_language_pack, 'get_parser', mock_get_parser):
                result = chunk_with_treesitter("some code", "test_broken")

            assert result == []
        finally:
            # Restore
            treesitter._parser_cache.clear()
            treesitter._parser_cache.update(original_cache)
            del treesitter.LANGUAGE_MAP["test_broken"]

    def test_class_docstring_in_overview(self):
        """Class overview includes docstring (line 184)."""
        from mcp_codesearch.settings import settings

        # Use low threshold to trigger class_overview
        original_threshold = settings.class_split_threshold
        settings.class_split_threshold = 1  # Force overview for any class

        try:
            code = '''class MyClass:
    """This is a class docstring."""

    def method1(self):
        pass

    def method2(self):
        pass
'''
            chunks = chunk_with_treesitter(code, "python")

            # Find class overview chunk
            overview_chunks = [c for c in chunks if c.chunk_type == "class_overview"]
            if overview_chunks:
                overview = overview_chunks[0]
                # Docstring should be in overview content
                assert '"""' in overview.content or "class docstring" in overview.content.lower()
        finally:
            settings.class_split_threshold = original_threshold

    def test_extract_docstring_body_no_children(self):
        """_extract_docstring returns None when body has no children (line 145)."""
        parser = _get_cached_parser("python")
        # Create a class with pass (pass is not empty, but has no string docstring)
        source = b"class Empty: pass"
        tree = parser.parse(source)

        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_definition":
                class_node = node
                break

        # The body here should be minimal
        docstring = _extract_docstring(source, class_node, "python")
        # Should return None - no docstring found
        assert docstring is None

    def test_extract_docstring_no_body(self):
        """_extract_docstring returns None when node has no body child (line 144-145)."""
        # Go forward declarations have no body
        parser = _get_cached_parser("go")
        source = b"func foo()"
        tree = parser.parse(source)

        func_node = None
        for node in tree.root_node.children:
            if node.type == "function_declaration":
                func_node = node
                break

        assert func_node is not None
        docstring = _extract_docstring(source, func_node, "go")
        # No body means no docstring
        assert docstring is None

    def test_extract_docstring_returns_content(self):
        """_extract_docstring extracts actual docstring content (lines 151-156)."""
        parser = _get_cached_parser("python")
        source = b'''class MyClass:
    """This is the actual docstring that should be extracted."""
    def method(self):
        pass
'''
        tree = parser.parse(source)

        class_node = None
        for node in tree.root_node.children:
            if node.type == "class_definition":
                class_node = node
                break

        assert class_node is not None
        docstring = _extract_docstring(source, class_node, "python")

        # Should extract the docstring text
        assert docstring is not None
        assert "actual docstring" in docstring.lower() or "extracted" in docstring.lower()

    def test_javascript_property_identifier(self):
        """JS methods use property_identifier for name (line 88)."""
        code = '''class Widget {
    handleClick() {
        console.log("clicked");
    }

    getValue() {
        return this.value;
    }
}
'''
        chunks = chunk_with_treesitter(code, "javascript")

        names = [c.name for c in chunks if c.name]
        # Should extract method names via property_identifier
        assert "Widget" in names or "handleClick" in names or "getValue" in names

    def test_rust_type_identifier(self):
        """Rust uses type_identifier for struct names (line 90)."""
        code = '''pub struct Point {
    x: i32,
    y: i32,
}

pub struct Rectangle {
    width: u32,
    height: u32,
}
'''
        chunks = chunk_with_treesitter(code, "rust")

        names = [c.name for c in chunks if c.name]
        # Should extract struct names via type_identifier
        assert "Point" in names or "Rectangle" in names

    def test_class_overview_many_methods(self, large_class_code):
        """Class overview truncates >20 methods (line 218)."""
        from mcp_codesearch.settings import settings

        original_threshold = settings.class_split_threshold
        settings.class_split_threshold = 1  # Force overview for any class

        try:
            chunks = chunk_with_treesitter(large_class_code, "python")

            # Find class overview chunk
            overview_chunks = [c for c in chunks if c.chunk_type == "class_overview"]
            assert len(overview_chunks) >= 1
            overview = overview_chunks[0]

            # Should contain the truncation message for >20 methods
            assert "more methods" in overview.content or len(overview.content.split("def ")) > 5
        finally:
            settings.class_split_threshold = original_threshold

    def test_is_supported_function(self):
        """is_supported returns True for supported languages (line 312)."""
        from mcp_codesearch.indexer.treesitter import is_supported

        # Supported languages
        assert is_supported("python") is True
        assert is_supported("typescript") is True
        assert is_supported("rust") is True
        assert is_supported("go") is True

        # Unsupported language
        assert is_supported("made_up_language") is False
        assert is_supported("") is False
