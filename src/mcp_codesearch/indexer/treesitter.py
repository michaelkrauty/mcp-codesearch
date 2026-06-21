"""Tree-sitter based code chunking."""

from __future__ import annotations

import logging
import threading
from typing import Any

import tree_sitter_language_pack
from pydantic import BaseModel

from mcp_codesearch.settings import settings

logger = logging.getLogger(__name__)

# Parser cache to avoid repeated get_parser() calls
# Protected by lock for thread safety during concurrent indexing
_parser_cache: dict[str, Any] = {}
_parser_cache_lock = threading.Lock()


class Chunk(BaseModel):
    """A code chunk extracted from a file."""

    content: str
    chunk_type: str  # function, class, method, block
    name: str | None
    start_line: int
    end_line: int
    context: str | None  # Parent class/module name
    imports: list[str] | None = None  # File-level imports (attached to all chunks)


# Map our language names to tree-sitter language names
LANGUAGE_MAP = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
    "scala": "scala",
    "csharp": "c_sharp",
    "bash": "bash",
    "html": "html",
    "css": "css",
    # Config/data languages
    "sql": "sql",
    "json": "json",
    "yaml": "yaml",
    "toml": "toml",
}

# Node types that represent top-level definitions
DEFINITION_TYPES = {
    "python": [
        "function_definition", "async_function_definition",
        "class_definition", "decorated_definition",
    ],
    "javascript": [
        "function_declaration", "class_declaration", "arrow_function", "method_definition"
    ],
    "typescript": [
        "function_declaration", "class_declaration", "arrow_function", "method_definition",
        "interface_declaration", "type_alias_declaration"
    ],
    "go": ["function_declaration", "method_declaration", "type_declaration"],
    "rust": ["function_item", "impl_item", "struct_item", "enum_item", "trait_item"],
    "java": ["method_declaration", "class_declaration", "interface_declaration"],
    "c": ["function_definition", "struct_specifier"],
    "cpp": ["function_definition", "class_specifier", "struct_specifier"],
    "ruby": ["method", "class", "module"],
    "php": ["function_definition", "class_declaration", "method_declaration"],
    "swift": ["function_declaration", "class_declaration", "struct_declaration"],
    "kotlin": ["function_declaration", "class_declaration"],
    "scala": ["function_definition", "class_definition", "object_definition"],
    "csharp": ["method_declaration", "class_declaration", "interface_declaration"],
    # SQL: CREATE statements for schemas
    "sql": ["create_table_statement", "create_view_statement", "create_function_statement",
            "create_procedure_statement", "create_index_statement"],
    # JSON/YAML/TOML: data structures (will fall back to line chunking if no definitions found)
    "json": ["object", "array"],  # Top-level structures
    "yaml": ["block_mapping", "block_sequence"],  # Top-level YAML structures
    "toml": ["table", "array"],  # Top-level TOML structures
}


# Direct-child node types that carry a definition's name, across languages.
_NAME_NODE_TYPES = frozenset(
    {
        "identifier",  # most languages
        "name",  # Ruby methods
        "property_identifier",  # JS/TS methods
        "type_identifier",  # Rust
        "field_identifier",  # Go methods
        "simple_identifier",  # Swift / Kotlin
        "constant",  # Ruby class / module
    }
)


def _name_from_declarator(node: Any, source: bytes) -> str | None:
    """Resolve the declared name from a C/C++ declarator subtree.

    A function_definition's name lives inside a function_declarator, which may
    be wrapped in pointer/reference/array declarators (e.g. ``int *foo()``). The
    name itself can be an identifier, a field_identifier (methods), a
    qualified_identifier (``ns::C::f``), a destructor_name, or an operator_name.
    """
    for child in node.children:
        if child.type == "function_declarator":
            for g in child.children:
                if g.type in (
                    "identifier",
                    "field_identifier",
                    "qualified_identifier",
                    "destructor_name",
                    "operator_name",
                ):
                    text = source[g.start_byte:g.end_byte].decode("utf-8", errors="ignore")
                    # Index the bare name for qualified C++ names (ns::C::f -> f),
                    # matching how methods are named in the other languages.
                    return text.rsplit("::", 1)[-1]
            return None
        if child.type in (
            "pointer_declarator",
            "reference_declarator",
            "array_declarator",
        ):
            inner = _name_from_declarator(child, source)
            if inner is not None:
                return inner
    return None


def _get_node_name(node: Any, source: bytes) -> str | None:
    """Extract name from a definition node."""
    # C/C++ carry the name inside a (possibly pointer/reference-wrapped)
    # function_declarator, AFTER the return type. Resolve that first so the
    # return type is not mistaken for the name (e.g. `Foo bar()` -> "bar").
    declarator_name = _name_from_declarator(node, source)
    if declarator_name is not None:
        return declarator_name
    for child in node.children:
        if child.type in _NAME_NODE_TYPES:
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
    return None


def _node_to_chunk_type(node_type: str) -> str:  # noqa: PLR0911
    """Map tree-sitter node type to our chunk type."""
    if "function" in node_type or "method" in node_type:
        return "function"
    if "class" in node_type:
        return "class"
    if "interface" in node_type or "trait" in node_type:
        return "interface"
    if "struct" in node_type:
        return "struct"
    if "enum" in node_type:
        return "enum"
    if "impl" in node_type:
        return "impl"
    if "module" in node_type:
        return "module"
    if "type" in node_type:
        return "type"
    return "block"


def _get_cached_parser(ts_language: str) -> Any | None:
    """Get parser from cache, creating if needed (thread-safe).

    Always uses the lock for portability - the performance overhead is minimal
    since parser creation is the slow path, and dict reads being "atomic" in
    CPython is an implementation detail not guaranteed across Python versions.
    """
    with _parser_cache_lock:
        if ts_language in _parser_cache:
            return _parser_cache[ts_language]

        try:
            parser = tree_sitter_language_pack.get_parser(ts_language)  # type: ignore[arg-type]
            _parser_cache[ts_language] = parser
        except (LookupError, ValueError, OSError, RuntimeError) as e:
            # LookupError: unsupported language
            # ValueError: invalid language specification
            # OSError: library loading failed
            # RuntimeError: parser initialization failed
            logger.debug(f"Cannot create tree-sitter parser for {ts_language}: {e}")
            _parser_cache[ts_language] = None

        return _parser_cache[ts_language]


def _build_context_path(context_parts: list[str]) -> str | None:
    """Build hierarchical context path from parts.

    Example: ['class:Outer', 'class:Inner'] -> 'class:Outer.class:Inner'
    """
    if not context_parts:
        return None
    return ".".join(context_parts)


def _extract_docstring(source: bytes, node: Any, language: str) -> str | None:
    """Extract docstring from class/function node if present."""
    # Find body/block child
    body = None
    for child in node.children:
        if child.type in ("block", "class_body", "body"):
            body = child
            break

    if not body or not body.children:
        return None

    first_stmt = body.children[0]

    # Python: docstring as first statement
    if language == "python":
        # Current grammar: string node directly in block
        if first_stmt.type == "string":
            raw = source[first_stmt.start_byte:first_stmt.end_byte]
            text = raw.decode("utf-8", errors="ignore")
            # Clean up triple quotes
            text = text.strip("'\"")
            return text[:300]  # Limit docstring length
        # Older grammar: expression_statement containing a string
        elif first_stmt.type == "expression_statement":
            for child in first_stmt.children:
                if child.type == "string":
                    raw = source[child.start_byte:child.end_byte]
                    text = raw.decode("utf-8", errors="ignore")
                    text = text.strip("'\"")
                    return text[:300]

    return None


def _generate_class_overview(node: Any, source: bytes, name: str | None, language: str) -> str:
    """Generate class overview with docstring and method signatures.

    For large classes, creates a summary instead of storing full content.
    Format:
        class ClassName:
            '''Docstring'''

            def method1(self, args): ...
            def method2(self, args): ...
    """
    parts = []

    # Get class declaration line
    class_line = source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
    # Extract just the first line (class definition)
    lines = class_line.split("\n")
    first_line = lines[0].rstrip() if lines else ""
    parts.append(first_line)

    # Try to extract docstring
    docstring = _extract_docstring(source, node, language)
    if docstring:
        # Format docstring nicely
        parts.append(f'    """{docstring}"""')

    # Extract method signatures
    method_sigs = []
    for child in node.children:
        if child.type in ("block", "class_body", "body"):
            for stmt in child.children:
                # Look for function definitions
                if "function" in stmt.type or "method" in stmt.type:
                    raw = source[stmt.start_byte:stmt.end_byte]
                    method_text = raw.decode("utf-8", errors="ignore")
                    # Extract first line (signature)
                    method_lines = method_text.split("\n")
                    sig_line = method_lines[0].rstrip() if method_lines else ""
                    method_sigs.append(f"    {sig_line.strip()} ...")
                # Handle decorated methods
                elif stmt.type == "decorated_definition":
                    for dec_child in stmt.children:
                        if "function" in dec_child.type or "method" in dec_child.type:
                            raw = source[dec_child.start_byte:dec_child.end_byte]
                            method_text = raw.decode("utf-8", errors="ignore")
                            dec_lines = method_text.split("\n")
                            sig_line = dec_lines[0].rstrip() if dec_lines else ""
                            method_sigs.append(f"    {sig_line.strip()} ...")
                            break

    if method_sigs:
        parts.append("")
        parts.extend(method_sigs[:20])  # Limit to 20 methods in overview
        if len(method_sigs) > 20:
            # Note: Individual methods are still extracted as separate searchable chunks
            remaining = len(method_sigs) - 20
            parts.append(f"    # ... and {remaining} more methods (each indexed separately)")

    return "\n".join(parts)


def chunk_with_treesitter(content: str, language: str) -> list[Chunk]:
    """
    Chunk code using tree-sitter AST parsing.

    Args:
        content: Source code content
        language: Language identifier (our naming)

    Returns:
        List of Chunk objects
    """
    ts_language = LANGUAGE_MAP.get(language)
    if not ts_language:
        return []

    parser = _get_cached_parser(ts_language)
    if parser is None:
        return []

    source = content.encode("utf-8")
    tree = parser.parse(source)

    definition_types = set(DEFINITION_TYPES.get(language, []))
    chunks: list[Chunk] = []

    # Iterative DFS over the AST. A recursive walker blows Python's default
    # recursion limit (1000) on files with deeply nested expressions —
    # heavily-parenthesized math, long method chains, large template trees, etc.
    # Children are pushed in reverse so they pop in source order.
    stack: list[tuple[Any, list[str]]] = [(tree.root_node, [])]
    while stack:
        node, context_parts = stack.pop()

        # Only NAMED definition nodes are real definitions. Some grammars name a
        # definition's node type identically to its keyword token (e.g. Ruby's
        # anonymous `class`/`module` tokens have node.type == "class"/"module"),
        # and emitting a chunk for that token would collide with the real
        # definition's point id (same start line) and overwrite its content.
        if not (node.is_named and node.type in definition_types):
            for child in reversed(node.children):
                stack.append((child, context_parts))
            continue

        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        name = _get_node_name(node, source)

        # Handle decorated definitions (Python). The wrapper node carries the
        # decorators, but the type and container decisions, the overview
        # generation, and the child traversal must run on the inner definition
        # (function_definition / class_definition). Driving them off the wrapper
        # types every decorated def as a generic "block" and a decorated class
        # never yields its overview or per-method chunks. The chunk content and
        # line range stay on the outer node so the decorators remain part of it.
        type_node = node
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in definition_types:
                    name = _get_node_name(child, source)
                    type_node = child
                    break

        chunk_type = _node_to_chunk_type(type_node.type)
        full_context = _build_context_path(context_parts)

        is_container = (
            "class" in type_node.type
            or "struct" in type_node.type
            or "impl" in type_node.type
        )
        lines = node.end_point[0] - node.start_point[0] + 1

        if is_container and lines > settings.class_split_threshold:
            overview_content = _generate_class_overview(type_node, source, name, language)
            # The overview is built from the inner class node and would drop the
            # @decorator lines that precede it; prepend them so the overview
            # chunk still contains the class decorators. The slice is empty for
            # an undecorated class (where type_node is node), making this a no-op.
            decorators = source[node.start_byte:type_node.start_byte].decode(
                "utf-8", errors="ignore"
            )
            overview_content = decorators + overview_content
            chunks.append(Chunk(
                content=overview_content,
                chunk_type="class_overview",
                name=name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                context=full_context,
            ))
        else:
            chunks.append(Chunk(
                content=text,
                chunk_type=chunk_type,
                name=name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                context=full_context,
            ))

        if is_container:
            type_prefix = "class" if "class" in type_node.type else chunk_type
            new_context = (
                context_parts + [f"{type_prefix}:{name}"] if name else context_parts
            )
            for child in reversed(type_node.children):
                stack.append((child, new_context))

    return chunks


def is_supported(language: str) -> bool:
    """Check if language is supported by tree-sitter chunking."""
    return language in LANGUAGE_MAP
