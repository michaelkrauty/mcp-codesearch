"""Code chunking router with fallback."""

from __future__ import annotations

import re

from mcp_codesearch.settings import settings

from .treesitter import Chunk, chunk_with_treesitter
from .treesitter import is_supported as treesitter_supported


def _chunk_fixed_size(
    content: str,
    chunk_size: int = 500,
    overlap: int = 25,
) -> list[Chunk]:
    """
    Fallback: chunk by lines with overlap.

    Used when tree-sitter can't extract meaningful definitions (e.g., config files).

    Args:
        content: File content to chunk
        chunk_size: Maximum lines per chunk
        overlap: Lines of overlap between adjacent chunks

    Raises:
        ValueError: If overlap >= chunk_size (would cause infinite loop)
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    lines = content.split("\n")
    chunks = []

    if len(lines) <= chunk_size:
        # Single chunk for small files
        return [
            Chunk(
                content=content,
                chunk_type="block",
                name=None,
                start_line=1,
                end_line=len(lines),
                context=None,
            )
        ]

    start = 0
    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        chunk_lines = lines[start:end]
        chunk_content = "\n".join(chunk_lines)

        chunks.append(
            Chunk(
                content=chunk_content,
                chunk_type="block",
                name=None,
                start_line=start + 1,  # 1-indexed
                end_line=end,
                context=None,
            )
        )

        # Break when we've processed all content
        if end >= len(lines):
            break

        # Ensure we always make forward progress (at least 1 line)
        # This prevents infinite loops when overlap >= chunk_size
        step = max(1, chunk_size - overlap)
        start += step

    return chunks


def _merge_small_chunks(chunks: list[Chunk], min_lines: int = 10) -> list[Chunk]:
    """Merge adjacent small FALLBACK chunks only.

    Only merges "block" type chunks (from line-based fallback chunking).
    AST-extracted chunks (functions, classes, etc.) are never merged,
    as each represents a semantically meaningful code unit.
    """
    if len(chunks) <= 1:
        return chunks

    merged = []
    current = None

    for chunk in chunks:
        chunk_lines = chunk.end_line - chunk.start_line + 1

        if current is None:
            current = chunk
            continue

        current_lines = current.end_line - current.start_line + 1

        # Only merge fallback "block" chunks, never AST-extracted chunks
        # This preserves individual functions, methods, classes as separate searchable units
        is_both_blocks = current.chunk_type == "block" and chunk.chunk_type == "block"

        # Merge if both are small fallback blocks and adjacent
        if (
            is_both_blocks
            and current_lines < min_lines
            and chunk_lines < min_lines
            and chunk.start_line <= current.end_line + 2  # Adjacent or 1 line gap
        ):
            # Merge into current (preserve original chunk type)
            current = Chunk(
                content=current.content + "\n" + chunk.content,
                chunk_type=current.chunk_type,
                name=current.name,
                start_line=current.start_line,
                end_line=chunk.end_line,
                context=current.context,
            )
        else:
            merged.append(current)
            current = chunk

    if current:
        merged.append(current)

    return merged


def chunk_file(content: str, language: str) -> list[Chunk]:
    """
    Chunk a file using appropriate strategy.

    1. Try tree-sitter AST-based chunking
    2. Fall back to fixed-size line-based chunking
    3. Attach file-level imports to all chunks for better semantic matching

    Args:
        content: File content
        language: Language identifier

    Returns:
        List of Chunk objects
    """
    chunks = []

    # Try tree-sitter first
    if treesitter_supported(language):
        chunks = chunk_with_treesitter(content, language)

    # If tree-sitter found nothing useful, use fallback
    if not chunks:
        chunks = _chunk_fixed_size(
            content,
            chunk_size=settings.chunk_max_lines,
            overlap=settings.chunk_overlap_lines,
        )

    # Merge small chunks
    chunks = _merge_small_chunks(chunks, min_lines=settings.chunk_min_lines)

    # Extract file-level imports and attach to all chunks
    imports = _extract_imports(content, language)
    if imports:
        for chunk in chunks:
            chunk.imports = imports

    return chunks


def _extract_module_docstring(  # noqa: PLR0912, PLR0915
    content: str, language: str
) -> str | None:
    """
    Extract module-level docstring from file content.

    Supports:
    - Python: '''docstring''' or \"\"\"docstring\"\"\"
    - JavaScript/TypeScript: /** JSDoc comment */
    - Rust: //! module doc or /// crate doc
    - Go: // Package comment before package declaration
    - Java: /** Javadoc */
    """
    # Skip shebang and encoding lines for all languages
    lines = content.split("\n")
    start_idx = 0
    for i, line in enumerate(lines[:5]):  # Check first 5 lines
        stripped = line.strip()
        if stripped.startswith("#!") or (stripped.startswith("#") and "coding" in stripped):
            start_idx = i + 1
            continue
        if not stripped:
            start_idx = i + 1
            continue
        break

    remaining = "\n".join(lines[start_idx:])

    if language == "python":
        # Python: triple-quoted string at start
        patterns = [
            r'^"""(.*?)"""',  # Double-quoted
            r"^'''(.*?)'''",  # Single-quoted
        ]
        for pattern in patterns:
            match = re.search(pattern, remaining, re.DOTALL)
            if match:
                docstring = match.group(1).strip()
                return docstring[:300] if docstring else None

    elif language in ("javascript", "typescript"):
        # JSDoc: /** ... */ at file start
        match = re.search(r'^\s*/\*\*(.*?)\*/', remaining, re.DOTALL)
        if match:
            # Clean up JSDoc formatting: remove leading * from each line
            docstring = match.group(1)
            docstring = re.sub(r'^\s*\*\s?', '', docstring, flags=re.MULTILINE)
            docstring = docstring.strip()
            return docstring[:300] if docstring else None

    elif language == "rust":
        # Rust: //! inner doc comments or /// outer doc at start
        doc_lines = []
        for line in remaining.split("\n")[:30]:  # Check first 30 lines
            stripped = line.strip()
            if stripped.startswith("//!") or stripped.startswith("///"):
                # Remove doc comment prefix and leading space
                doc_content = re.sub(r'^//[!/]\s?', '', stripped)
                doc_lines.append(doc_content)
            elif stripped.startswith("//"):
                continue  # Regular comment, skip
            elif stripped and not stripped.startswith("#"):  # Non-comment code reached
                break
        if doc_lines:
            return "\n".join(doc_lines)[:300]

    elif language == "go":
        # Go: // comments before package declaration
        doc_lines = []
        for line in remaining.split("\n")[:30]:
            stripped = line.strip()
            if stripped.startswith("//"):
                doc_content = stripped[2:].strip()
                doc_lines.append(doc_content)
            elif stripped.startswith("package "):
                break  # Reached package declaration
            elif stripped and not stripped.startswith("//"):
                break  # Non-comment code
        if doc_lines:
            return "\n".join(doc_lines)[:300]

    elif language == "java":
        # Javadoc: /** ... */ at file start (after package/import)
        # Look for first /** that comes before a class
        match = re.search(r'/\*\*(.*?)\*/', remaining, re.DOTALL)
        if match:
            docstring = match.group(1)
            docstring = re.sub(r'^\s*\*\s?', '', docstring, flags=re.MULTILINE)
            docstring = docstring.strip()
            return docstring[:300] if docstring else None

    return None


def _extract_imports(content: str, language: str) -> list[str]:
    """Extract import/require statements from file content."""
    imports = []

    if language == "python":
        # Python: import X, from X import Y
        for match in re.finditer(r'^(?:from\s+(\S+)|import\s+(\S+))', content, re.MULTILINE):
            module = match.group(1) or match.group(2)
            if module:
                # Get base module (before any dots or commas)
                parts = module.split(".")
                base = parts[0] if parts else module
                base_parts = base.split(",")
                base = base_parts[0] if base_parts else base
                if base not in imports:
                    imports.append(base)
    elif language in ("javascript", "typescript"):
        # JS/TS: import X from 'Y', require('Y')
        js_import_pattern = r"(?:from\s+['\"]([^'\"]+)['\"]|require\(['\"]([^'\"]+)['\"])"
        for match in re.finditer(js_import_pattern, content):
            module = match.group(1) or match.group(2)
            if module and module not in imports:
                imports.append(module)
    elif language == "go":
        # Go: import "X" or import ( "X" "Y" )
        for match in re.finditer(r'"([^"]+)"', content[:1000]):  # Check first 1000 chars
            full_module = match.group(1)
            parts = full_module.split("/")
            module = parts[-1] if parts else full_module  # Get package name
            if module and module not in imports:
                imports.append(module)

    return imports[:15]  # Limit to 15


def generate_file_summary(content: str, chunks: list[Chunk], language: str) -> str:
    """
    Generate intelligent file summary from content and AST chunks.

    Extracts:
    1. Module docstring (if present)
    2. Import statements (summarized)
    3. Top-level definitions (from chunks)

    Falls back to content[:500] if nothing useful extracted.
    """
    parts = []

    # 1. Try to extract module docstring
    docstring = _extract_module_docstring(content, language)
    if docstring:
        parts.append(docstring)

    # 2. Extract and summarize imports
    imports = _extract_imports(content, language)
    if imports:
        parts.append(f"Imports: {', '.join(imports)}")

    # 3. List top-level definitions from chunks
    TYPE_ABBREV = {
        "function": "fn",
        "class": "cls",
        "class_overview": "cls",
        "interface": "ifc",
        "struct": "struct",
        "enum": "enum",
        "impl": "impl",
        "module": "mod",
        "type": "type",
        "block": "blk",
    }
    definitions = []
    for chunk in chunks:
        if chunk.name and chunk.context is None:  # Top-level only
            type_abbrev = TYPE_ABBREV.get(chunk.chunk_type, chunk.chunk_type[:3])
            definitions.append(f"{type_abbrev}:{chunk.name}")

    if definitions:
        parts.append(f"Defines: {', '.join(definitions[:15])}")

    # 4. Build summary
    if parts:
        return "\n".join(parts)

    # Fallback: use first N chars, trying to find a clean break
    fallback = content[:500]
    # Try to end at a newline
    last_newline = fallback.rfind("\n")
    if last_newline > 200:
        fallback = fallback[:last_newline]

    return fallback
