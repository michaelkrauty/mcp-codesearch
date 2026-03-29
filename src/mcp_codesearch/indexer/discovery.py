"""File discovery with .gitignore support."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pathspec
from pydantic import BaseModel, ConfigDict
from vector_core.utils.hashing import hash_content

from mcp_codesearch.settings import settings

logger = logging.getLogger(__name__)


def _safe_read_file(file_path: Path) -> str | None:
    """Read file atomically without following symlinks (TOCTOU-safe).

    This prevents race conditions where a file could be replaced with a symlink
    between the is_symlink() check and the actual file read.

    Args:
        file_path: Path to the file to read

    Returns:
        File contents as string, or None if file is a symlink, doesn't exist,
        or cannot be read.
    """
    # On Windows, O_NOFOLLOW is not available, so fall back to is_symlink() check
    # with a comment noting the limitation
    if sys.platform == "win32":
        # Windows limitation: No O_NOFOLLOW support, small TOCTOU window exists
        if file_path.is_symlink():
            return None
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    # Unix: Use O_NOFOLLOW for atomic symlink rejection
    fd = -1
    try:
        fd = os.open(str(file_path), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as f:
            fd = -1  # fdopen took ownership
            return f.read()
    except OSError:
        # Symlink, doesn't exist, or permission denied
        return None
    except Exception as e:
        # Log for debugging, but continue (one bad file shouldn't stop indexing)
        logger.debug(f"Unexpected error reading {file_path}: {type(e).__name__}: {e}")
        return None
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

# Language detection by extension
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".cs": "csharp",
    ".vue": "vue",
    ".svelte": "svelte",
    ".md": "markdown",
    ".sql": "sql",
    ".sh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
}


class FileInfo(BaseModel):
    """Information about a discovered file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path  # Absolute path
    rel_path: str  # Relative to codebase root
    language: str
    size_bytes: int
    content: str
    content_hash: str
    line_count: int
    mtime: float  # Modification time (seconds since epoch)


class FileMetadata(BaseModel):
    """Lightweight file metadata for fast change detection."""

    rel_path: str
    size_bytes: int
    mtime: float
    content_hash: str


def _load_gitignore(codebase_path: Path) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from codebase root."""
    gitignore_path = codebase_path / ".gitignore"
    if not gitignore_path.exists():
        return None

    with open(gitignore_path, encoding="utf-8", errors="ignore") as f:
        patterns = f.read().splitlines()

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _detect_language(path: Path) -> str | None:
    """Detect language from file extension."""
    return EXTENSION_TO_LANGUAGE.get(path.suffix.lower())


def discover_files(  # noqa: PLR0912
    codebase_path: str | Path,
    include_extensions: set[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_file_size_kb: int | None = None,
) -> Iterator[FileInfo]:
    """
    Discover code files in a codebase.

    Args:
        codebase_path: Root path of codebase
        include_extensions: Extensions to include (default: from settings)
        exclude_patterns: Additional patterns to exclude
        max_file_size_kb: Max file size in KB (default: from settings)

    Yields:
        FileInfo for each discovered file
    """
    codebase_path = Path(codebase_path).resolve()
    extensions = include_extensions or settings.code_extensions
    max_size = (max_file_size_kb or settings.max_file_size_kb) * 1024

    # Load .gitignore
    gitignore = _load_gitignore(codebase_path)

    # Build additional exclude spec
    exclude_spec = None
    if exclude_patterns:
        exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", exclude_patterns)

    # Always exclude these directories
    always_exclude = {
        ".git", ".svn", ".hg", "node_modules", "__pycache__",
        ".venv", "venv", ".tox", ".pytest_cache", ".mypy_cache",
        "dist", "build", ".next", ".nuxt", "target", "coverage",
    }

    # followlinks=False prevents infinite loops from symlinks pointing to parent directories
    for root, dirs, files in os.walk(codebase_path, followlinks=False):
        root_path = Path(root)
        rel_root = root_path.relative_to(codebase_path)

        # Filter directories in-place (also skip symlinked directories)
        dirs[:] = [
            d for d in dirs
            if d not in always_exclude
            and not d.startswith(".")
            and not (root_path / d).is_symlink()
        ]

        for filename in files:
            file_path = root_path / filename
            rel_path = str(rel_root / filename)

            # Skip hidden files and symlinks
            if filename.startswith("."):
                continue
            if file_path.is_symlink():
                continue

            # Check extension
            if file_path.suffix.lower() not in extensions:
                continue

            # Check gitignore
            if gitignore and gitignore.match_file(rel_path):
                continue

            # Check exclude patterns
            if exclude_spec and exclude_spec.match_file(rel_path):
                continue

            # Check file size
            try:
                stat = file_path.stat()
                if stat.st_size > max_size:
                    continue
                if stat.st_size == 0:
                    continue
            except OSError as e:
                logger.warning(f"Failed to stat file {rel_path}: {e}")
                continue

            # Read content (TOCTOU-safe)
            content = _safe_read_file(file_path)
            if content is None:
                logger.debug(f"Skipping file {rel_path}: could not read (symlink or permission issue)")
                continue
            # Check if content had encoding issues (replacement chars present)
            if "\ufffd" in content:
                logger.debug(
                    f"File {rel_path} has encoding issues (non-UTF-8 characters replaced)"
                )

            # Detect language
            language = _detect_language(file_path)
            if not language:
                continue

            yield FileInfo(
                path=file_path,
                rel_path=rel_path,
                language=language,
                size_bytes=stat.st_size,
                content=content,
                content_hash=hash_content(content),
                line_count=content.count("\n") + 1,
                mtime=stat.st_mtime,
            )


def get_file_hash(file_path: Path) -> str | None:
    """Get hash of file content without loading full content."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        return hash_content(content)
    except OSError as e:
        logger.debug(f"Cannot hash file {file_path}: {e}")
        return None


def get_file_stat(file_path: Path) -> tuple[float, int] | None:
    """Get file mtime and size without reading content. Returns (mtime, size)."""
    try:
        stat = file_path.stat()
        return (stat.st_mtime, stat.st_size)
    except OSError as e:
        logger.debug(f"Cannot stat file {file_path}: {e}")
        return None


def read_specific_files(
    codebase_path: str | Path,
    relative_paths: set[str],
) -> Iterator[FileInfo]:
    """
    Read only specific files by their relative paths.

    This is more efficient than discover_files() when you know exactly which
    files need to be read (e.g., during incremental indexing where only
    new/modified files need content loaded).

    Args:
        codebase_path: Root path of codebase
        relative_paths: Set of relative paths to read

    Yields:
        FileInfo for each valid file
    """
    codebase_path = Path(codebase_path).resolve()

    for rel_path in relative_paths:
        file_path = codebase_path / rel_path

        # Validate path containment (prevent traversal attacks like "../../../etc/passwd")
        try:
            resolved = file_path.resolve()
            resolved.relative_to(codebase_path)
        except ValueError:
            logger.warning(f"Path traversal attempt blocked: {rel_path}")
            continue

        # Skip if doesn't exist or is symlink
        if not resolved.exists() or resolved.is_symlink():
            continue

        # Detect language
        language = _detect_language(resolved)
        if not language:
            continue

        # Get stats
        try:
            stat = resolved.stat()
            if stat.st_size == 0:
                continue
        except OSError as e:
            logger.warning(f"Failed to stat file {rel_path}: {e}")
            continue

        # Read content (TOCTOU-safe)
        content = _safe_read_file(resolved)
        if content is None:
            logger.debug(f"Skipping file {rel_path}: could not read (symlink or permission issue)")
            continue

        yield FileInfo(
            path=resolved,
            rel_path=rel_path,
            language=language,
            size_bytes=stat.st_size,
            content=content,
            content_hash=hash_content(content),
            line_count=content.count("\n") + 1,
            mtime=stat.st_mtime,
        )


def scan_file_metadata(
    codebase_path: str | Path,
    include_extensions: set[str] | None = None,
    max_file_size_kb: int | None = None,
) -> Iterator[tuple[str, float, int]]:
    """
    Fast scan returning only (rel_path, mtime, size) for each file.

    This is much faster than discover_files() as it doesn't read file content.
    Use for quick change detection before committing to full file reads.

    Yields:
        Tuple of (rel_path, mtime, size_bytes)
    """
    codebase_path = Path(codebase_path).resolve()
    extensions = include_extensions or settings.code_extensions
    max_size = (max_file_size_kb or settings.max_file_size_kb) * 1024

    # Load .gitignore
    gitignore = _load_gitignore(codebase_path)

    # Always exclude these directories
    always_exclude = {
        ".git", ".svn", ".hg", "node_modules", "__pycache__",
        ".venv", "venv", ".tox", ".pytest_cache", ".mypy_cache",
        "dist", "build", ".next", ".nuxt", "target", "coverage",
    }

    # followlinks=False prevents infinite loops from symlinks pointing to parent directories
    for root, dirs, files in os.walk(codebase_path, followlinks=False):
        root_path = Path(root)
        rel_root = root_path.relative_to(codebase_path)

        # Filter directories in-place (also skip symlinked directories)
        dirs[:] = [
            d for d in dirs
            if d not in always_exclude
            and not d.startswith(".")
            and not (root_path / d).is_symlink()
        ]

        for filename in files:
            file_path = root_path / filename
            rel_path = str(rel_root / filename)

            # Skip hidden files and symlinks
            if filename.startswith("."):
                continue
            if file_path.is_symlink():
                continue

            # Check extension
            if file_path.suffix.lower() not in extensions:
                continue

            # Check gitignore
            if gitignore and gitignore.match_file(rel_path):
                continue

            # Check file size and get stats
            try:
                stat = file_path.stat()
                if stat.st_size > max_size or stat.st_size == 0:
                    continue
            except OSError:
                continue

            yield (rel_path, stat.st_mtime, stat.st_size)
