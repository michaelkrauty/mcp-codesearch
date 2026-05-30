"""File discovery with nested .gitignore and .codesearchignore support."""

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


# Directories never traversed during discovery (VCS internals, dependencies,
# build artifacts, caches). These are pruned regardless of ignore files.
_ALWAYS_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git", ".svn", ".hg", "node_modules", "__pycache__",
        ".venv", "venv", ".tox", ".pytest_cache", ".mypy_cache",
        "dist", "build", ".next", ".nuxt", "target", "coverage",
    }
)

# Ignore files honored at every directory level, using gitignore syntax.
# ".codesearchignore" lets a project exclude paths from indexing without
# changing git's behavior (which editing ".gitignore" would).
_IGNORE_FILENAMES: tuple[str, ...] = (".gitignore", ".codesearchignore")


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


def _load_ignore_spec(path: Path) -> pathspec.GitIgnoreSpec | None:
    """Parse a gitignore-syntax file into a spec, or None if absent/unreadable."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return pathspec.GitIgnoreSpec.from_lines(f.read().splitlines())
    except OSError:
        return None


def _load_gitignore(codebase_path: Path) -> pathspec.GitIgnoreSpec | None:
    """Load root-level .gitignore patterns (kept for backwards compatibility)."""
    return _load_ignore_spec(codebase_path / ".gitignore")


def _dir_ignore_specs(directory: Path) -> list[tuple[Path, pathspec.GitIgnoreSpec]]:
    """Ignore specs defined directly in ``directory`` (e.g. its own .gitignore)."""
    specs: list[tuple[Path, pathspec.GitIgnoreSpec]] = []
    for name in _IGNORE_FILENAMES:
        spec = _load_ignore_spec(directory / name)
        if spec is not None:
            specs.append((directory, spec))
    return specs


def _root_ignore_specs(root: Path) -> list[tuple[Path, pathspec.GitIgnoreSpec]]:
    """Root-level specs: ``.git/info/exclude`` (lowest precedence), then ignore files.

    The user's global ``core.excludesFile`` is intentionally not consulted, so that
    indexing stays reproducible and independent of per-machine git configuration.
    """
    specs: list[tuple[Path, pathspec.GitIgnoreSpec]] = []
    info_exclude = _load_ignore_spec(root / ".git" / "info" / "exclude")
    if info_exclude is not None:
        specs.append((root, info_exclude))
    specs.extend(_dir_ignore_specs(root))
    return specs


def _detect_language(path: Path) -> str | None:
    """Detect language from file extension."""
    return EXTENSION_TO_LANGUAGE.get(path.suffix.lower())


def _is_ignored(
    abs_path: Path,
    rel_to_root: str,
    specs: list[tuple[Path, pathspec.GitIgnoreSpec]],
    exclude_spec: pathspec.GitIgnoreSpec | None,
    *,
    is_dir: bool,
) -> bool:
    """Decide whether a path is excluded by the accumulated ignore specs.

    Mirrors git precedence: the programmatic ``exclude_spec`` wins outright, then
    the deepest matching ignore file decides (``!`` negations re-include the path),
    falling back to shallower files and finally ``.git/info/exclude``.

    ``specs`` are ordered root -> deep; each is matched relative to its own base
    directory. A trailing slash is appended for directories so that directory-only
    patterns such as ``build/`` match.
    """
    suffix = "/" if is_dir else ""
    if exclude_spec is not None and exclude_spec.match_file(rel_to_root + suffix):
        return True
    # Deepest spec first: the most specific ignore file takes precedence.
    for base, spec in reversed(specs):
        try:
            rel = str(abs_path.relative_to(base))
        except ValueError:
            continue
        result = spec.check_file(rel + suffix)
        if result.include is not None:
            # include=True  -> matched an ignore pattern  -> excluded
            # include=False -> matched a negation pattern  -> re-included
            return result.include
    return False


def _walk_codebase(  # noqa: PLR0912
    codebase_path: Path,
    extensions: set[str],
    max_size: int,
    exclude_spec: pathspec.GitIgnoreSpec | None = None,
) -> Iterator[tuple[Path, str, os.stat_result]]:
    """Walk a codebase, yielding ``(abs_path, rel_path, stat)`` for indexable files.

    This is the single traversal shared by :func:`discover_files` and
    :func:`scan_file_metadata` so the two cannot drift apart. It handles directory
    pruning (always-excluded dirs, dot directories, symlinks, ignored directories),
    nested ``.gitignore`` / ``.codesearchignore`` accumulation plus
    ``.git/info/exclude``, symlink and hidden-file skipping, extension filtering,
    and size limits. Ignore files are honored at every directory level, matching
    git's "deeper files override shallower" semantics.
    """
    # Accumulated specs per directory, ordered root -> deep. Parent lists are
    # shared by reference when a directory adds no specs of its own.
    specs_by_dir: dict[Path, list[tuple[Path, pathspec.GitIgnoreSpec]]] = {}

    # followlinks=False prevents infinite loops from symlinks pointing to parents.
    for root, dirs, files in os.walk(codebase_path, followlinks=False):
        root_path = Path(root)
        rel_root = root_path.relative_to(codebase_path)

        if root_path == codebase_path:
            current_specs = _root_ignore_specs(codebase_path)
        else:
            parent_specs = specs_by_dir.get(root_path.parent, [])
            local_specs = _dir_ignore_specs(root_path)
            current_specs = parent_specs + local_specs if local_specs else parent_specs
        specs_by_dir[root_path] = current_specs

        # Prune directories in place: never-traversed dirs, dot dirs, symlinks, and
        # directories excluded by ignore rules (git does not descend into them).
        kept_dirs = []
        for d in dirs:
            if d in _ALWAYS_EXCLUDE_DIRS or d.startswith("."):
                continue
            dir_path = root_path / d
            if dir_path.is_symlink():
                continue
            # exclude_spec (the programmatic exclude_patterns) is intentionally NOT
            # applied to directory pruning: historically it matched at file level only,
            # so pruning here would change behavior (e.g. a later "!dir/keep.py" could no
            # longer re-include a child). Nested .gitignore / .codesearchignore specs DO
            # prune directories, matching git's non-recursion into excluded subtrees.
            if _is_ignored(dir_path, str(rel_root / d), current_specs, None, is_dir=True):
                continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs

        for filename in files:
            if filename.startswith("."):
                continue
            file_path = root_path / filename
            if file_path.is_symlink():
                continue
            if file_path.suffix.lower() not in extensions:
                continue
            rel_path = str(rel_root / filename)
            if _is_ignored(file_path, rel_path, current_specs, exclude_spec, is_dir=False):
                continue
            try:
                stat = file_path.stat()
            except OSError as e:
                logger.warning(f"Failed to stat file {rel_path}: {e}")
                continue
            if stat.st_size == 0 or stat.st_size > max_size:
                continue
            yield file_path, rel_path, stat


def discover_files(
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
        exclude_patterns: Additional gitignore-style patterns to exclude
        max_file_size_kb: Max file size in KB (default: from settings)

    Yields:
        FileInfo for each discovered file
    """
    codebase_path = Path(codebase_path).resolve()
    extensions = include_extensions or settings.code_extensions
    max_size = (max_file_size_kb or settings.max_file_size_kb) * 1024
    exclude_spec = (
        pathspec.GitIgnoreSpec.from_lines(exclude_patterns) if exclude_patterns else None
    )

    for file_path, rel_path, stat in _walk_codebase(
        codebase_path, extensions, max_size, exclude_spec
    ):
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
    Use for quick change detection before committing to full file reads. Honors
    the same directory pruning and ignore rules as discover_files(), so change
    detection and full indexing agree on which files are indexable.

    Yields:
        Tuple of (rel_path, mtime, size_bytes)
    """
    codebase_path = Path(codebase_path).resolve()
    extensions = include_extensions or settings.code_extensions
    max_size = (max_file_size_kb or settings.max_file_size_kb) * 1024

    for _file_path, rel_path, stat in _walk_codebase(codebase_path, extensions, max_size):
        yield (rel_path, stat.st_mtime, stat.st_size)
