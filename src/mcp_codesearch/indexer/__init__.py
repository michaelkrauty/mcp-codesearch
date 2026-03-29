"""Indexing pipeline components."""

from .change_detect import detect_changes
from .chunker import chunk_file
from .discovery import discover_files

__all__ = ["discover_files", "chunk_file", "detect_changes"]
