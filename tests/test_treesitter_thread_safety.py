"""Concurrency tests for cached tree-sitter parsers."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from mcp_codesearch.indexer import treesitter


def test_same_language_parser_is_not_driven_concurrently(monkeypatch) -> None:
    class RecordingParser:
        def __init__(self) -> None:
            self._guard = threading.Lock()
            self._active = 0
            self.overlapped = False

        def parse(self, _source: bytes) -> SimpleNamespace:
            with self._guard:
                self._active += 1
                self.overlapped |= self._active > 1
            time.sleep(0.05)
            with self._guard:
                self._active -= 1
            root = SimpleNamespace(is_named=False, children=[])
            return SimpleNamespace(root_node=root)

    parser = RecordingParser()
    monkeypatch.setattr(treesitter, "_parser_cache", {"python": parser})
    monkeypatch.setattr(treesitter, "_parser_locks", {})

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda content: treesitter.chunk_with_treesitter(content, "python"),
                ("def first(): pass", "def second(): pass"),
            )
        )

    assert results == [[], []]
    assert not parser.overlapped
