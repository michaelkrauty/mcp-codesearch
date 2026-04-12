"""Unit tests for IndexingService helpers that do not require Qdrant/embeddings."""

from pathlib import Path
from unittest.mock import MagicMock

from mcp_codesearch.indexer.discovery import FileInfo
from mcp_codesearch.services import indexing_service as idx_svc
from mcp_codesearch.services.indexing_service import IndexingService


def _make_file(name: str, content: str) -> FileInfo:
    """Build a FileInfo suitable for _prepare_files without touching disk."""
    return FileInfo(
        path=Path(f"/tmp/{name}"),
        rel_path=name,
        language="python",
        size_bytes=len(content),
        content=content,
        content_hash="deadbeef",
        line_count=content.count("\n") + 1,
        mtime=0.0,
    )


def _make_service() -> IndexingService:
    """Construct an IndexingService with stub dependencies.

    _prepare_files only touches self._global_vocab.tokenize and the static
    _chunk_embedding_text method, so the other deps can be bare mocks.
    """
    vocab = MagicMock()
    vocab.tokenize = MagicMock(return_value=[])
    return IndexingService(
        storage=MagicMock(),
        embedder=MagicMock(),
        global_vocab=vocab,
    )


class TestPrepareFilesFaultIsolation:
    """One unparseable file must not abort the whole indexing run."""

    def test_skips_file_when_chunker_raises(self, monkeypatch, caplog):
        """A chunker crash on one file logs a warning and lets the loop continue."""
        service = _make_service()

        good1 = _make_file("good1.py", "def a():\n    pass\n")
        bad = _make_file("bad.py", "def b():\n    pass\n")
        good2 = _make_file("good2.py", "def c():\n    pass\n")

        real_chunk_file = idx_svc.chunk_file

        def failing_chunk_file(content, language):
            if content == bad.content:
                raise RecursionError("simulated pathological file")
            return real_chunk_file(content, language)

        monkeypatch.setattr(idx_svc, "chunk_file", failing_chunk_file)

        with caplog.at_level("WARNING", logger="mcp_codesearch.services.indexing_service"):
            prepared, tokens_per_doc = service._prepare_files([good1, bad, good2])

        rel_paths = [p.file_info.rel_path for p in prepared]
        assert rel_paths == ["good1.py", "good2.py"]
        # The bad file should be absent from token rows too (one summary + N chunks per good file).
        assert len(tokens_per_doc) >= 2

        warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("bad.py" in m for m in warnings), warnings
        assert any("RecursionError" in m for m in warnings), warnings

    def test_all_files_prepared_when_chunker_is_healthy(self):
        """Sanity check: the happy path still works after adding the try/except."""
        service = _make_service()
        files = [
            _make_file("a.py", "def a():\n    pass\n"),
            _make_file("b.py", "def b():\n    pass\n"),
        ]

        prepared, _ = service._prepare_files(files)

        assert [p.file_info.rel_path for p in prepared] == ["a.py", "b.py"]
        # Each file should produce at least one chunk.
        assert all(len(p.chunks) >= 1 for p in prepared)
