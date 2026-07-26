"""A failed incremental index must leave the existing index serving queries.

Embedding is the slow, network-dependent step of an incremental run, and the
one most likely to fail. These tests inject that failure against a real
collection and assert that the points which were already there are still
there afterwards.

This is the behaviour the per-batch swap exists to provide. Under the previous
ordering -- delete every changed file's points, then embed -- the same failure
left those files with no points at all until a later run happened to succeed.
"""

from unittest.mock import AsyncMock

import httpx
import pytest

from mcp_codesearch import singletons
from mcp_codesearch.server import code_search
from mcp_codesearch.settings import settings
from mcp_codesearch.storage.qdrant import collection_name

from .conftest import requires_full_stack


async def _point_count(collection: str) -> int:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.qdrant_url}/collections/{collection}", timeout=10.0
        )
        response.raise_for_status()
        return response.json()["result"]["points_count"]


def _rewrite_every_file(root) -> None:
    """Give every source file new content, so all of them need re-indexing."""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".ts", ".js", ".md"}:
            path.write_text(
                path.read_text() + "\n\n# marker_for_the_failed_run\n",
                encoding="utf-8",
            )


@requires_full_stack
class TestIncrementalFailureIsNonDestructive:
    """An embedding failure must not cost the index its existing content."""

    async def test_existing_points_survive_an_embedding_failure(
        self, temp_codebase, monkeypatch
    ):
        collection = collection_name(str(temp_codebase))

        # Index for real, then take the state we expect to still be there.
        await code_search(query="main entry point", path=str(temp_codebase))
        before = await _point_count(collection)
        assert before > 0, "nothing was indexed, so the test proves nothing"

        _rewrite_every_file(temp_codebase)

        # Fail the way a saturated or unreachable embedding service fails,
        # after change detection and chunking, before anything is written.
        embedder = await singletons.get_embedder()
        monkeypatch.setattr(
            embedder,
            "embed_all",
            AsyncMock(side_effect=RuntimeError("injected embedding failure")),
        )

        with pytest.raises(Exception, match="injected embedding failure"):
            await code_search(query="main entry point", path=str(temp_codebase))

        # The whole point: the previous points are untouched, so the index
        # answers exactly as it did before the failed run.
        assert await _point_count(collection) == before

        monkeypatch.undo()
        recovered = await code_search(
            query="marker_for_the_failed_run", path=str(temp_codebase)
        )
        assert "error" not in recovered.lower()
        assert await _point_count(collection) > 0

    async def test_a_failed_run_does_not_compound_across_retries(
        self, temp_codebase, monkeypatch
    ):
        """Retrying a failing run must not erode the index a little each time.

        The previous ordering deleted before embedding with no restore, so each
        attempt removed another set of points that nothing added back.
        """
        collection = collection_name(str(temp_codebase))

        await code_search(query="main entry point", path=str(temp_codebase))
        before = await _point_count(collection)
        assert before > 0

        _rewrite_every_file(temp_codebase)

        embedder = await singletons.get_embedder()
        monkeypatch.setattr(
            embedder,
            "embed_all",
            AsyncMock(side_effect=RuntimeError("injected embedding failure")),
        )

        for _ in range(3):
            with pytest.raises(Exception, match="injected embedding failure"):
                await code_search(query="main entry point", path=str(temp_codebase))
            assert await _point_count(collection) == before
