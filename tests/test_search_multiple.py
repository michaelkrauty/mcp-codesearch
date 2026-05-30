"""Unit tests for ``search_multiple``: concurrency, per-codebase grouping, and the
cross-codebase ``global_ranking`` path.

These tests mock the heavy boundaries (auto-indexing, the search service, the
codebase search, and the storage/embedder/vocabulary singletons) so they run with
no Qdrant or embedding service and assert behaviour, ordering, and output shape.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_codesearch.services.search_service import SearchResponse
from mcp_codesearch.storage.qdrant import SearchResult
from mcp_codesearch.tools.search import (
    _first_line,
    _format_global,
    _global_key,
    _SourcedResult,
    search_multiple,
)

MOD = "mcp_codesearch.tools.search"


def _result(
    path: str,
    score: float,
    *,
    point_type: str = "chunk",
    name: str | None = None,
    language: str = "python",
    start_line: int = 1,
    end_line: int = 5,
    content: str | None = "code",
    summary: str | None = None,
    line_count: int | None = None,
) -> SearchResult:
    """Build a SearchResult with sensible defaults for the fields the formatters read."""
    return SearchResult(
        path=path,
        score=score,
        point_type=point_type,
        language=language,
        name=name,
        start_line=start_line,
        end_line=end_line,
        content=content,
        summary=summary,
        line_count=line_count,
        chunk_type="function" if point_type == "chunk" else None,
    )


def _stats(time_ms: int = 5) -> MagicMock:
    stats = MagicMock()
    stats.files_deleted = 0
    stats.indexing_time_ms = time_ms
    return stats


@contextmanager
def _boundaries(
    *,
    auto_index: AsyncMock,
    get_search_service: AsyncMock | None = None,
    search_codebase: AsyncMock | None = None,
) -> Iterator[None]:
    """Patch every external boundary ``search_multiple`` touches.

    The storage/embedder/vocabulary singletons are always replaced with trivial async
    mocks; callers override only the pieces a given test cares about.
    """
    mocks = {
        "auto_index": auto_index,
        "get_search_service": get_search_service or AsyncMock(return_value=MagicMock()),
        "search_codebase": search_codebase or AsyncMock(return_value=[]),
        "get_storage": AsyncMock(return_value=MagicMock()),
        "get_embedder": AsyncMock(return_value=MagicMock()),
        "get_global_vocab": AsyncMock(return_value=MagicMock()),
    }
    with ExitStack() as stack:
        for name, mock in mocks.items():
            stack.enter_context(patch(f"{MOD}.{name}", new=mock))
        yield


def _two_codebases(tmp_path) -> tuple[str, str]:
    a = tmp_path / "repo_a"
    a.mkdir()
    b = tmp_path / "repo_b"
    b.mkdir()
    return str(a), str(b)


# --------------------------------------------------------------------------- #
# Grouped (default) behaviour
# --------------------------------------------------------------------------- #


async def test_grouped_preserves_input_order_under_concurrency(tmp_path):
    """Sections appear in input order even when an earlier codebase resolves later."""
    a, b = _two_codebases(tmp_path)

    async def fake_search(query, skip_cache=False):
        # Make the FIRST codebase the slowest; gather must still order it first.
        await asyncio.sleep(0.05 if a in query.path else 0.0)
        return SearchResponse(formatted_output=f"HIT::{query.path}\n", results_count=1)

    svc = MagicMock()
    svc.search = AsyncMock(side_effect=fake_search)

    with _boundaries(
        auto_index=AsyncMock(return_value=(0, 0, _stats(), "")),
        get_search_service=AsyncMock(return_value=svc),
    ):
        out = await search_multiple(query="anything", paths=[a, b])

    assert out.index(f"=== {a} ===") < out.index(f"=== {b} ===")
    assert f"HIT::{a}" in out and f"HIT::{b}" in out


async def test_grouped_runs_concurrently(tmp_path):
    """Searches overlap: peak in-flight count equals the number of codebases.

    Structural rather than wall-clock based — each search holds its slot across an
    ``await`` while siblings start, so a sequential implementation would peak at 1.
    """
    paths = []
    for i in range(5):
        d = tmp_path / f"r{i}"
        d.mkdir()
        paths.append(str(d))

    active = 0
    peak = 0

    async def fake_search(query, skip_cache=False):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.02)  # hold the slot so siblings overlap
            return SearchResponse(formatted_output="HIT\n", results_count=1)
        finally:
            active -= 1

    svc = MagicMock()
    svc.search = AsyncMock(side_effect=fake_search)

    with _boundaries(
        auto_index=AsyncMock(return_value=(0, 0, _stats(), "")),
        get_search_service=AsyncMock(return_value=svc),
    ):
        out = await search_multiple(query="anything", paths=paths)

    assert out.count("HIT") == 5
    assert peak == 5, f"expected all 5 searches in flight at once, peak was {peak}"


async def test_grouped_error_in_one_codebase_isolated(tmp_path):
    """An indexing error in one codebase surfaces inline without blocking the others."""
    a, b = _two_codebases(tmp_path)

    async def fake_index(path):
        if "repo_a" in path:
            return (0, 0, None, "Error: Qdrant unavailable for repo_a")
        return (0, 0, _stats(), "")

    svc = MagicMock()
    svc.search = AsyncMock(
        return_value=SearchResponse(formatted_output="HIT_B\n", results_count=1)
    )

    with _boundaries(
        auto_index=AsyncMock(side_effect=fake_index),
        get_search_service=AsyncMock(return_value=svc),
    ):
        out = await search_multiple(query="anything", paths=[a, b])

    assert "Qdrant unavailable for repo_a" in out
    assert "HIT_B" in out
    assert out.index(f"=== {a} ===") < out.index(f"=== {b} ===")


async def test_grouped_unexpected_exception_does_not_leak_or_abort(tmp_path):
    """A raised exception becomes a generic per-section error; siblings still run."""
    a, b = _two_codebases(tmp_path)

    async def fake_search(query, skip_cache=False):
        if a in query.path:
            raise RuntimeError("boom-internal-detail")
        return SearchResponse(formatted_output="HIT_B\n", results_count=1)

    svc = MagicMock()
    svc.search = AsyncMock(side_effect=fake_search)

    with _boundaries(
        auto_index=AsyncMock(return_value=(0, 0, _stats(), "")),
        get_search_service=AsyncMock(return_value=svc),
    ):
        out = await search_multiple(query="anything", paths=[a, b])

    assert "Search failed" in out
    assert "boom-internal-detail" not in out  # internal detail not leaked to caller
    assert "HIT_B" in out


async def test_grouped_shows_cached_results(tmp_path):
    """Regression: a cache hit (results_count == 0) must still render its results."""
    (a,) = (_two_codebases(tmp_path)[0],)

    # A cached SearchResponse carries formatted_output but leaves results_count at 0.
    svc = MagicMock()
    svc.search = AsyncMock(
        return_value=SearchResponse(
            formatted_output="1. [python] cached.py:1-5\n", was_cached=True
        )
    )

    with _boundaries(
        auto_index=AsyncMock(return_value=(0, 0, _stats(), "")),
        get_search_service=AsyncMock(return_value=svc),
    ):
        out = await search_multiple(query="anything", paths=[a])

    assert "cached.py" in out
    assert "No results found" not in out


# --------------------------------------------------------------------------- #
# global_ranking behaviour
# --------------------------------------------------------------------------- #


async def test_global_ranking_merges_and_tags_sources(tmp_path):
    """Results merge into one ranked list, each tagged with its source codebase."""
    a, b = _two_codebases(tmp_path)

    async def fake_codebase(query, codebase_path, **kwargs):
        if a in codebase_path:
            return [_result("a1.py", 0.9, name="alpha"), _result("a2.py", 0.5, name="beta")]
        return [_result("b1.py", 0.8, name="gamma")]

    with _boundaries(
        auto_index=AsyncMock(return_value=(0, 0, _stats(), "")),
        search_codebase=AsyncMock(side_effect=fake_codebase),
    ):
        out = await search_multiple(
            query="anything", paths=[a, b], global_ranking=True
        )

    assert "=== " not in out  # single global list, not per-codebase sections
    assert "alpha" in out and "beta" in out and "gamma" in out
    assert f"({a})" in out and f"({b})" in out  # source tags
    # Each codebase's rank-1 (alpha, gamma) outranks the rank-2 result (beta).
    assert out.index("alpha") < out.index("beta")
    assert out.index("gamma") < out.index("beta")


async def test_global_ranking_dedupes_nested_codebase(tmp_path):
    """The same physical file indexed under a parent and a nested repo collapses to one."""
    parent = tmp_path / "parent"
    parent.mkdir()
    sub = parent / "sub"
    sub.mkdir()

    async def fake_codebase(query, codebase_path, **kwargs):
        if codebase_path == str(sub.resolve()):
            return [_result("x.py", 0.9, name="dup")]
        return [_result("sub/x.py", 0.7, name="dup")]

    with _boundaries(
        auto_index=AsyncMock(return_value=(0, 0, _stats(), "")),
        search_codebase=AsyncMock(side_effect=fake_codebase),
    ):
        out = await search_multiple(
            query="anything",
            paths=[str(parent), str(sub)],
            global_ranking=True,
            output_format="json",
        )

    data = json.loads(out)
    assert len(data["results"]) == 1  # deduped: parent/sub/x.py appears once


async def test_global_ranking_json_shape_and_skipped(tmp_path):
    """JSON output carries rank/source/scores and lists skipped codebases (first line)."""
    a, b = _two_codebases(tmp_path)

    async def fake_index(path):
        if "repo_b" in path:
            return (0, 0, None, "Error: embedding service down\nstack trace detail")
        return (0, 0, _stats(), "")

    async def fake_codebase(query, codebase_path, **kwargs):
        return [_result("a1.py", 0.9, name="alpha")]

    with _boundaries(
        auto_index=AsyncMock(side_effect=fake_index),
        search_codebase=AsyncMock(side_effect=fake_codebase),
    ):
        out = await search_multiple(
            query="anything", paths=[a, b], global_ranking=True, output_format="json"
        )

    data = json.loads(out)
    assert len(data["results"]) == 1
    top = data["results"][0]
    assert top["source"] == a
    assert top["rank"] == 1
    assert "rrf_score" in top and "score" in top
    assert data["skipped"][0]["path"] == b
    assert data["skipped"][0]["error"] == "Error: embedding service down"  # first line only


async def test_global_ranking_all_codebases_failed(tmp_path):
    """When every codebase errors, report no results plus a skipped footer."""
    a, b = _two_codebases(tmp_path)

    with _boundaries(
        auto_index=AsyncMock(return_value=(0, 0, None, "Error: boom")),
        search_codebase=AsyncMock(return_value=[]),
    ):
        out = await search_multiple(
            query="anything", paths=[a, b], global_ranking=True
        )

    assert "No results found" in out
    assert "Skipped 2 codebase(s)" in out
    assert a in out and b in out


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_global_key_collapses_same_physical_file():
    parent = _SourcedResult(
        source="parent",
        abs_source="/a/b",
        result=_result("sub/x.py", 0.9, start_line=1, end_line=5),
    )
    sub = _SourcedResult(
        source="sub",
        abs_source="/a/b/sub",
        result=_result("x.py", 0.7, start_line=1, end_line=5),
    )
    other = _SourcedResult(
        source="sub",
        abs_source="/a/b/sub",
        result=_result("y.py", 0.7, start_line=1, end_line=5),
    )
    assert _global_key(parent) == _global_key(sub)
    assert _global_key(sub) != _global_key(other)


def test_format_global_empty():
    assert _format_global([], "text", []) == "No results found."
    assert _format_global([], "json", []) == json.dumps({"results": []}, indent=2)


def test_first_line():
    assert _first_line("first\nsecond\nthird") == "first"
    assert _first_line("\n\n  hello \nworld") == "hello"
    assert _first_line("   ") == ""
