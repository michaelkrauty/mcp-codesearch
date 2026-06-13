"""Tests for retrieval-layer filter pushdown (issue #25).

Path constraints (the exact changed-file set and file: pattern tokens)
must reach Qdrant as payload filter conditions on every retrieval call,
instead of only post-filtering a bounded candidate pool. These tests
drive the real QdrantStorage and search_codebase against mocked Qdrant
clients, including a simulated Qdrant whose scroll honors path filters,
to prove the #25 early-termination scenario is fixed.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.models import MatchAny, MatchText, MatchValue

from mcp_codesearch.search.query import search_codebase
from mcp_codesearch.storage.qdrant import QdrantStorage


def _point(point_id: int, **payload) -> SimpleNamespace:
    base = {"type": "chunk", "path": "src/x.py", "language": "python",
            "name": "", "summary": "", "content": ""}
    base.update(payload)
    return SimpleNamespace(id=point_id, payload=base)


@pytest.fixture
def storage():
    s = QdrantStorage(url="http://localhost:9999")  # never contacted
    s._client_mock = AsyncMock()
    s._get_client = AsyncMock(return_value=s._client_mock)  # type: ignore[method-assign]
    return s


def _scroll_filters(client) -> list:
    return [c.kwargs["scroll_filter"] for c in client.scroll.call_args_list]


def _path_conditions(flt) -> dict[str, object]:
    """Map of match-type name -> match object for path conditions in must."""
    out: dict[str, object] = {}
    for cond in flt.must or []:
        if cond.key == "path":
            out[type(cond.match).__name__] = cond.match
    return out


class TestPathPushdownConditions:
    def test_restrict_paths_builds_match_any(self):
        (cond,) = QdrantStorage._path_pushdown_conditions(["a.py", "b.py"], None)
        assert cond.key == "path"
        assert isinstance(cond.match, MatchAny)
        assert cond.match.any == ["a.py", "b.py"]

    def test_path_tokens_build_match_text(self):
        (cond,) = QdrantStorage._path_pushdown_conditions(None, ["db", "py"])
        assert cond.key == "path"
        assert isinstance(cond.match, MatchText)
        assert cond.match.text == "db py"

    def test_both_and_neither(self):
        assert QdrantStorage._path_pushdown_conditions(None, None) == []
        conds = QdrantStorage._path_pushdown_conditions(["a.py"], ["db", "py"])
        assert [type(c.match).__name__ for c in conds] == ["MatchAny", "MatchText"]


class TestExactMatchPushdown:
    """The #25 sharp edge: both the fast-path filter AND the exhaustive
    query_filter must carry the path constraints, otherwise the scroll
    loop's early termination can exhaust the scan budget on other files."""

    @pytest.mark.asyncio
    async def test_constraints_in_fast_and_exhaustive_filters(self, storage):
        storage._client_mock.scroll = AsyncMock(return_value=([], None))

        await storage.exact_match_search(
            "col", "foo", mode="chunk",
            restrict_paths=["a.py", "b.py"], path_text_tokens=["db", "py"],
        )

        fast, exhaustive = _scroll_filters(storage._client_mock)
        assert fast.should  # fast path ran first
        assert not getattr(exhaustive, "should", None)
        for flt in (fast, exhaustive):
            conds = _path_conditions(flt)
            assert conds["MatchAny"].any == ["a.py", "b.py"]
            assert conds["MatchText"].text == "db py"
            # path must never join the should-conditions: that would admit
            # path-token candidates into the content/name/summary scan.
            assert all(c.key != "path" for c in (flt.should or []))

    @pytest.mark.asyncio
    async def test_constraints_present_when_fast_path_ineligible(self, storage):
        """Queries that skip the MatchText fast path still push paths down."""
        storage._client_mock.scroll = AsyncMock(return_value=([], None))

        await storage.exact_match_search(
            "col", "()", restrict_paths=["a.py"], path_text_tokens=["db", "py"],
        )

        (flt,) = _scroll_filters(storage._client_mock)
        conds = _path_conditions(flt)
        assert conds["MatchAny"].any == ["a.py"]
        assert conds["MatchText"].text == "db py"

    @pytest.mark.asyncio
    async def test_no_constraints_no_path_conditions(self, storage):
        storage._client_mock.scroll = AsyncMock(return_value=([], None))

        await storage.exact_match_search("col", "foo")

        for flt in _scroll_filters(storage._client_mock):
            if flt is not None:
                assert _path_conditions(flt) == {}


class TestSparseOnlyPushdown:
    @pytest.mark.asyncio
    async def test_conditions_in_query_filter(self, storage):
        storage._client_mock.query_points = AsyncMock(
            return_value=SimpleNamespace(points=[])
        )
        sparse = SimpleNamespace(indices=[1], values=[1.0])

        await storage.sparse_only_search(
            "col", sparse, restrict_paths=["a.py"], path_text_tokens=["db", "py"],
        )

        flt = storage._client_mock.query_points.call_args.kwargs["query_filter"]
        conds = _path_conditions(flt)
        assert conds["MatchAny"].any == ["a.py"]
        assert conds["MatchText"].text == "db py"


class TestHybridPushdown:
    @pytest.mark.asyncio
    async def test_conditions_in_filter_conditions(self, storage):
        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=[])
        with patch(
            "mcp_codesearch.storage.qdrant.HybridSearcher", return_value=searcher
        ):
            await storage.hybrid_search(
                "col", [0.1], SimpleNamespace(indices=[1], values=[1.0]),
                restrict_paths=["a.py"], path_text_tokens=["db", "py"],
            )

        conds = searcher.search.call_args.kwargs["filter_conditions"]
        path_conds = {type(c.match).__name__: c.match for c in conds if c.key == "path"}
        assert path_conds["MatchAny"].any == ["a.py"]
        assert path_conds["MatchText"].text == "db py"


class TestPathIndexCreation:
    @pytest.mark.asyncio
    async def test_ensure_text_indexes_includes_path(self, storage):
        assert await storage._ensure_text_indexes("col") is True

        created = [
            c.kwargs["field_name"]
            for c in storage._client_mock.create_payload_index.call_args_list
        ]
        assert set(created) == {"content", "name", "summary", "path"}
        assert len(created) == 4

    def test_should_condition_fields_exclude_path(self, storage):
        """_TEXT_INDEX_FIELDS feeds exact-match should-conditions and must
        not contain path, or path-token hits would enter content scans."""
        assert "path" not in storage._TEXT_INDEX_FIELDS
        assert set(storage._TEXT_INDEXED_PAYLOAD_FIELDS) == {
            *storage._TEXT_INDEX_FIELDS, "path",
        }


def _filtering_scroll(points: list[SimpleNamespace], ascii_tokens: bool = False):
    """A fake Qdrant scroll that honors path/type must-conditions.

    Implements MatchValue, MatchAny (exact full-value set), and MatchText
    on path (whole-token AND, word tokenizer = maximal alnum runs).
    should-conditions are treated as match-all, which only widens the
    candidate set the scoring loop then narrows. ``ascii_tokens``
    simulates a tokenizer that splits on non-ASCII characters, the kind
    of unforeseen edge the zero-result fallback exists for.
    """
    token_re = re.compile(r"[a-z0-9]+" if ascii_tokens else r"[^\W_]+", re.UNICODE)

    def path_tokens(path: str) -> set[str]:
        return set(token_re.findall(path.lower()))

    def matches(p: SimpleNamespace, flt) -> bool:
        if flt is None:
            return True
        for cond in flt.must or []:
            value = p.payload.get(cond.key)
            match = cond.match
            if isinstance(match, MatchValue):
                if value != match.value:
                    return False
            elif isinstance(match, MatchAny):
                if value not in match.any:
                    return False
            elif isinstance(match, MatchText) and cond.key == "path":
                wanted = match.text.lower().split()
                if not all(t in path_tokens(str(value)) for t in wanted):
                    return False
        return True

    async def scroll(collection, scroll_filter=None, limit=10, offset=None,
                     with_payload=None):
        eligible = [p for p in points if matches(p, scroll_filter)]
        start = offset or 0
        page = eligible[start:start + limit]
        next_offset = start + limit if start + limit < len(eligible) else None
        return page, next_offset

    return AsyncMock(side_effect=scroll)


def _early_termination_dataset() -> list[SimpleNamespace]:
    """A collection where 'init' hits in other files fill the first scroll
    page (1000 points) and trip exact-match early termination (10
    high-quality name hits) before src/db.py is ever reached."""
    points = [
        _point(i, path=f"src/other{i}.rs", name="init", content="fn init() {}")
        for i in range(12)
    ]
    points += [
        _point(i, path=f"src/filler{i}.rs", content="nothing relevant")
        for i in range(12, 1000)
    ]
    points.append(
        _point(1000, path="src/db.py", name="init", content="def init(): pass")
    )
    return points


class TestIssue25EndToEnd:
    """fn:init file:db.py must find db.py's init even when other files'
    init hits would exhaust the early-termination scan budget."""

    @pytest.mark.asyncio
    async def test_file_pattern_survives_early_termination(self, storage):
        storage._client_mock.scroll = _filtering_scroll(_early_termination_dataset())

        results = await search_codebase(
            query="fn:init file:db.py",
            codebase_path="/tmp",
            storage=storage,
            embedder=MagicMock(),  # NAME_ONLY plan never embeds
            global_vocab=MagicMock(),
        )

        assert [r.path for r in results] == ["src/db.py"]
        assert results[0].name == "init"

    @pytest.mark.asyncio
    async def test_restrict_paths_survives_early_termination(self, storage):
        """The search_changed shape: an exact path set instead of file:."""
        storage._client_mock.scroll = _filtering_scroll(_early_termination_dataset())

        results = await search_codebase(
            query="fn:init",
            codebase_path="/tmp",
            storage=storage,
            embedder=MagicMock(),
            global_vocab=MagicMock(),
            restrict_paths=["src/db.py"],
        )

        assert [r.path for r in results] == ["src/db.py"]

    @pytest.mark.asyncio
    async def test_pushdown_skipped_when_index_creation_fails(self, storage):
        """No text index => MatchText on path is untrusted and must not be
        sent; the query falls back to post-filtering only."""
        storage._client_mock.create_payload_index = AsyncMock(
            side_effect=RuntimeError("forbidden")
        )
        storage._client_mock.scroll = _filtering_scroll(_early_termination_dataset())

        results = await search_codebase(
            query="fn:init file:db.py",
            codebase_path="/tmp",
            storage=storage,
            embedder=MagicMock(),
            global_vocab=MagicMock(),
        )

        # Without the index gate the pushdown cannot run; with the budget
        # exhausted by other files the post-filter finds nothing. The point
        # of this test is the absence of MatchText, not the empty result.
        assert results == []
        for flt in _scroll_filters(storage._client_mock):
            if flt is not None:
                assert "MatchText" not in _path_conditions(flt)


class TestZeroResultFallback:
    @pytest.mark.asyncio
    async def test_tokenizer_edge_case_recovered_without_tokens(self, storage):
        """If the token pushdown unexpectedly excludes the true match (here:
        a tokenizer that splits unicode differently than assumed), the
        search reruns once without path tokens and the post-filter still
        delivers the match."""
        points = [
            _point(1, path="src/naïve.py", name="init", content="def init(): pass"),
        ]
        # ascii_tokens=True: the fake Qdrant tokenizes "naïve" as "na"+"ve",
        # so MatchText(text="naïve py") matches nothing.
        storage._client_mock.scroll = _filtering_scroll(points, ascii_tokens=True)

        results = await search_codebase(
            query="fn:init file:naïve.py",
            codebase_path="/tmp",
            storage=storage,
            embedder=MagicMock(),
            global_vocab=MagicMock(),
        )

        assert [r.path for r in results] == ["src/naïve.py"]
