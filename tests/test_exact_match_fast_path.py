"""Tests for the full-text pre-filtered exact-match fast path.

The fast path must never change WHAT exact_match_search returns — only
how many points get scanned. These tests drive the real QdrantStorage
methods against a mocked Qdrant client and assert on the filters sent to
scroll, the index-creation calls, and the fallback wiring.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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
    """scroll_filter kwarg of each scroll call, in order."""
    return [c.kwargs["scroll_filter"] for c in client.scroll.call_args_list]


class TestPrefilterApplicable:
    """Gate: the fast path may only run when it provably cannot drop matches."""

    @pytest.mark.parametrize("query", ["foo", "foo_bar", "a", "x" * 64, "Foo.bar()", "naïve"])
    def test_eligible_queries(self, storage, query):
        assert storage._text_prefilter_applicable(query) is True

    @pytest.mark.parametrize("query", ["()", "...", "  ", "", "x" * 65, f"ok {'y' * 65}"])
    def test_ineligible_queries(self, storage, query):
        assert storage._text_prefilter_applicable(query) is False


class TestFastPathWiring:
    @pytest.mark.asyncio
    async def test_eligible_query_scrolls_with_text_prefilter(self, storage):
        """First scroll carries should=[MatchText x3] plus the must conditions."""
        storage._client_mock.scroll = AsyncMock(
            return_value=([_point(1, content="def foo(): pass")], None)
        )

        results = await storage.exact_match_search("col", "foo", mode="chunk")

        assert [r.path for r in results] == ["src/x.py"]
        (flt,) = _scroll_filters(storage._client_mock)
        assert {c.key for c in flt.should} == {"content", "name", "summary"}
        assert all(c.match.text == "foo" for c in flt.should)
        assert [c.key for c in flt.must] == ["type"]

    @pytest.mark.asyncio
    async def test_zero_fast_path_results_falls_back_to_exhaustive_scan(self, storage):
        """A miss on the pre-filter triggers one legacy scan with no should."""
        big = "x" * 60000 + " foo " + "y" * 100  # substring-fallback territory
        storage._client_mock.scroll = AsyncMock(
            side_effect=[([], None), ([_point(2, content=big)], None)]
        )

        results = await storage.exact_match_search("col", "foo", mode="chunk")

        assert len(results) == 1
        fast, legacy = _scroll_filters(storage._client_mock)
        assert fast.should and not getattr(legacy, "should", None)

    @pytest.mark.asyncio
    async def test_fast_path_error_falls_back_to_exhaustive_scan(self, storage):
        storage._client_mock.scroll = AsyncMock(
            side_effect=[RuntimeError("index exploded"),
                         ([_point(3, name="foo")], None)]
        )

        results = await storage.exact_match_search("col", "foo")

        assert [r.score for r in results] == [3.0]
        assert storage._client_mock.scroll.call_count == 2

    @pytest.mark.asyncio
    async def test_ineligible_query_uses_single_unfiltered_scan(self, storage):
        storage._client_mock.scroll = AsyncMock(
            return_value=([_point(4, content="weird () token")], None)
        )

        await storage.exact_match_search("col", "()")

        (flt,) = _scroll_filters(storage._client_mock)
        assert flt is None or not getattr(flt, "should", None)
        storage._client_mock.create_payload_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_fast_path_results_match_legacy_scoring(self, storage):
        """Scoring (name=3.0 > summary=2.0 > content=1.0) is untouched."""
        pts = [
            _point(1, name="foo"),
            _point(2, summary="about foo"),
            _point(3, content="foo()"),
        ]
        storage._client_mock.scroll = AsyncMock(return_value=(pts, None))

        results = await storage.exact_match_search("col", "foo")

        assert [r.score for r in results] == [3.0, 2.0, 1.0]


class TestEnsureTextIndexes:
    @pytest.mark.asyncio
    async def test_creates_four_indexes_once_per_collection(self, storage):
        storage._client_mock.scroll = AsyncMock(return_value=([], None))

        await storage.exact_match_search("col", "foo")
        await storage.exact_match_search("col", "bar")

        created = {
            c.kwargs["field_name"]
            for c in storage._client_mock.create_payload_index.call_args_list
        }
        assert created == {"content", "name", "summary", "path"}
        assert storage._client_mock.create_payload_index.call_count == 4

    @pytest.mark.asyncio
    async def test_index_params_pin_the_documented_contract(self, storage):
        """min_token_len=1 and max_token_len matching the gate constant."""
        storage._client_mock.scroll = AsyncMock(return_value=([], None))

        await storage.exact_match_search("col", "foo")

        schema = storage._client_mock.create_payload_index.call_args.kwargs["field_schema"]
        assert schema.min_token_len == 1
        assert schema.max_token_len == storage._TEXT_INDEX_MAX_TOKEN_LEN
        assert schema.lowercase is True

    @pytest.mark.asyncio
    async def test_index_creation_failure_disables_fast_path(self, storage):
        """A failed index build must route to the exhaustive scan: a
        partially-indexed MatchText filter could return a non-empty
        subset and wrongly suppress the fallback."""
        storage._client_mock.create_payload_index = AsyncMock(
            side_effect=RuntimeError("forbidden")
        )
        storage._client_mock.scroll = AsyncMock(
            return_value=([_point(1, content="foo")], None)
        )

        results = await storage.exact_match_search("col", "foo")

        assert len(results) == 1  # search proceeded despite index failure
        (flt,) = _scroll_filters(storage._client_mock)
        assert not getattr(flt, "should", None)  # exhaustive, not fast path

        # Second search: no creation retry (the loop aborted on the first
        # field's failure, so exactly one attempt was made), still exhaustive.
        await storage.exact_match_search("col", "bar")
        assert storage._client_mock.create_payload_index.call_count == 1
        assert not any(getattr(f, "should", None) for f in _scroll_filters(storage._client_mock))

    @pytest.mark.asyncio
    async def test_create_collection_ensures_indexes(self, storage):
        storage._core.create_collection = AsyncMock()

        await storage.create_collection("newcol")

        assert storage._client_mock.create_payload_index.call_count == 4

    @pytest.mark.asyncio
    async def test_delete_collection_invalidates_index_cache(self, storage):
        """A same-process delete + recreate must re-create the text indexes
        (force_reindex path); Qdrant drops them with the collection."""
        storage._core.create_collection = AsyncMock()
        storage._core.delete_collection = AsyncMock()

        await storage.create_collection("col")
        await storage.delete_collection("col")
        await storage.create_collection("col")

        assert storage._client_mock.create_payload_index.call_count == 8


class TestExactMatchRanking:
    """The scan returns the highest-scored `limit` matches, not the first
    `limit` reached in scroll order (which is point-id order, unrelated to
    match quality). Without ranking, a lone name match for a widely-referenced
    symbol is crowded out by content matches that scroll ahead of it — e.g.
    `cls:Foo` surfacing files that mention Foo but never its definition.
    """

    @pytest.mark.asyncio
    async def test_late_name_match_not_crowded_out_by_earlier_content(self, storage):
        # Five content matches (score 1.0) scroll before the one name match
        # (score 3.0); limit is smaller than the content run.
        pts = [_point(i, content="uses foo() here") for i in range(1, 6)]
        pts.append(_point(99, name="foo", content="def foo(): ..."))
        storage._client_mock.scroll = AsyncMock(return_value=(pts, None))

        results = await storage.exact_match_search("col", "foo", mode="chunk", limit=3)

        assert len(results) == 3
        assert results[0].score == 3.0  # the definition ranks first
        assert any(r.name == "foo" for r in results)  # and is actually present

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_descending(self, storage):
        pts = [
            _point(1, content="foo()"),       # 1.0
            _point(2, name="foo"),            # 3.0
            _point(3, summary="re: foo"),     # 2.0
            _point(4, content="also foo"),    # 1.0
        ]
        storage._client_mock.scroll = AsyncMock(return_value=(pts, None))

        results = await storage.exact_match_search("col", "foo", mode="chunk", limit=10)

        assert [r.score for r in results] == [3.0, 2.0, 1.0, 1.0]

    @pytest.mark.asyncio
    async def test_limit_still_caps_returned_results(self, storage):
        pts = [_point(i, name="foo") for i in range(1, 8)]
        storage._client_mock.scroll = AsyncMock(return_value=(pts, None))

        results = await storage.exact_match_search("col", "foo", mode="chunk", limit=2)

        assert len(results) == 2


class TestExactMatchRankFalse:
    """rank=False keeps scroll order so callers that discard the top score
    tier (find_references drops the name-equal definition) still see the
    lower-tier matches they want, instead of a pool ranked to all-definitions.
    """

    @pytest.mark.asyncio
    async def test_rank_false_preserves_scroll_order(self, storage):
        # Content references scroll first, definitions (name matches) after.
        pts = [_point(i, content="calls foo()") for i in range(1, 4)]
        pts += [_point(50 + i, name="foo") for i in range(1, 6)]
        storage._client_mock.scroll = AsyncMock(return_value=(pts, None))

        results = await storage.exact_match_search(
            "col", "foo", mode="chunk", limit=3, rank=False
        )

        # First three in scroll order — the content references, NOT the
        # higher-scored definitions that scroll later.
        assert [r.score for r in results] == [1.0, 1.0, 1.0]

    @pytest.mark.asyncio
    async def test_rank_false_does_not_starve_references(self, storage):
        # Mirrors find_references: a flood of same-named definitions plus a few
        # references. rank=False must still surface references for the caller.
        refs = [_point(i, content="uses foo here") for i in range(1, 4)]
        defs = [_point(100 + i, name="foo") for i in range(60)]
        storage._client_mock.scroll = AsyncMock(return_value=(refs + defs, None))

        results = await storage.exact_match_search(
            "col", "foo", mode="chunk", limit=9, rank=False
        )
        content_refs = [r for r in results if r.score == 1.0]
        assert len(content_refs) == 3  # all references present, not ranked away

    @pytest.mark.asyncio
    async def test_rank_false_does_not_early_terminate_across_scroll_pages(self, storage):
        # Multi-batch version of the above: the name-match definitions (the
        # high-quality tier find_references discards) all land in the FIRST
        # scroll page, the references in a LATER page. High-quality early
        # termination must NOT stop the scroll after page one, or the
        # references are never fetched and find_references reports none. The
        # single-page test above cannot catch this since it never advances
        # past page one. limit (30) exceeds page one's matches so scan_cap is
        # not what stops it — only the early-termination gate matters here.
        defs = [_point(100 + i, name="foo") for i in range(12)]  # 12 name matches (3.0)
        refs = [_point(i, content="uses foo here") for i in range(1, 4)]  # 3 usages (1.0)
        storage._client_mock.scroll = AsyncMock(
            side_effect=[(defs, "page2"), (refs, None)]
        )

        results = await storage.exact_match_search(
            "col", "foo", mode="chunk", limit=30, rank=False
        )

        # Both scroll pages must be fetched, so the references survive.
        assert storage._client_mock.scroll.call_count == 2
        assert len([r for r in results if r.score == 1.0]) == 3
