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
    async def test_creates_three_indexes_once_per_collection(self, storage):
        storage._client_mock.scroll = AsyncMock(return_value=([], None))

        await storage.exact_match_search("col", "foo")
        await storage.exact_match_search("col", "bar")

        created = {
            c.kwargs["field_name"]
            for c in storage._client_mock.create_payload_index.call_args_list
        }
        assert created == {"content", "name", "summary"}
        assert storage._client_mock.create_payload_index.call_count == 3

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

        # Second search: no creation retry, still exhaustive.
        await storage.exact_match_search("col", "bar")
        assert storage._client_mock.create_payload_index.call_count == 3
        assert not any(getattr(f, "should", None) for f in _scroll_filters(storage._client_mock))

    @pytest.mark.asyncio
    async def test_create_collection_ensures_indexes(self, storage):
        storage._core.create_collection = AsyncMock()

        await storage.create_collection("newcol")

        assert storage._client_mock.create_payload_index.call_count == 3

    @pytest.mark.asyncio
    async def test_delete_collection_invalidates_index_cache(self, storage):
        """A same-process delete + recreate must re-create the text indexes
        (force_reindex path); Qdrant drops them with the collection."""
        storage._core.create_collection = AsyncMock()
        storage._core.delete_collection = AsyncMock()

        await storage.create_collection("col")
        await storage.delete_collection("col")
        await storage.create_collection("col")

        assert storage._client_mock.create_payload_index.call_count == 6
