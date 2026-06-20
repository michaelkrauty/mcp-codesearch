"""Tests for query planning and execution."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_codesearch.search.preprocess import ParsedQuery, preprocess_query
from mcp_codesearch.search.query import (
    QueryType,
    _apply_path_boost,
    _filter_by_parsed_query,
    _merge_results,
    _normalize_scores,
    _plan_query,
    format_results,
    search_codebase,
)
from mcp_codesearch.storage.qdrant import SearchResult


class TestQueryPlanner:
    """Tests for query type classification and planning."""

    def test_name_only_function(self):
        _, parsed = preprocess_query("fn:handleRequest")
        plan = _plan_query("fn:handleRequest", parsed)

        assert plan.query_type == QueryType.NAME_ONLY
        assert plan.name_filter == "handleRequest"
        assert plan.skip_embedding is True

    def test_name_only_class(self):
        _, parsed = preprocess_query("class:UserService")
        plan = _plan_query("class:UserService", parsed)

        assert plan.query_type == QueryType.NAME_ONLY
        assert plan.name_filter == "UserService"
        assert plan.skip_embedding is True

    def test_name_semantic(self):
        _, parsed = preprocess_query("fn:validate input data")
        plan = _plan_query("fn:validate input data", parsed)

        assert plan.query_type == QueryType.NAME_SEMANTIC
        assert plan.name_filter == "validate"
        assert plan.skip_embedding is False

    def test_semantic_query(self):
        _, parsed = preprocess_query("error handling code")
        plan = _plan_query("error handling code", parsed)

        assert plan.query_type == QueryType.SEMANTIC
        assert plan.name_filter is None
        assert plan.skip_embedding is False

    def test_exact_phrase(self):
        query = '"exact phrase match"'
        _, parsed = preprocess_query(query)
        plan = _plan_query(query, parsed)

        assert plan.query_type == QueryType.EXACT_PHRASE
        assert plan.skip_embedding is True
        assert "exact phrase match" in plan.search_text


class TestPathBoosting:
    """Tests for path-based score boosting."""

    def _make_result(self, path: str, score: float) -> SearchResult:
        return SearchResult(
            path=path,
            score=score,
            point_type="chunk",
            language="python",
            summary=None,
            line_count=None,
            chunk_type="function",
            name="test",
            start_line=1,
            end_line=10,
            content="test content",
        )

    def test_src_boost(self):
        results = [self._make_result("src/main.py", 0.5)]
        boosted = _apply_path_boost(results)

        # src/ should boost score
        assert boosted[0].score > 0.5

    def test_test_demotion(self):
        results = [self._make_result("tests/test_main.py", 0.5)]
        boosted = _apply_path_boost(results)

        # tests/ should demote score
        assert boosted[0].score < 0.5

    def test_vendor_demotion(self):
        results = [self._make_result("vendor/lib/code.py", 0.5)]
        boosted = _apply_path_boost(results)

        # vendor/ should demote score significantly
        assert boosted[0].score < 0.5  # Lower than original
        assert boosted[0].score < 0.35  # Significantly demoted

    def test_relative_ordering(self):
        results = [
            self._make_result("tests/test.py", 0.6),
            self._make_result("src/main.py", 0.5),
        ]
        boosted = _apply_path_boost(results)

        # src/ file should rank higher despite lower base score
        src_result = next(r for r in boosted if "src/" in r.path)
        test_result = next(r for r in boosted if "tests/" in r.path)
        assert src_result.score > test_result.score

    def test_exact_match_tiers_survive_boost(self):
        """Exact-match score tiers (name=3.0/summary=2.0/content=1.0) must not
        be clamped into a tie; sort must order by tier, not arrival order."""
        # Reverse arrival order: content tier first, name tier last
        results = [
            self._make_result("lib/content_hit.py", 1.0),
            self._make_result("lib/summary_hit.py", 2.0),
            self._make_result("lib/name_hit.py", 3.0),
        ]
        boosted = _apply_path_boost(results)

        assert [r.path for r in boosted] == [
            "lib/name_hit.py",
            "lib/summary_hit.py",
            "lib/content_hit.py",
        ]
        # Scores above 1.0 must be preserved (normalization happens later)
        assert boosted[0].score > boosted[1].score > boosted[2].score
        assert boosted[0].score > 1.0

    def test_demoted_exact_name_match_outranks_boosted_content_match(self):
        """A name-tier exact match in a demoted test path must still outrank
        a content-tier match in a boosted src path."""
        results = [
            self._make_result("src/incidental.py", 1.0),  # content tier, boosted
            self._make_result("tests/test_target.py", 3.0),  # name tier, demoted
        ]
        boosted = _apply_path_boost(results)

        assert boosted[0].path == "tests/test_target.py"
        assert boosted[0].score > boosted[1].score


class TestScoreNormalization:
    """Tests for score normalization."""

    def _make_result(self, score: float) -> SearchResult:
        return SearchResult(
            path="test.py",
            score=score,
            point_type="chunk",
            language="python",
            summary=None,
            line_count=None,
            chunk_type="function",
            name="test",
            start_line=1,
            end_line=10,
            content="test content",
        )

    def test_normalization_range(self):
        results = [
            self._make_result(2.5),
            self._make_result(1.5),
            self._make_result(0.5),
        ]
        normalized = _normalize_scores(results)

        # All scores should be in 0-1 range
        for r in normalized:
            assert 0.0 <= r.score <= 1.0

        # Top score should be 1.0
        assert normalized[0].score == 1.0

    def test_empty_results(self):
        normalized = _normalize_scores([])
        assert normalized == []

    def test_zero_scores(self):
        results = [self._make_result(0.0)]
        normalized = _normalize_scores(results)
        assert normalized[0].score == 0.0


class TestResultFiltering:
    """Tests for query-based result filtering."""

    def _make_result(self, path: str, name: str | None = None) -> SearchResult:
        return SearchResult(
            path=path,
            score=0.5,
            point_type="chunk",
            language="python",
            summary=None,
            line_count=None,
            chunk_type="function",
            name=name,
            start_line=1,
            end_line=10,
            content="test content",
        )

    def test_path_prefix_filter(self):
        parsed = ParsedQuery(text="", path_prefix="src/", exclude_paths=[])
        results = [
            self._make_result("src/main.py"),
            self._make_result("tests/test.py"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        assert len(filtered) == 1
        assert filtered[0].path == "src/main.py"

    def test_path_prefix_matches_mid_path_component(self):
        """path:embeddings should match directories containing 'embeddings'"""
        parsed = ParsedQuery(text="", path_prefix="embeddings", exclude_paths=[])
        results = [
            self._make_result("src/mcp_codesearch/embeddings/client.py"),
            self._make_result("src/mcp_codesearch/embeddings/sparse.py"),
            self._make_result("src/mcp_codesearch/server.py"),
            self._make_result("tests/test_embeddings.py"),  # filename match, not dir
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        # Only matches directory components, not filenames
        assert len(filtered) == 2
        paths = [r.path for r in filtered]
        assert "src/mcp_codesearch/embeddings/client.py" in paths
        assert "src/mcp_codesearch/embeddings/sparse.py" in paths
        # test_embeddings.py should NOT match (embeddings is in filename, not dir)
        assert "tests/test_embeddings.py" not in paths

    def test_path_prefix_matches_partial_component(self):
        """path:mcp should match src/mcp_codesearch/..."""
        parsed = ParsedQuery(text="", path_prefix="mcp", exclude_paths=[])
        results = [
            self._make_result("src/mcp_codesearch/server.py"),
            self._make_result("src/other/module.py"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        assert len(filtered) == 1
        assert filtered[0].path == "src/mcp_codesearch/server.py"

    def test_path_prefix_still_matches_from_start(self):
        """path:src should still work as prefix from start"""
        parsed = ParsedQuery(text="", path_prefix="src", exclude_paths=[])
        results = [
            self._make_result("src/main.py"),
            self._make_result("tests/test.py"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        assert len(filtered) == 1
        assert filtered[0].path == "src/main.py"

    def test_path_exclude_filter(self):
        parsed = ParsedQuery(text="", exclude_paths=["test"])
        results = [
            self._make_result("src/main.py"),
            self._make_result("tests/test.py"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        assert len(filtered) == 1
        assert filtered[0].path == "src/main.py"

    def test_function_name_filter(self):
        parsed = ParsedQuery(text="", function_name="handle")
        results = [
            self._make_result("a.py", "handleRequest"),
            self._make_result("b.py", "process"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        assert len(filtered) == 1
        assert filtered[0].name == "handleRequest"

    def test_file_pattern_exact_filename(self):
        """file:db.py keeps src/db.py and drops src/db_pool.py."""
        parsed = ParsedQuery(text="", file_pattern="db.py")
        results = [
            self._make_result("src/db.py"),
            self._make_result("src/db_pool.py"),
            self._make_result("src/other/db.py"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        assert [r.path for r in filtered] == ["src/db.py", "src/other/db.py"]

    def test_file_pattern_glob(self):
        """file:*.sql matches by glob against the filename component."""
        parsed = ParsedQuery(text="", file_pattern="*.sql")
        results = [
            self._make_result("migrations/001_init.sql"),
            self._make_result("src/db.py"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        assert len(filtered) == 1
        assert filtered[0].path == "migrations/001_init.sql"

    def test_file_pattern_case_insensitive(self):
        parsed = ParsedQuery(text="", file_pattern="DB.PY")
        results = [self._make_result("src/db.py")]
        filtered = _filter_by_parsed_query(results, parsed)

        assert len(filtered) == 1

    def test_file_pattern_combined_with_path_prefix(self):
        parsed = ParsedQuery(text="", file_pattern="*.py", path_prefix="src")
        results = [
            self._make_result("src/db.py"),
            self._make_result("src/schema.sql"),
            self._make_result("tests/test_db.py"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        assert len(filtered) == 1
        assert filtered[0].path == "src/db.py"


class TestResultMerging:
    """Tests for result merging."""

    def _make_result(self, path: str, line: int, score: float) -> SearchResult:
        return SearchResult(
            path=path,
            score=score,
            point_type="chunk",
            language="python",
            summary=None,
            line_count=None,
            chunk_type="function",
            name="test",
            start_line=line,
            end_line=line + 10,
            content="test content",
        )

    def test_no_duplicates(self):
        primary = [
            self._make_result("a.py", 1, 1.0),
            self._make_result("b.py", 1, 0.9),
        ]
        secondary = [
            self._make_result("a.py", 1, 0.8),  # Duplicate
            self._make_result("c.py", 1, 0.7),
        ]
        merged = _merge_results(primary, secondary)

        assert len(merged) == 3
        paths = [r.path for r in merged]
        assert paths.count("a.py") == 1

    def test_primary_priority(self):
        primary = [self._make_result("a.py", 1, 1.0)]
        secondary = [self._make_result("a.py", 1, 0.5)]

        merged = _merge_results(primary, secondary)

        # Primary's score should be preserved
        assert merged[0].score == 1.0

    def test_order_preserved(self):
        primary = [
            self._make_result("a.py", 1, 1.0),
            self._make_result("b.py", 1, 0.9),
        ]
        secondary = [
            self._make_result("c.py", 1, 0.8),
        ]
        merged = _merge_results(primary, secondary)

        # Order: primary first, then secondary
        assert merged[0].path == "a.py"
        assert merged[1].path == "b.py"
        assert merged[2].path == "c.py"


class TestFormatResults:
    """Tests for format_results function."""

    def _make_file_result(self, path: str, score: float = 0.9) -> SearchResult:
        return SearchResult(
            path=path,
            score=score,
            point_type="file",
            language="python",
            summary="Test file summary",
            line_count=100,
            chunk_type=None,
            name=None,
            start_line=None,
            end_line=None,
            content=None,
        )

    def _make_chunk_result(
        self, path: str, name: str, score: float = 0.8
    ) -> SearchResult:
        return SearchResult(
            path=path,
            score=score,
            point_type="chunk",
            language="python",
            summary=None,
            line_count=None,
            chunk_type="function",
            name=name,
            start_line=10,
            end_line=25,
            content="def test():\n    pass",
        )

    def test_empty_results_text(self):
        """Empty results in text format."""
        result = format_results([], output_format="text")
        assert "No results found" in result

    def test_empty_results_json(self):
        """Empty results in JSON format."""
        result = format_results([], output_format="json")
        assert result == "[]"

    def test_text_format_file(self):
        """Text format for file result."""
        results = [self._make_file_result("src/main.py")]

        output = format_results(results, output_format="text")

        assert "src/main.py" in output
        assert "python" in output
        assert "100 lines" in output

    def test_text_format_chunk(self):
        """Text format for chunk result."""
        results = [self._make_chunk_result("src/main.py", "handleRequest")]

        output = format_results(results, output_format="text")

        assert "src/main.py:10-25" in output
        assert "handleRequest" in output
        assert "function" in output

    def test_json_format(self):
        """JSON format output."""
        results = [self._make_file_result("src/main.py", 0.9)]

        output = format_results(results, output_format="json")
        parsed = json.loads(output)

        assert len(parsed) == 1
        assert parsed[0]["path"] == "src/main.py"
        assert parsed[0]["score"] == 0.9

    def test_json_format_all_fields(self):
        """JSON format includes all fields."""
        results = [self._make_chunk_result("src/main.py", "test")]

        output = format_results(results, output_format="json")
        parsed = json.loads(output)

        assert "path" in parsed[0]
        assert "score" in parsed[0]
        assert "type" in parsed[0]
        assert "language" in parsed[0]
        assert "chunk_type" in parsed[0]
        assert "name" in parsed[0]
        assert "start_line" in parsed[0]
        assert "end_line" in parsed[0]
        assert "content" in parsed[0]

    def test_markdown_format_file(self):
        """Markdown format for file result."""
        results = [self._make_file_result("src/main.py")]

        output = format_results(results, output_format="markdown")

        assert "`src/main.py`" in output
        assert "###" in output  # Header
        assert "python" in output

    def test_markdown_format_chunk(self):
        """Markdown format for chunk result."""
        results = [self._make_chunk_result("src/main.py", "handleRequest")]

        output = format_results(results, output_format="markdown")

        assert "```python" in output  # Code block
        assert "handleRequest" in output

    def test_content_truncation(self):
        """Content is truncated in output."""
        result = SearchResult(
            path="test.py",
            score=0.5,
            point_type="chunk",
            language="python",
            summary=None,
            line_count=None,
            chunk_type="function",
            name="test",
            start_line=1,
            end_line=10,
            content="x" * 1000,
        )

        output = format_results([result], content_length=100)

        # Should be much shorter than 1000 chars
        assert len(output) < 500

    def test_summary_truncation(self):
        """Summary is truncated in output."""
        result = SearchResult(
            path="test.py",
            score=0.5,
            point_type="file",
            language="python",
            summary="x" * 500,
            line_count=100,
            chunk_type=None,
            name=None,
            start_line=None,
            end_line=None,
            content=None,
        )

        output = format_results([result], summary_length=50)

        # Should be much shorter than 500 chars
        assert len(output) < 300


class TestSearchCodebase:
    """Tests for search_codebase function execution paths."""

    @pytest.mark.asyncio
    async def test_exact_phrase_query_path(self):
        """EXACT_PHRASE query type uses exact_match_search (line 317)."""
        from unittest.mock import AsyncMock, MagicMock

        from mcp_codesearch.search.query import search_codebase

        mock_storage = MagicMock()
        mock_storage.exact_match_search = AsyncMock(return_value=[
            SearchResult(
                path="src/main.py",
                score=1.0,
                point_type="chunk",
                language="python",
                summary=None,
                line_count=None,
                chunk_type="function",
                name="test",
                start_line=1,
                end_line=10,
                content="exact phrase here",
            )
        ])

        mock_embedder = MagicMock()

        # Mock global vocabulary
        mock_global_vocab = MagicMock()
        mock_global_vocab.vectorize_query.return_value = MagicMock(indices=[0, 1], values=[1.0, 1.0])

        # Quoted query triggers EXACT_PHRASE path
        results = await search_codebase(
            query='"exact phrase"',
            codebase_path="/test",
            storage=mock_storage,
            embedder=mock_embedder,
            global_vocab=mock_global_vocab,
            mode="both",
            language=None,
            limit=10,
        )

        # Should have called exact_match_search, not hybrid_search
        mock_storage.exact_match_search.assert_called_once()
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_name_semantic_query_path(self):
        """NAME_SEMANTIC query type does name lookup + semantic search (lines 328-350)."""
        from unittest.mock import AsyncMock, MagicMock

        from mcp_codesearch.search.query import search_codebase

        mock_storage = MagicMock()
        # Name results from exact_match_search
        mock_storage.exact_match_search = AsyncMock(return_value=[
            SearchResult(
                path="src/main.py",
                score=1.0,
                point_type="chunk",
                language="python",
                summary=None,
                line_count=None,
                chunk_type="function",
                name="validate",
                start_line=1,
                end_line=10,
                content="def validate(): pass",
            )
        ])
        # Semantic results from hybrid_search
        mock_storage.hybrid_search = AsyncMock(return_value=[
            SearchResult(
                path="src/other.py",
                score=0.8,
                point_type="chunk",
                language="python",
                summary=None,
                line_count=None,
                chunk_type="function",
                name="check_input",
                start_line=1,
                end_line=10,
                content="def check_input(): pass",
            )
        ])

        mock_embedder = MagicMock()
        mock_embedder.embed_single_cached = AsyncMock(return_value=[0.1] * 384)

        # Mock global vocabulary
        mock_global_vocab = MagicMock()
        mock_global_vocab.vectorize_query.return_value = MagicMock(indices=[0, 1, 2], values=[1.0, 1.0, 1.0])

        # fn:validate + additional text triggers NAME_SEMANTIC path
        results = await search_codebase(
            query="fn:validate input data",
            codebase_path="/test",
            storage=mock_storage,
            embedder=mock_embedder,
            global_vocab=mock_global_vocab,
            mode="both",
            language=None,
            limit=10,
        )

        # Should have called both exact_match_search and hybrid_search
        mock_storage.exact_match_search.assert_called_once()
        mock_storage.hybrid_search.assert_called_once()
        # Results should include both (merged)
        assert len(results) >= 1


    @pytest.mark.asyncio
    async def test_name_lookup_without_postfilter_ranks_exact(self):
        """A bare cls:/fn: lookup ranks exact-match results (rank=True) so the
        definition is not lost to scroll-order truncation."""
        mock_storage = MagicMock()
        mock_storage.exact_match_search = AsyncMock(return_value=[])
        mock_embedder = MagicMock()
        mock_global_vocab = MagicMock()

        await search_codebase(
            query="cls:Foo", codebase_path="/test", storage=mock_storage,
            embedder=mock_embedder, global_vocab=mock_global_vocab, limit=10,
        )

        assert mock_storage.exact_match_search.call_args.kwargs["rank"] is True

    @pytest.mark.asyncio
    async def test_name_lookup_with_path_filter_disables_ranking(self):
        """With an un-pushed post-filter (path:) ranking is disabled so the
        score sort cannot starve the lower-scored matches the filter wants."""
        mock_storage = MagicMock()
        mock_storage.exact_match_search = AsyncMock(return_value=[])
        mock_embedder = MagicMock()
        mock_global_vocab = MagicMock()

        await search_codebase(
            query="cls:Foo path:src", codebase_path="/test", storage=mock_storage,
            embedder=mock_embedder, global_vocab=mock_global_vocab, limit=10,
        )

        assert mock_storage.exact_match_search.call_args.kwargs["rank"] is False

    @pytest.mark.asyncio
    async def test_scope_filter_disables_ranking(self):
        """scope: is post-filtered too, so it disables exact-match ranking."""
        mock_storage = MagicMock()
        mock_storage.exact_match_search = AsyncMock(return_value=[])
        mock_embedder = MagicMock()
        mock_global_vocab = MagicMock()

        await search_codebase(
            query="cls:Foo scope:test", codebase_path="/test", storage=mock_storage,
            embedder=mock_embedder, global_vocab=mock_global_vocab, limit=10,
        )

        assert mock_storage.exact_match_search.call_args.kwargs["rank"] is False


class TestFilterByParsedQueryEdgeCases:
    """Tests for edge cases in _filter_by_parsed_query."""

    def _make_result(self, path, name=None, chunk_type=None):
        return SearchResult(
            path=path,
            score=0.5,
            point_type="chunk",
            language="python",
            summary=None,
            line_count=10,
            chunk_type=chunk_type,
            name=name,
            start_line=1,
            end_line=10,
            content="test content",
        )

    def test_path_prefix_substring_in_component(self):
        """path:code should match mcp_codesearch (line 173 - substring match)."""
        parsed = ParsedQuery(text="", path_prefix="code", exclude_paths=[])
        results = [
            self._make_result("src/mcp_codesearch/server.py"),
            self._make_result("src/other/module.py"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        # "code" is substring of "mcp_codesearch" (line 173)
        assert len(filtered) == 1
        assert filtered[0].path == "src/mcp_codesearch/server.py"

    def test_class_name_filter(self):
        """class:Foo should filter by class name (lines 187-188)."""
        parsed = ParsedQuery(text="", class_name="Service", exclude_paths=[])
        results = [
            self._make_result("src/main.py", name="UserService"),
            self._make_result("src/main.py", name="handleRequest"),
            self._make_result("src/main.py", name=None),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        assert len(filtered) == 1
        assert filtered[0].name == "UserService"

    def test_scope_test_filter(self):
        """scope:test filters to test files/names (lines 192-198)."""
        parsed = ParsedQuery(text="", scope="test", exclude_paths=[])
        results = [
            self._make_result("tests/test_main.py", name="test_func"),
            self._make_result("src/main.py", name="main"),
            self._make_result("src/helper.py", name="test_helper"),  # name has test
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        # Matches files with 'test' in path OR names with 'test'
        assert len(filtered) == 2
        paths = [r.path for r in filtered]
        assert "tests/test_main.py" in paths
        assert "src/helper.py" in paths  # name has 'test'

    def test_scope_impl_filter(self):
        """scope:impl excludes test files (lines 199-205)."""
        parsed = ParsedQuery(text="", scope="impl", exclude_paths=[])
        results = [
            self._make_result("tests/test_main.py", name="test_func"),
            self._make_result("src/main.py", name="main"),
            self._make_result("src/helper.py", name="test_helper"),  # name has test
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        # Only src/main.py passes (no 'test' in path or name)
        assert len(filtered) == 1
        assert filtered[0].path == "src/main.py"

    def test_scope_function_filter(self):
        """scope:function filters to function chunks (lines 206-211)."""
        parsed = ParsedQuery(text="", scope="function", exclude_paths=[])
        results = [
            self._make_result("src/main.py", name="main", chunk_type="function"),
            self._make_result("src/main.py", name="MyClass", chunk_type="class"),
            self._make_result("src/main.py", name="helper", chunk_type="method"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        # Only function and method types
        assert len(filtered) == 2
        names = [r.name for r in filtered]
        assert "main" in names
        assert "helper" in names

    def test_scope_class_filter(self):
        """scope:class filters to class chunks (lines 212-217)."""
        parsed = ParsedQuery(text="", scope="class", exclude_paths=[])
        results = [
            self._make_result("src/main.py", name="main", chunk_type="function"),
            self._make_result("src/main.py", name="MyClass", chunk_type="class"),
            self._make_result("src/main.py", name="MyStruct", chunk_type="struct"),
            self._make_result("src/main.py", name="Overview", chunk_type="class_overview"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        # Only class, struct, class_overview types
        assert len(filtered) == 3
        names = [r.name for r in filtered]
        assert "MyClass" in names
        assert "MyStruct" in names
        assert "Overview" in names

    def test_scope_test_anchored_rejects_substrings(self):
        """scope:test anchors on path/name and does not match substrings such
        as contest/latest/attestation."""
        parsed = ParsedQuery(text="", scope="test", exclude_paths=[])
        results = [
            self._make_result("src/contest.py", name="contest"),
            self._make_result("src/latest.py", name="latest"),
            self._make_result("src/attestation.py", name="attestation"),
            self._make_result("tests/test_main.py", name="test_main"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        # Only the genuine test file survives.
        assert [r.path for r in filtered] == ["tests/test_main.py"]

    def test_scope_impl_keeps_nontest_substrings(self):
        """scope:impl keeps non-test files whose name merely contains 'test' as
        a substring, and drops genuine tests."""
        parsed = ParsedQuery(text="", scope="impl", exclude_paths=[])
        results = [
            self._make_result("src/contest.py", name="contest"),
            self._make_result("src/latest.py", name="latest"),
            self._make_result("src/attestation.py", name="attestation"),
            self._make_result("tests/test_main.py", name="test_main"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        assert [r.path for r in filtered] == [
            "src/contest.py",
            "src/latest.py",
            "src/attestation.py",
        ]

    def test_scope_class_includes_enum_type_module(self):
        """scope:class includes enum, type-alias and module chunk types, which
        the indexer also emits, alongside class/struct/interface."""
        parsed = ParsedQuery(text="", scope="class", exclude_paths=[])
        results = [
            self._make_result("src/m.py", name="C", chunk_type="class"),
            self._make_result("src/m.py", name="E", chunk_type="enum"),
            self._make_result("src/m.py", name="T", chunk_type="type"),
            self._make_result("src/m.py", name="M", chunk_type="module"),
            self._make_result("src/m.py", name="S", chunk_type="struct"),
            self._make_result("src/m.py", name="O", chunk_type="class_overview"),
            self._make_result("src/m.py", name="f", chunk_type="function"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        kept = {r.name for r in filtered}
        assert kept == {"C", "E", "T", "M", "S", "O"}

    def test_scope_test_recognizes_common_conventions(self):
        """scope:test detects test directory and filename conventions across
        ecosystems (Jest __tests__, JUnit *Test, RSpec/Kotlin *Spec, dotted
        *.test/*.spec, Django tests.py, prefix Test* class names)."""
        parsed = ParsedQuery(text="", scope="test", exclude_paths=[])
        results = [
            self._make_result("src/__tests__/helpers.ts", name="buildUser"),
            self._make_result("app/UserServiceTest.java", name="setUp"),
            self._make_result("app/FooTests.cs", name="Setup"),
            self._make_result("src/OrderSpec.kt", name="describe"),
            self._make_result("pkg/foo.test.ts", name="render"),
            self._make_result("pkg/bar.spec.js", name="mount"),
            self._make_result("app/tests.py", name="run"),
            self._make_result("src/main.py", name="TestUserService"),
            self._make_result("integration_tests/runner.py", name="run"),
            self._make_result("spec/support/factory_bot.rb", name="configure"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        # Every result is recognized as a test.
        assert len(filtered) == len(results)

    def test_scope_impl_keeps_substrings_drops_conventions(self):
        """scope:impl drops the test conventions above but keeps implementation
        files whose path/name merely contain a test substring."""
        parsed = ParsedQuery(text="", scope="impl", exclude_paths=[])
        results = [
            self._make_result("src/__tests__/helpers.ts", name="buildUser"),
            self._make_result("app/UserServiceTest.java", name="setUp"),
            self._make_result("src/contest.py", name="latestEntry"),
            self._make_result("src/service.py", name="manifest"),
        ]
        filtered = _filter_by_parsed_query(results, parsed)

        assert [r.path for r in filtered] == ["src/contest.py", "src/service.py"]
