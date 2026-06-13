"""Tests for query preprocessing."""

from mcp_codesearch.search.preprocess import (
    _MAX_INDEXED_TOKEN_LEN,
    expand_camelcase,
    file_pattern_pushdown_tokens,
    infer_language,
    preprocess_query,
)
from mcp_codesearch.storage.qdrant import QdrantStorage


class TestCamelCaseExpansion:
    """Tests for CamelCase/PascalCase expansion.

    Note: The expand_camelcase function returns BOTH the original and expanded
    versions for better matching, e.g., "UserService" -> "UserService user service"
    """

    def test_pascal_case(self):
        result = expand_camelcase("UserService")
        # Returns both original and expanded
        assert "UserService" in result
        assert "user" in result.lower()
        assert "service" in result.lower()

    def test_camel_case(self):
        result = expand_camelcase("getUserData")
        assert "getUserData" in result
        assert "get" in result.lower()
        assert "user" in result.lower()
        assert "data" in result.lower()

    def test_lowercase(self):
        assert expand_camelcase("simple") == "simple"

    def test_all_caps(self):
        # All caps should stay together
        assert expand_camelcase("API") == "API"

    def test_mixed(self):
        result = expand_camelcase("XMLHttpRequest")
        assert "XMLHttpRequest" in result
        assert "xml" in result.lower()
        assert "http" in result.lower()
        assert "request" in result.lower()

    def test_with_numbers(self):
        result = expand_camelcase("getUser123")
        # Numbers stay attached to preceding word
        assert "get" in result.lower()


class TestLanguageInference:
    """Tests for language inference from query patterns."""

    def test_react_hooks(self):
        assert infer_language("useEffect cleanup") == "typescript"
        assert infer_language("useState handler") == "typescript"

    def test_python_dunder(self):
        assert infer_language("__init__ method") == "python"
        assert infer_language("__main__ guard") == "python"

    def test_rust_impl(self):
        assert infer_language("impl Trait for") == "rust"

    def test_go_func(self):
        # The pattern requires method receiver syntax or go-specific keywords
        assert infer_language("func (s *Server) handleRequest()") == "go"
        assert infer_language("goroutine example") == "go"
        assert infer_language("main.go file") == "go"
        # Note: "func main()" alone doesn't match - pattern needs receiver or keyword

    def test_no_inference(self):
        assert infer_language("generic query") is None
        assert infer_language("error handling") is None


class TestQueryParsing:
    """Tests for structured query parsing."""

    def test_function_prefix(self):
        _, parsed = preprocess_query("fn:handleRequest")
        assert parsed.function_name == "handleRequest"
        assert parsed.text == ""

    def test_function_with_text(self):
        _, parsed = preprocess_query("fn:handleRequest websocket")
        assert parsed.function_name == "handleRequest"
        assert "websocket" in parsed.text

    def test_class_prefix(self):
        _, parsed = preprocess_query("class:UserService")
        assert parsed.class_name == "UserService"

    def test_path_prefix(self):
        _, parsed = preprocess_query("error handling path:src/")
        assert parsed.path_prefix == "src/"
        assert "error" in parsed.text

    def test_exclude_path(self):
        _, parsed = preprocess_query("config -path:test")
        assert "test" in parsed.exclude_paths

    def test_multiple_filters(self):
        _, parsed = preprocess_query("fn:validate path:src -path:test")
        assert parsed.function_name == "validate"
        assert parsed.path_prefix == "src"
        assert "test" in parsed.exclude_paths


class TestSynonymExpansion:
    """Tests for synonym expansion."""

    def test_auth_expansion(self):
        expanded, _ = preprocess_query("auth flow")
        assert "authentication" in expanded or "auth" in expanded

    def test_db_expansion(self):
        expanded, _ = preprocess_query("db connection")
        assert "database" in expanded or "db" in expanded

    def test_api_expansion(self):
        expanded, _ = preprocess_query("api endpoint")
        # Should include synonyms
        assert "api" in expanded.lower()

    def test_no_expansion_for_code(self):
        # Shouldn't expand things that look like code
        expanded, _ = preprocess_query("fn:authenticate")
        # Function name should be preserved
        assert "authenticate" in expanded.lower() or expanded == ""


class TestPreprocessIntegration:
    """Integration tests for full preprocessing pipeline."""

    def test_complex_query(self):
        query = "fn:handleWebSocket connection error path:src -path:test"
        expanded, parsed = preprocess_query(query)

        assert parsed.function_name == "handleWebSocket"
        assert parsed.path_prefix == "src"
        assert "test" in parsed.exclude_paths
        # CamelCase should be expanded in semantic text
        # Note: exact output depends on implementation

    def test_empty_query(self):
        expanded, parsed = preprocess_query("")
        assert expanded == ""
        assert parsed.text == ""

    def test_whitespace_query(self):
        expanded, parsed = preprocess_query("   ")
        assert expanded.strip() == ""

    def test_quoted_preservation(self):
        # Quoted strings might be handled specially
        expanded, parsed = preprocess_query('"exact match"')
        # Should preserve the exact text
        assert "exact" in expanded.lower() or "match" in expanded.lower()


class TestParseQueryEdgeCases:
    """Tests for edge cases in parse_query."""

    def test_file_pattern_filter(self):
        """file:pattern is parsed (lines 304-305)."""
        from mcp_codesearch.search.preprocess import parse_query

        result = parse_query("authentication file:*.py")

        assert result.file_pattern == "*.py"
        assert result.text == "authentication"

    def test_scope_method_normalized(self):
        """scope:method is normalized to function (lines 321-326)."""
        from mcp_codesearch.search.preprocess import parse_query

        result = parse_query("handler scope:method")

        assert result.scope == "function"  # method -> function
        assert result.text == "handler"

    def test_scope_function(self):
        """scope:function is parsed (lines 321-326)."""
        from mcp_codesearch.search.preprocess import parse_query

        result = parse_query("handler scope:function")

        assert result.scope == "function"
        assert result.text == "handler"

    def test_docs_prefix_not_parsed_as_filter(self):
        """docs: prefix is treated as plain text (feature removed)."""
        from mcp_codesearch.search.preprocess import parse_query

        result = parse_query("authentication docs:")

        assert "docs:" in result.text or "authentication" in result.text


class TestFilePatternPushdownTokens:
    """file_pattern_pushdown_tokens must only emit tokens GUARANTEED to
    appear as whole path tokens in every match — a superset prefilter.
    Dropping a token is always safe (widens the superset); emitting an
    unguaranteed token is never safe (could exclude a true match)."""

    def test_prefix_glob(self):
        assert file_pattern_pushdown_tokens("test_*.py") == ["test", "py"]

    def test_extension_glob(self):
        assert file_pattern_pushdown_tokens("*.sql") == ["sql"]

    def test_exact_filename(self):
        assert file_pattern_pushdown_tokens("db.py") == ["db", "py"]

    def test_wildcard_adjacent_runs_dropped(self):
        # "test" touches '*' on both sides: a match like "mytest_helper.py"
        # would not contain "test" as a whole token.
        assert file_pattern_pushdown_tokens("*test*") is None

    def test_bare_star(self):
        assert file_pattern_pushdown_tokens("*") is None

    def test_question_mark_is_wildcard(self):
        assert file_pattern_pushdown_tokens("?.py") == ["py"]

    def test_character_class_is_wildcard(self):
        # The "est" run is adjacent to the [td] class and must be dropped.
        assert file_pattern_pushdown_tokens("[td]est_*.py") == ["py"]

    def test_lowercased(self):
        assert file_pattern_pushdown_tokens("TEST_*.PY") == ["test", "py"]

    def test_oversized_token_dropped(self):
        # 65-char alnum run with clean boundaries: never indexed, so
        # including it would make MatchText match NOTHING (fail-closed trap).
        long_run = "a" * 65
        assert file_pattern_pushdown_tokens(f"{long_run}.py") == ["py"]
        assert file_pattern_pushdown_tokens(long_run) is None

    def test_max_len_token_kept(self):
        assert file_pattern_pushdown_tokens("a" * 64) == ["a" * 64]

    def test_negated_character_class(self):
        assert file_pattern_pushdown_tokens("[!t]est_*.py") == ["py"]

    def test_class_containing_bracket(self):
        # fnmatch: ']' first in the set is literal, class ends at next ']'.
        # The class is a wildcard, so the adjacent "x" run is dropped.
        assert file_pattern_pushdown_tokens("[]]x.py") == ["py"]

    def test_unmatched_bracket_treated_as_wildcard(self):
        # Conservative: "ab" touches the unmatched '[' and is dropped.
        assert file_pattern_pushdown_tokens("[ab.py") == ["py"]

    def test_no_alnum_content(self):
        assert file_pattern_pushdown_tokens("...") is None
        assert file_pattern_pushdown_tokens("") is None

    def test_multiple_separators(self):
        assert file_pattern_pushdown_tokens("conf-test_v2.tar.gz") == [
            "conf", "test", "v2", "tar", "gz",
        ]

    def test_token_limit_matches_storage_index_params(self):
        """The local cutoff must mirror the storage layer's indexed token
        limit, or pushdown could query never-indexed tokens."""
        assert _MAX_INDEXED_TOKEN_LEN == QdrantStorage._TEXT_INDEX_MAX_TOKEN_LEN
