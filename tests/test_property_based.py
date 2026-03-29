"""Property-based tests using Hypothesis for mcp-codesearch.

These tests verify that parsing and processing functions never crash
regardless of input, providing robustness guarantees.
"""

import string
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from vector_core.utils.hashing import hash_content

from mcp_codesearch.indexer.discovery import _detect_language
from mcp_codesearch.search.preprocess import ParsedQuery, preprocess_query


class TestPreprocessQueryPropertyBased:
    """Property-based tests for query preprocessing."""

    @given(st.text(max_size=1000))
    @settings(max_examples=200)
    def test_preprocess_never_crashes(self, query: str) -> None:
        """preprocess_query should never raise an exception for any input."""
        # This should never crash
        result, parsed = preprocess_query(query)

        # Basic sanity checks
        assert isinstance(result, str)
        assert isinstance(parsed, ParsedQuery)

    @given(st.text(alphabet=string.printable, max_size=500))
    @settings(max_examples=100)
    def test_preprocess_printable_text(self, query: str) -> None:
        """preprocess_query handles all printable characters."""
        result, parsed = preprocess_query(query)
        assert isinstance(result, str)
        assert isinstance(parsed, ParsedQuery)

    @given(
        st.text(max_size=100),
        st.sampled_from(["fn:", "function:", "class:", "cls:", "struct:", "path:", "-path:"]),
    )
    @settings(max_examples=100)
    def test_preprocess_with_prefixes(self, suffix: str, prefix: str) -> None:
        """preprocess_query handles prefix syntax correctly."""
        query = prefix + suffix
        result, parsed = preprocess_query(query)
        assert isinstance(result, str)
        assert isinstance(parsed, ParsedQuery)

    @given(st.text(max_size=200))
    @settings(max_examples=100)
    def test_preprocess_quoted_strings(self, text: str) -> None:
        """preprocess_query handles quoted strings."""
        # Test with various quote styles
        for quote in ['"', "'"]:
            query = f"{quote}{text}{quote}"
            result, parsed = preprocess_query(query)
            assert isinstance(result, str)

    @given(st.lists(st.text(max_size=50), max_size=10))
    @settings(max_examples=50)
    def test_preprocess_multi_term_queries(self, terms: list[str]) -> None:
        """preprocess_query handles multi-term queries."""
        query = " ".join(terms)
        result, parsed = preprocess_query(query)
        assert isinstance(result, str)
        assert isinstance(parsed, ParsedQuery)


class TestHashContentPropertyBased:
    """Property-based tests for content hashing."""

    @given(st.text(max_size=10000))
    @settings(max_examples=100)
    def test_hash_deterministic(self, content: str) -> None:
        """Same content always produces same hash."""
        hash1 = hash_content(content)
        hash2 = hash_content(content)
        assert hash1 == hash2

    @given(st.text(min_size=1, max_size=1000), st.text(min_size=1, max_size=1000))
    @settings(max_examples=100)
    def test_hash_different_content(self, content1: str, content2: str) -> None:
        """Different content produces different hashes (with high probability)."""
        assume(content1 != content2)
        hash1 = hash_content(content1)
        hash2 = hash_content(content2)
        # Different content should produce different hashes
        assert hash1 != hash2

    @given(st.text(max_size=5000))
    @settings(max_examples=100)
    def test_hash_format(self, content: str) -> None:
        """Hash is always 64-character hex string."""
        result = hash_content(content)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestDetectLanguagePropertyBased:
    """Property-based tests for language detection."""

    @given(st.text(alphabet=string.ascii_letters + string.digits + "_-.", max_size=100))
    @settings(max_examples=100)
    def test_detect_language_never_crashes(self, filename: str) -> None:
        """_detect_language should never crash for any filename."""
        path = Path(f"/fake/{filename}")
        result = _detect_language(path)
        # Result should be None or a string
        assert result is None or isinstance(result, str)

    @given(st.sampled_from([
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
        ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
        ".scala", ".cs", ".vue", ".svelte", ".md", ".sql", ".sh",
        ".yaml", ".yml", ".json", ".toml", ".html", ".css", ".scss",
    ]))
    def test_detect_known_extensions(self, ext: str) -> None:
        """Known extensions should return a language."""
        path = Path(f"/fake/file{ext}")
        result = _detect_language(path)
        assert result is not None
        assert isinstance(result, str)


class TestQueryParsingEdgeCases:
    """Test edge cases that might cause parsing issues."""

    @given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=500))
    @settings(max_examples=100)
    def test_unicode_handling(self, query: str) -> None:
        """preprocess_query handles Unicode correctly."""
        result, parsed = preprocess_query(query)
        assert isinstance(result, str)

    @pytest.mark.parametrize("edge_case", [
        "",  # Empty
        " ",  # Space only
        "\n",  # Newline only
        "\t",  # Tab only
        "   \n\t  ",  # Mixed whitespace
        "fn:",  # Prefix only
        "path:",  # Prefix only
        "-path:",  # Negative prefix only
        '"',  # Single quote
        '""',  # Empty quotes
        "a" * 10000,  # Very long query
        "fn:a fn:b fn:c",  # Multiple same prefix
        "path:a path:b -path:c -path:d",  # Complex path filters
        "\\n\\t\\r",  # Escaped characters
        "日本語検索",  # Japanese
        "🔍 search 🔎",  # Emoji
        "SELECT * FROM",  # SQL-like
        "<script>alert('xss')</script>",  # HTML/XSS-like
        "${PATH}",  # Shell variable-like
        "{{template}}",  # Template-like
    ])
    def test_edge_case_queries(self, edge_case: str) -> None:
        """Known edge cases should not crash."""
        result, parsed = preprocess_query(edge_case)
        assert isinstance(result, str)
        assert isinstance(parsed, ParsedQuery)
