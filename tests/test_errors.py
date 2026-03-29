"""Tests for error collection and reporting."""

from pathlib import Path

from vector_core.errors import (
    ErrorCategory,
    ErrorCollector,
    ErrorSeverity,
    IndexingError,
)


class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""

    def test_severity_values(self):
        """Severity enum has expected values."""
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.ERROR.value == "error"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_category_values(self):
        """Category enum has expected values."""
        assert ErrorCategory.FILE_ACCESS.value == "file_access"
        assert ErrorCategory.PARSE_ERROR.value == "parse_error"
        assert ErrorCategory.ENCODING.value == "encoding"
        assert ErrorCategory.EMBEDDING.value == "embedding"
        assert ErrorCategory.STORAGE.value == "storage"
        assert ErrorCategory.UNKNOWN.value == "unknown"


class TestIndexingError:
    """Tests for IndexingError dataclass."""

    def test_basic_creation(self):
        """Create IndexingError with required fields."""
        error = IndexingError(
            path="src/main.py",
            message="Failed to parse",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.PARSE_ERROR,
        )

        assert error.path == "src/main.py"
        assert error.message == "Failed to parse"
        assert error.severity == ErrorSeverity.ERROR
        assert error.category == ErrorCategory.PARSE_ERROR
        # New model uses separate fields for exception details
        assert error.exception_type is None
        assert error.exception_message is None
        assert error.traceback_str is None

    def test_with_exception(self):
        """Create IndexingError with exception details."""
        error = IndexingError(
            path="src/main.py",
            message="Failed",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.STORAGE,
            exception_type="ConnectionError",
            exception_message="timeout",
        )

        assert error.exception_type == "ConnectionError"
        assert error.exception_message == "timeout"


class TestErrorCollectorAdd:
    """Tests for ErrorCollector.add method."""

    def test_add_error(self):
        """Add error to collector."""
        collector = ErrorCollector()

        collector.add(
            path="test.py",
            message="Test error",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.UNKNOWN,
        )

        assert len(collector.errors) == 1
        assert collector.errors[0].path == "test.py"
        assert collector.errors[0].message == "Test error"

    def test_add_with_path_object(self):
        """Add error with Path object."""
        collector = ErrorCollector()

        collector.add(
            path=Path("/some/path/file.py"),
            message="Error",
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.FILE_ACCESS,
        )

        assert collector.errors[0].path == "/some/path/file.py"

    def test_add_with_exception(self):
        """Add error with exception object."""
        collector = ErrorCollector()
        exc = ValueError("Something went wrong")

        collector.add(
            path="file.py",
            message="Error occurred",
            exception=exc,
        )

        # ErrorCollector.add extracts exception details into separate fields
        assert collector.errors[0].exception_type == "ValueError"
        assert collector.errors[0].exception_message == "Something went wrong"

    def test_max_errors_limit(self):
        """Respects max errors limit."""
        # Create collector with max_errors set
        collector = ErrorCollector(max_errors=5)

        for i in range(10):
            collector.add(path=f"file{i}.py", message=f"Error {i}")

        assert len(collector.errors) == 5
        assert collector.truncated_count == 5  # 5 errors were truncated


class TestErrorCollectorConvenienceMethods:
    """Tests for convenience error methods."""

    def test_add_file_access_error(self):
        """File access error helper."""
        collector = ErrorCollector()
        collector.add_file_access_error("test.py", ValueError("denied"))

        error = collector.errors[0]
        assert error.severity == ErrorSeverity.ERROR
        assert error.category == ErrorCategory.FILE_ACCESS
        assert error.exception_type == "ValueError"
        assert "denied" in (error.exception_message or "")

    def test_add_encoding_error(self):
        """Encoding error helper."""
        collector = ErrorCollector()
        collector.add_encoding_error("test.py")

        error = collector.errors[0]
        assert error.severity == ErrorSeverity.WARNING
        assert error.category == ErrorCategory.ENCODING

    def test_add_parse_error(self):
        """Parse error helper."""
        collector = ErrorCollector()
        collector.add_parse_error("test.py")

        error = collector.errors[0]
        assert error.severity == ErrorSeverity.WARNING
        assert error.category == ErrorCategory.PARSE_ERROR

    def test_add_embedding_error(self):
        """Embedding error helper."""
        collector = ErrorCollector()
        collector.add_embedding_error("test.py")

        error = collector.errors[0]
        assert error.severity == ErrorSeverity.ERROR
        assert error.category == ErrorCategory.EMBEDDING


class TestErrorCollectorProperties:
    """Tests for ErrorCollector properties."""

    def test_has_errors_empty(self):
        """has_errors False when empty."""
        collector = ErrorCollector()
        assert collector.has_errors is False

    def test_has_errors_with_errors(self):
        """has_errors True when errors present."""
        collector = ErrorCollector()
        collector.add(path="test.py", message="Error")
        assert collector.has_errors is True

    def test_error_count(self):
        """error_count returns total count."""
        collector = ErrorCollector()
        collector.add(path="a.py", message="Error 1")
        collector.add(path="b.py", message="Error 2")
        collector.add(path="c.py", message="Error 3")

        assert collector.error_count == 3

    def test_warning_count(self):
        """warning_count returns only warnings."""
        collector = ErrorCollector()
        collector.add(path="a.py", message="E1", severity=ErrorSeverity.WARNING)
        collector.add(path="b.py", message="E2", severity=ErrorSeverity.ERROR)
        collector.add(path="c.py", message="E3", severity=ErrorSeverity.WARNING)

        assert collector.warning_count == 2

    def test_critical_count(self):
        """critical_count returns only critical errors."""
        collector = ErrorCollector()
        collector.add(path="a.py", message="E1", severity=ErrorSeverity.CRITICAL)
        collector.add(path="b.py", message="E2", severity=ErrorSeverity.ERROR)
        collector.add(path="c.py", message="E3", severity=ErrorSeverity.CRITICAL)

        assert collector.critical_count == 2


class TestErrorCollectorByCategory:
    """Tests for by_category grouping."""

    def test_groups_by_category(self):
        """Groups errors by category."""
        collector = ErrorCollector()
        collector.add(path="a.py", message="E1", category=ErrorCategory.PARSE_ERROR)
        collector.add(path="b.py", message="E2", category=ErrorCategory.ENCODING)
        collector.add(path="c.py", message="E3", category=ErrorCategory.PARSE_ERROR)

        by_cat = collector.by_category()

        assert ErrorCategory.PARSE_ERROR in by_cat
        assert len(by_cat[ErrorCategory.PARSE_ERROR]) == 2
        assert ErrorCategory.ENCODING in by_cat
        assert len(by_cat[ErrorCategory.ENCODING]) == 1

    def test_empty_collector(self):
        """Empty collector returns empty dict."""
        collector = ErrorCollector()
        assert collector.by_category() == {}


class TestErrorCollectorFormatSummary:
    """Tests for format_summary method."""

    def test_empty_summary(self):
        """Empty collector returns empty string."""
        collector = ErrorCollector()
        assert collector.format_summary() == ""

    def test_basic_summary(self):
        """Summary contains error count and categories."""
        collector = ErrorCollector()
        collector.add(path="a.py", message="Error 1", category=ErrorCategory.PARSE_ERROR)
        collector.add(path="b.py", message="Error 2", category=ErrorCategory.PARSE_ERROR)

        summary = collector.format_summary()

        assert "2 issues" in summary
        assert "parse_error" in summary
        assert "a.py" in summary

    def test_summary_limits_per_category(self):
        """Summary shows first 3 per category."""
        collector = ErrorCollector()
        for i in range(5):
            collector.add(
                path=f"file{i}.py",
                message=f"Error {i}",
                category=ErrorCategory.ENCODING,
            )

        summary = collector.format_summary()

        assert "file0.py" in summary
        assert "file2.py" in summary
        assert "and 2 more" in summary

    def test_summary_max_errors_note(self):
        """Summary notes when max errors reached."""
        collector = ErrorCollector(max_errors=3)

        for i in range(5):
            collector.add(path=f"file{i}.py", message=f"Error {i}")

        summary = collector.format_summary()

        # Format: "(Showing 3 of 5 errors; 2 omitted)"
        assert "Showing 3 of 5 errors" in summary
        assert "2 omitted" in summary


class TestErrorCollectorClear:
    """Tests for clear method."""

    def test_clear_removes_all(self):
        """Clear removes all errors."""
        collector = ErrorCollector()
        collector.add(path="a.py", message="E1")
        collector.add(path="b.py", message="E2")

        collector.clear()

        assert len(collector.errors) == 0
        assert collector.has_errors is False
