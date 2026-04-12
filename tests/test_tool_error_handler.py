"""Tests for the tool_error_handler decorator."""

import logging

import pytest

from mcp_codesearch.tools._errors import tool_error_handler


class TestToolErrorHandler:
    """The decorator must never let FastMCP see an exception with empty __str__.

    FastMCP's Tool.run wraps exceptions as `f"Error executing tool {name}: {e}"`,
    so a bare `TimeoutError()` or `BrokenPipeError()` turns into a useless
    "Error executing tool X: " at the client. The decorator's job is to
    (a) log the full traceback, and (b) make sure the re-raised exception
    always has a non-empty message containing the original exception type.
    """

    async def test_passes_return_value_through_on_success(self):
        @tool_error_handler
        async def ok_tool(x: int) -> str:
            return f"result={x}"

        assert await ok_tool(7) == "result=7"

    async def test_preserves_exception_with_useful_message(self, caplog):
        @tool_error_handler
        async def raising_tool() -> str:
            raise ValueError("something specific broke")

        with caplog.at_level(logging.ERROR, logger="mcp_codesearch.tools._errors"):
            with pytest.raises(ValueError, match="something specific broke"):
                await raising_tool()

        # Full traceback should be logged to stderr for post-mortem.
        assert any(
            "raising_tool" in r.message and "ValueError" in r.message
            for r in caplog.records
        )

    async def test_wraps_empty_message_exception_with_type_name(self, caplog):
        """Bare TimeoutError() has str(e) == ''; must not reach the client empty."""

        @tool_error_handler
        async def timing_out_tool() -> str:
            raise TimeoutError  # noqa: TRY003 - intentional bare raise

        with caplog.at_level(logging.ERROR, logger="mcp_codesearch.tools._errors"):
            with pytest.raises(RuntimeError) as exc_info:
                await timing_out_tool()

        # Client-facing message must include the original exception type.
        assert "TimeoutError" in str(exc_info.value)
        # Must preserve the chain so debuggers can still see the real cause.
        assert isinstance(exc_info.value.__cause__, TimeoutError)

        # And the full traceback must still hit stderr.
        assert any(
            "timing_out_tool" in r.message and "TimeoutError" in r.message
            for r in caplog.records
        )

    async def test_wraps_broken_pipe_error(self):
        """Another empty-__str__ classic."""

        @tool_error_handler
        async def disconnected_tool() -> str:
            raise BrokenPipeError

        with pytest.raises(RuntimeError) as exc_info:
            await disconnected_tool()

        assert "BrokenPipeError" in str(exc_info.value)

    async def test_leaves_base_exceptions_alone(self):
        """KeyboardInterrupt / SystemExit must propagate unchanged."""

        @tool_error_handler
        async def cancellable_tool() -> str:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            await cancellable_tool()

    async def test_preserves_function_metadata(self):
        """functools.wraps means FastMCP's signature introspection keeps working."""

        @tool_error_handler
        async def documented_tool(x: int, y: str = "default") -> str:
            """Original docstring."""
            return f"{x}-{y}"

        assert documented_tool.__name__ == "documented_tool"
        assert documented_tool.__doc__ == "Original docstring."
        # __wrapped__ is set by functools.wraps so inspect.signature() can
        # follow through to the original.
        assert hasattr(documented_tool, "__wrapped__")
