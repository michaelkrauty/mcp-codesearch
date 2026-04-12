"""Error handling wrapper for MCP tool entry points.

FastMCP's Tool.run catches tool exceptions and raises::

    ToolError(f"Error executing tool {name}: {e}")

For exceptions whose ``__str__`` returns an empty string (bare ``TimeoutError()``,
``BrokenPipeError()``, many asyncio/network errors constructed with no args),
this produces a useless ``"Error executing tool X: "`` at the MCP client with
no trailing detail, and the root cause is invisible because FastMCP doesn't log
the traceback either.

The ``tool_error_handler`` decorator closes both gaps: it logs the full traceback
via ``logger.exception`` so stderr always captures the real cause, and it
normalizes empty-message exceptions so the client-facing error always carries at
least the exception type name.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from functools import wraps

logger = logging.getLogger(__name__)


def tool_error_handler[**P, R](
    fn: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """Log full tracebacks and ensure MCP tool errors are never empty strings.

    Place this decorator BELOW ``@mcp.tool()`` in the stack so FastMCP registers
    the wrapped function::

        @mcp.tool()
        @tool_error_handler
        async def my_tool(...) -> str:
            ...

    Behavior:

    - On success: passes the return value through unchanged.
    - On exception with a non-empty ``str(e)``: logs the traceback, re-raises
      the original exception unchanged (preserving type + chain).
    - On exception with an empty ``str(e)``: logs the traceback, then re-raises
      inside a ``RuntimeError`` whose message contains ``type(e).__name__`` so
      FastMCP's ``f"...: {e}"`` produces something useful. The original
      exception is chained via ``__cause__``.
    """

    @wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            # Full traceback to stderr for post-mortem even when the MCP client
            # message is terse.
            logger.exception(
                f"Tool {fn.__name__} raised {type(e).__name__}: {e!r}"
            )
            if str(e):
                raise
            raise RuntimeError(
                f"{type(e).__name__} raised with no message "
                f"(see mcp-codesearch stderr log for traceback)"
            ) from e

    return wrapper
