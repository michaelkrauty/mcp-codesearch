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

Where the exception wraps another, the names of the wrapped causes are reported
too. A wrapper can be empty-messaged around a cause that is also empty-messaged
-- ``ResponseHandlingException(ReadTimeout(''))`` from ``qdrant_client`` is the
case seen in practice -- and there the type names are the only thing that
distinguishes a timeout from a connection failure.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from functools import wraps

logger = logging.getLogger(__name__)

# Enough to name a wrapper and what it wraps without turning an unrelated
# handling chain into a wall of text.
_MAX_CAUSE_DEPTH = 4


def _describe(exc: BaseException) -> str:
    """Name an exception together with the causes it wraps.

    A wrapper may stringify to nothing and carry its entire meaning in the
    exception it holds: ``qdrant_client`` reports a transport failure as
    ``ResponseHandlingException(ReadTimeout(''))``, where both layers have an
    empty message and the type names are the only signal that a timeout is what
    went wrong. Reporting the type alone leaves a caller unable to tell a
    timeout from a refused connection, so the chain is walked and named.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and len(parts) < _MAX_CAUSE_DEPTH:
        if id(current) in seen:  # defensive: chains can be cyclic
            break
        seen.add(id(current))

        # A __str__ of its own can raise, and this runs while already handling
        # a failure: letting that escape would replace the error being reported
        # with an unrelated one. Degrade to the type name instead.
        try:
            message = str(current).strip()
        except Exception:  # noqa: BLE001 - diagnostics must not add a failure
            message = ""
        name = type(current).__name__
        parts.append(f"{name}: {message}" if message else name)

        # Prefer the causes a raiser stated deliberately -- an explicit
        # ``raise ... from e``, then a wrapper holding its cause as its sole
        # argument (how qdrant_client wraps httpx errors) -- and fall back to
        # the implicit context last, since that one can be incidental.
        following = current.__cause__
        if following is None and current.args:
            first = current.args[0]
            if isinstance(first, BaseException):
                following = first
        # ``raise ... from None`` sets __suppress_context__ to say the context
        # is not the cause; reporting it anyway would contradict the raiser.
        if following is None and not current.__suppress_context__:
            following = current.__context__
        current = following

    return " <- ".join(parts)


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
                f"{_describe(e)} raised with no message "
                f"(see mcp-codesearch stderr log for traceback)"
            ) from e

    return wrapper
