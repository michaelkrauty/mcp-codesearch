"""Progress heartbeats for long-running MCP tool calls."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from mcp.server.context import CallNext, HandlerResult, ServerMiddleware, ServerRequestContext


class ToolProgressMiddleware(ServerMiddleware[Any]):
    """Keep progress-aware MCP clients from timing out healthy tool calls."""

    def __init__(self, interval_seconds: float = 15.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._interval_seconds = interval_seconds

    async def __call__(
        self,
        context: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> HandlerResult:
        if context.method != "tools/call" or context.request_id is None:
            return await call_next(context)

        name = context.params.get("name") if context.params else None
        tool_name = name if isinstance(name, str) else "tool"
        heartbeat = asyncio.create_task(self._report_progress(context, tool_name))
        try:
            return await call_next(context)
        finally:
            heartbeat.cancel()
            # Progress is advisory. A client disconnect or notification failure
            # must not replace the tool's actual result or exception.
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat

    async def _report_progress(
        self,
        context: ServerRequestContext[Any, Any],
        tool_name: str,
    ) -> None:
        """Report elapsed time until the wrapped request completes."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        while True:
            await asyncio.sleep(self._interval_seconds)
            elapsed = loop.time() - started
            await context.session.report_progress(
                elapsed,
                message=f"{tool_name} is still running",
            )
