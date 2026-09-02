"""Stdio fixture server for progress-notification protocol tests."""

from __future__ import annotations

import time

from mcp.server import MCPServer

from mcp_codesearch.progress import ToolProgressMiddleware
from mcp_codesearch.services.indexing_service import _run_sync

server = MCPServer(
    "progress-test",
    middleware=[ToolProgressMiddleware(interval_seconds=0.01)],
)


@server.tool()
async def slow_tool() -> str:
    await _run_sync(time.sleep, 0.2)
    return "complete"


if __name__ == "__main__":
    server.run(transport="stdio")
