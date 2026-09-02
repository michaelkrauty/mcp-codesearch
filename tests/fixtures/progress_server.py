"""Stdio fixture server for progress-notification protocol tests."""

from __future__ import annotations

import asyncio

from mcp.server import MCPServer

from mcp_codesearch.progress import ToolProgressMiddleware

server = MCPServer(
    "progress-test",
    middleware=[ToolProgressMiddleware(interval_seconds=0.01)],
)


@server.tool()
async def slow_tool() -> str:
    await asyncio.sleep(0.08)
    return "complete"


if __name__ == "__main__":
    server.run(transport="stdio")
