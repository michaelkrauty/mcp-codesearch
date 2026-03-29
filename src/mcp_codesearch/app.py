"""FastMCP application instance for mcp-codesearch."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_codesearch import __version__

mcp = FastMCP(
    "codesearch",
    instructions=(
        "Use code_search for semantic/conceptual queries about what code does. "
        "More powerful than grep for understanding code behavior, finding "
        "implementations by concept, or exploring unfamiliar codebases."
    ),
)
mcp._mcp_server.version = __version__
