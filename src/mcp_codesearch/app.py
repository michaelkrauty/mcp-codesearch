"""MCPServer application instance for mcp-codesearch."""

from __future__ import annotations

from mcp.server import MCPServer

from mcp_codesearch import __version__

mcp = MCPServer(
    "codesearch",
    version=__version__,
    instructions=(
        "Use code_search for semantic/conceptual queries about what code does. "
        "More powerful than grep for understanding code behavior, finding "
        "implementations by concept, or exploring unfamiliar codebases."
    ),
)
