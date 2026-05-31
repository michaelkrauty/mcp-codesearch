"""MCP Code Search - Semantic code search using Qdrant and OpenAI-compatible embeddings"""

__version__ = "1.2.1"


def main() -> None:
    """Run the MCP server."""
    from mcp_codesearch.server import main as _main

    _main()


__all__ = ["main", "__version__"]
