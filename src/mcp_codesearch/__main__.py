"""Entry point for mcp-codesearch server."""

import sys

HELP_TEXT = """
mcp-codesearch - Semantic Code Search MCP Server

USAGE:
    mcp-codesearch          Start the MCP server (stdio transport)
    mcp-codesearch --help   Show this help message

DESCRIPTION:
    MCP server providing semantic code search using hybrid vectors
    (dense embeddings + sparse TF-IDF). Integrates with Claude Code
    and other MCP clients.

MCP TOOLS PROVIDED:
    code_search       Search indexed codebases semantically
    search_multiple   Search across multiple codebases
    search_changed    Search in recently changed files (git-aware)
    find_similar      Find code similar to a snippet
    find_references   Find all usages of a symbol
    force_reindex     Re-index a codebase
    index_status      Check indexing status
    list_collections  List indexed codebases
    preview_index     Preview what would be indexed
    delete_collection Delete an indexed codebase
    cleanup_orphans   Remove orphaned collections

SEARCH SYNTAX:
    function:name    Search for function by name
    class:name       Search for class by name
    path:prefix      Filter results to path prefix
    -path:exclude    Exclude paths containing string

EXAMPLES:
    # Add to Claude Code MCP config:
    claude mcp add codesearch -- mcp-codesearch

    # Search queries (via MCP):
    code_search("error handling", path=".")
    code_search("function:process_data", path=".", mode="chunk")

CONFIGURATION (environment variables):
    VECTOR_QDRANT_URL      Qdrant server URL (default: http://localhost:6333)
    VECTOR_EMBEDDING_URL   Embeddings API URL (default: http://localhost:8080)
    VECTOR_EMBEDDING_MODEL Embedding model name (required)
    VECTOR_EMBEDDING_DIM   Embedding dimensions (required)

For more information, see the project README.
"""


def main() -> None:
    """Run the MCP server or show help."""
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_TEXT)
        sys.exit(0)

    from mcp_codesearch.server import main as server_main  # noqa: PLC0415
    server_main()


if __name__ == "__main__":
    main()
