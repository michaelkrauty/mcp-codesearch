"""MCP Codesearch tool modules.

Modularized tool implementations for the mcp-codesearch server.
Each module handles a specific category of tools:

- search.py: Core search operations (code_search, search_multiple, search_changed)
- similar.py: Similarity and reference search (find_similar, find_references)
- indexing.py: Index management (index_status, force_reindex, preview_index)
- collections.py: Collection management (list_collections, delete_collection, cleanup_orphans)

Tools are registered via @mcp.tool() decorator when modules are imported.
Import all modules in server.py to register all tools.
"""

# Import all tool modules to register their tools with mcp
from mcp_codesearch.tools import collections, indexing, search, similar

__all__ = [
    "search",
    "similar",
    "indexing",
    "collections",
]
