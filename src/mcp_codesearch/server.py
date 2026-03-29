"""FastMCP server with code search tools."""

from __future__ import annotations

import atexit
import logging
import sys
from typing import TYPE_CHECKING, Any

from vector_core import sync_cleanup_wrapper, verify_tools_registered

if TYPE_CHECKING:
    from vector_core import AsyncSingleton

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import mcp instance from app module (shared across tool modules)
# Import tool modules to register tools with mcp instance
from mcp_codesearch import tools  # noqa: E402, F401
from mcp_codesearch.app import mcp  # noqa: E402

# Re-export helper functions for backward compatibility
from mcp_codesearch.helpers import (  # noqa: E402, F401
    auto_index,
    format_index_message,
    validate_git_since,
)

# Import singletons module for cleanup access and backward-compatible re-exports
from mcp_codesearch.singletons import (  # noqa: E402, F401
    _embedder,
    _global_vocab,
    _indexing_service,
    _search_service,
    _storage,
    cleanup_resources,
    # Re-export singleton getters for backward compatibility
    get_embedder,
    get_global_vocab,
    get_indexing_service,
    get_search_service,
    get_storage,
)

# Expected tools for verification (catches silent import failures)
EXPECTED_TOOLS = [
    "code_search",
    "search_multiple",
    "search_changed",
    "find_similar",
    "find_references",
    "index_status",
    "force_reindex",
    "preview_index",
    "list_collections",
    "delete_collection",
    "cleanup_orphans",
]

# Re-export tools for backward compatibility with tests
from mcp_codesearch.tools.collections import (  # noqa: E402, F401
    cleanup_orphans,
    delete_collection,
    list_collections,
)
from mcp_codesearch.tools.indexing import (  # noqa: E402, F401
    force_reindex,
    index_status,
    preview_index,
)
from mcp_codesearch.tools.search import (  # noqa: E402, F401
    code_search,
    search_changed,
    search_multiple,
)
from mcp_codesearch.tools.similar import (  # noqa: E402, F401
    find_references,
    find_similar,
)

# ============= Cleanup =============


def _sync_cleanup() -> None:
    """Sync wrapper for cleanup, called on exit.

    Uses sync_cleanup_wrapper from vector-core for consistent cleanup
    handling across all MCP servers.
    """
    singletons: list[AsyncSingleton[Any]] = [
        _storage, _embedder, _global_vocab, _indexing_service, _search_service
    ]
    if not any(s.is_initialized for s in singletons):
        return

    # During interpreter shutdown, event loops can be in inconsistent states
    # Just reset singletons synchronously to avoid "Event loop is closed" errors
    if sys.is_finalizing():
        for singleton in singletons:
            try:
                singleton.reset()
            except (RuntimeError, AttributeError, TypeError):
                pass  # Expected during interpreter shutdown
            except Exception as e:
                try:
                    logger.debug(f"Unexpected error resetting singleton: {type(e).__name__}: {e}")
                except Exception:
                    pass  # Logger may be finalizing
        return

    sync_cleanup_wrapper(cleanup_resources, singletons)


# Register cleanup handler
atexit.register(_sync_cleanup)


# ============= Main =============


def main() -> None:
    """Run the MCP server."""
    # Verify all expected tools are registered (catches silent import failures)
    verify_tools_registered(mcp, EXPECTED_TOOLS, "mcp-codesearch")

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
