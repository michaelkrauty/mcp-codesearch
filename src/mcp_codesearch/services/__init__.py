"""Service layer for mcp-codesearch.

Provides clean separation between MCP tool handlers and business logic.
"""

from mcp_codesearch.services.indexing_service import (
    IndexingService,
    IndexingStats,
    PreparedFile,
)
from mcp_codesearch.services.search_service import (
    SearchQuery,
    SearchResponse,
    SearchService,
)

__all__ = [
    "IndexingService",
    "IndexingStats",
    "PreparedFile",
    "SearchQuery",
    "SearchResponse",
    "SearchService",
]
