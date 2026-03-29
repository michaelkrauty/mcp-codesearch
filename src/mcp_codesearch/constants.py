"""Constants for mcp-codesearch.

Centralizes magic numbers and configuration values used across the codebase.
"""

# =============================================================================
# Exact Match Search
# =============================================================================

# Batch size for scrolling through points in exact match search
EXACT_MATCH_SCROLL_BATCH_SIZE = 1000

# Maximum scroll iterations to prevent infinite loops
EXACT_MATCH_MAX_ITERATIONS = 1000

# Score weights for exact match field matching (higher = better match)
EXACT_MATCH_NAME_SCORE = 3.0  # Symbol name match
EXACT_MATCH_SUMMARY_SCORE = 2.0  # File summary match
EXACT_MATCH_CONTENT_SCORE = 1.0  # Content body match

# =============================================================================
# Query Planning
# =============================================================================

# Minimum semantic search score before falling back to exact match
SEMANTIC_SCORE_THRESHOLD = 0.3

# =============================================================================
# Early Termination
# =============================================================================

# Number of high-quality results needed to trigger early termination
EARLY_TERMINATION_THRESHOLD = 10

# Score threshold to consider a result "high quality" for early termination
HIGH_QUALITY_SCORE = 2.5

# =============================================================================
# Synonym Expansion
# =============================================================================

# Maximum number of words to expand synonyms for (performance guard)
SYNONYM_MAX_EXPANSIONS = 5

# Maximum synonyms to add per word (limits query explosion)
SYNONYM_MAX_PER_WORD = 2

# =============================================================================
# Caching
# =============================================================================

# LRU cache size for point ID generation
POINT_ID_CACHE_SIZE = 10000
