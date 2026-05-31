"""Configuration via environment variables."""

from __future__ import annotations

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from vector_core.settings import VectorCoreSettingsMixin


class CodeSearchSettings(VectorCoreSettingsMixin, BaseSettings):
    """Code-search specific settings.

    Inherits vector-core settings (embedding_url, qdrant_url, etc.) via mixin.
    """

    model_config = SettingsConfigDict(env_prefix="CODESEARCH_")

    # Tree-sitter / AST chunking settings
    class_split_threshold: int = 50  # Classes larger than this get overview + methods
    chunk_min_lines: int = 10  # Merge adjacent chunks smaller than this
    chunk_max_lines: int = 500  # Maximum lines per fallback chunk
    chunk_overlap_lines: int = 25  # Lines of overlap between fallback chunks

    # Code file extensions to index
    code_extensions: set[str] = {
        ".py", ".ipynb", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
        ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
        ".scala", ".cs", ".vue", ".svelte", ".md", ".sql", ".sh",
        ".yaml", ".yml", ".json", ".toml", ".html", ".css", ".scss",
    }

    # Qdrant tuning
    upsert_concurrency: int = 1  # Max concurrent upsert batches (1 for stability)
    upsert_batch_timeout: float = 300.0  # Timeout for batch upsert operations (5 minutes)
    deletion_concurrency: int = 50  # Max concurrent deletion operations during incremental indexing

    # Search result cache settings
    search_cache_max_size: int = 100  # Maximum cached search results
    search_cache_ttl_seconds: float = 300.0  # Cache TTL (5 minutes)
    search_cache_eviction_ratio: float = 0.2  # Fraction to evict when full

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> CodeSearchSettings:
        """Validate chunk settings to prevent invalid configurations."""
        if self.chunk_min_lines >= self.chunk_max_lines:
            raise ValueError(
                f"chunk_min_lines ({self.chunk_min_lines}) must be less than "
                f"chunk_max_lines ({self.chunk_max_lines})"
            )
        if self.chunk_overlap_lines >= self.chunk_max_lines:
            raise ValueError(
                f"chunk_overlap_lines ({self.chunk_overlap_lines}) must be less than "
                f"chunk_max_lines ({self.chunk_max_lines})"
            )
        return self


settings = CodeSearchSettings()


# Path-based score adjustments (additive, not multiplicative)
# Positive values boost, negative values demote
# Total adjustment is capped at PATH_BOOST_MAX to prevent score inversion
PATH_BOOST_PATTERNS: dict[str, float] = {
    # Boost source code (+)
    "src/": 0.10,
    "lib/": 0.08,
    "core/": 0.08,
    "app/": 0.05,
    "pkg/": 0.05,
    # Demote test/generated code (-)
    "test/": -0.10,
    "tests/": -0.10,
    "__test__/": -0.15,
    ".test.": -0.10,
    ".spec.": -0.10,
    "_test.": -0.10,
    "test_": -0.08,
    "mock/": -0.15,
    "mocks/": -0.15,
    "fixture": -0.18,
    "fixtures/": -0.18,
    # Demote vendored/generated (-)
    "vendor/": -0.25,
    "third_party/": -0.25,
    "generated/": -0.30,
    ".gen.": -0.25,
    ".pb.": -0.20,  # Protocol buffers
    "_pb2.": -0.20,
    # Demote examples (slightly -)
    "example/": -0.05,
    "examples/": -0.05,
    "sample/": -0.05,
}

# Maximum total adjustment from path boosting (prevents score inversion)
PATH_BOOST_MAX: float = 0.30

