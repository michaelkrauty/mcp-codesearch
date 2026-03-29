# Changelog

## [1.0.0] - 2026-03-20

Initial public release.

### Features

- **Semantic code search** using Qdrant and OpenAI-compatible embeddings
- **11 MCP tools**: code_search, find_references, find_similar, search_changed, search_multiple, index_status, force_reindex, preview_index, list_collections, delete_collection, cleanup_orphans
- **AST-aware chunking** for 18 languages (Python, TypeScript, Rust, Go, Java, C/C++, etc.)
- **Hybrid search** combining dense embeddings with TF-IDF sparse vectors (RRF fusion)
- **Query preprocessing** with code synonym expansion and structured query syntax (`function:`, `class:`, `path:`, `scope:`)
- **Git integration** for searching changed files (`--since`, branch diffs)
- **Multi-collection search** across multiple indexed codebases
- **Incremental indexing** with change detection (only re-indexes modified files)
- **Graceful degradation** to sparse-only search when embeddings are unavailable
