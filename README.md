# mcp-codesearch

MCP server for semantic code search with AST-aware chunking, hybrid vectors, and query syntax.

## Prerequisites

- **Python 3.12+**
- **Linux or macOS** (uses POSIX file locking via vector-core; not compatible with Windows)
- [Qdrant](https://qdrant.tech/) vector database (default: `localhost:6333`)
- An OpenAI-compatible embedding API (e.g., llama.cpp, Ollama, or any `/v1/embeddings` endpoint; default: `localhost:8080`)

## Installation

Requires [vector-core](https://github.com/michaelkrauty/vector-core).

```bash
pip install git+https://github.com/michaelkrauty/vector-core.git@v1.0.5
pip install git+https://github.com/michaelkrauty/mcp-codesearch.git
```

Or clone both repos and install locally:

```bash
git clone https://github.com/michaelkrauty/vector-core.git
git clone https://github.com/michaelkrauty/mcp-codesearch.git
pip install -e vector-core/
pip install -e mcp-codesearch/
```

## Quick Start

```bash
# Register with Claude Code:
claude mcp add codesearch -- mcp-codesearch

# Or add to your MCP client config (e.g., claude_desktop_config.json):
# {
#   "mcpServers": {
#     "codesearch": {
#       "command": "mcp-codesearch",
#       "env": {
#         "VECTOR_QDRANT_URL": "http://localhost:6333",
#         "VECTOR_EMBEDDING_URL": "http://localhost:8080",
#         "VECTOR_EMBEDDING_MODEL": "your-model-name",
#         "VECTOR_EMBEDDING_DIM": "768"
#       }
#     }
#   }
# }
```

## Features

- **Hybrid Search**: Dense embeddings + sparse TF-IDF with RRF fusion
- **AST-Aware Chunking**: Tree-sitter extracts functions, classes, methods with context
- **18 Languages with AST Support**: Python, JS/TS, Go, Rust, Java, C/C++, Ruby, PHP, Swift, Kotlin, Scala, C#, SQL, JSON, YAML, TOML (line-based fallback for Bash, HTML, CSS, and other file types)
- **Query Syntax**: `function:name`, `class:name`, `path:prefix`, `-path:exclude`
- **Incremental Indexing**: Change detection via mtime+size before hashing
- **Query Preprocessing**: Synonym expansion (`fn` → `function`, `db` → `database`)

## Tools (11 total)

### Search (5)
| Tool | Description |
|------|-------------|
| `code_search` | Main search with auto-indexing |
| `search_multiple` | Search across multiple codebases |
| `search_changed` | Search in recently changed files (git-aware) |
| `find_similar` | Find code similar to a snippet |
| `find_references` | Find all usages of a symbol |

### Index Management (3)
| Tool | Description |
|------|-------------|
| `index_status` | Check indexing status, file count, pending changes |
| `force_reindex` | Force complete re-indexing |
| `preview_index` | Preview what would be indexed |

### Collection Management (3)
| Tool | Description |
|------|-------------|
| `list_collections` | List all indexed codebases |
| `delete_collection` | Remove index for a codebase |
| `cleanup_orphans` | Remove orphaned collections |

## Query Syntax

```bash
# Natural language (semantic search)
code_search("websocket reconnection logic")

# Function search
code_search("function:handleRequest")
code_search("fn:handleRequest")  # alias

# Class search
code_search("class:WebSocketClient")
code_search("cls:WebSocketClient")  # alias

# Path filtering
code_search("auth path:src/services")
code_search("test -path:vendor -path:node_modules")

# Struct search (Rust, C, Go)
code_search("struct:Message")

# Combined
code_search("function:process_data path:src -path:test")

# Exact phrase
code_search('"exact function name"')
```

### Synonym Expansion

Common abbreviations automatically expanded:
- `fn`, `func` → `function`
- `cls` → `class`
- `db` → `database`
- `ws` → `websocket`
- `auth` → `authentication`, `authorization`
- `req`, `res` → `request`, `response`

### Additional Query Syntax

```bash
# Alternative function search aliases
code_search("def:processData")
code_search("method:handleRequest")

# Type/struct alias
code_search("type:UserConfig")

# Scope filters (restrict to chunk types)
code_search("error scope:function")    # Only function chunks
code_search("model scope:class")       # Only class chunks
code_search("validate scope:test")     # Only test functions
code_search("handler scope:impl")      # Non-test code only
```

## Search Modes

| Mode | Description |
|------|-------------|
| `file` | File-level results (overview) |
| `chunk` | Function/class-level results (detailed) |
| `both` | Combined ranking (default) |

## AST Chunking

Tree-sitter extracts semantic units:
- Functions (with docstrings)
- Classes (with methods if small, or overview + separate methods if large)
- Methods (with parent class context)
- Modules (imports, top-level statements)

Fallback to line-based chunking for non-code files (JSON, YAML, TOML, Markdown).

## Path Boosting

Search results boosted/demoted by path:

| Pattern | Adjustment |
|---------|------------|
| `src/` | +10% |
| `lib/`, `core/` | +8% |
| `test/`, `tests/` | -10% |
| `vendor/` | -25% |
| `generated/` | -30% |

## Git Integration

`search_changed` supports git revisions:
```bash
search_changed("auth logic", since="HEAD~5")
search_changed("database", since="main")
search_changed("fix", since="abc123")
search_changed("config", since="3.days.ago")
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_QDRANT_URL` | `http://localhost:6333` | Qdrant server |
| `VECTOR_EMBEDDING_URL` | `http://localhost:8080` | OpenAI-compatible embeddings API |
| `VECTOR_EMBEDDING_MODEL` | *(required)* | Embedding model name (e.g., `nomic-embed-text`, `text-embedding-3-small`) |
| `VECTOR_EMBEDDING_DIM` | *(required)* | Vector dimension (must match your model, e.g., `768`, `1536`) |

Codesearch-specific settings (configured via environment variables with the `CODESEARCH_` prefix):

| Variable | Default | Description |
|----------|---------|-------------|
| `CODESEARCH_CLASS_SPLIT_THRESHOLD` | `50` | Lines threshold for splitting large classes |
| `CODESEARCH_CHUNK_MIN_LINES` | `10` | Merge chunks smaller than this |
| `CODESEARCH_CHUNK_MAX_LINES` | `500` | Max lines per fallback chunk |
| `CODESEARCH_CHUNK_OVERLAP_LINES` | `25` | Overlap between fallback chunks |
| `CODESEARCH_SEARCH_CACHE_MAX_SIZE` | `100` | Max cached search results |
| `CODESEARCH_SEARCH_CACHE_TTL_SECONDS` | `300` | Search cache TTL (seconds) |
| `CODESEARCH_SEARCH_CACHE_EVICTION_RATIO` | `0.2` | Fraction of cache to evict when full |
| `CODESEARCH_UPSERT_BATCH_TIMEOUT` | `300` | Batch operation timeout (seconds) |
| `CODESEARCH_UPSERT_CONCURRENCY` | `1` | Max concurrent upsert batches |
| `CODESEARCH_DELETION_CONCURRENCY` | `50` | Concurrent Qdrant operations during incremental indexing |

## Change Detection

Fast incremental updates:
1. Check mtime + size (skip unchanged files)
2. Hash only modified files
3. Re-index only changed chunks

Avoids full re-embedding on every search.

## Storage

| Data | Location |
|------|----------|
| Index | Qdrant collection `codesearch_{path_hash}` |
| Metadata | Stored in Qdrant point payloads |

Each indexed codebase gets a unique collection based on path hash.

## Supported Languages

**Full tree-sitter AST support (18 languages):**
Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, Swift, Kotlin, Scala, C#, SQL, JSON, YAML, TOML

**Line-based fallback:** Bash, HTML, CSS, and all other file types (Markdown, Vue, Svelte, config files, etc.) are indexed with line-based chunking.

## Dependencies

Requires vector-core components:
- EmbeddingClient, GlobalVocabulary (embeddings)
- QdrantStorage, HybridSearcher (storage)

External libraries:
- tree-sitter-language-pack (AST parsing)
- pathspec (.gitignore support)
