# Changelog

## [1.4.0] - 2026-06-11

### Added

- **Embedding-model mismatch detection (same-dimension case).** v1.3.0 catches an embedding-model swap whose output dimension differs; this release closes the silent half: a swap to a *same-dimension* model passes the dimension check, every Qdrant operation succeeds, and searches quietly return meaningless results because query and stored vectors come from incompatible embedding spaces. The configured `VECTOR_EMBEDDING_MODEL` is now recorded in each collection's metadata at index time and compared whenever an existing collection is reused; a definite mismatch fails fast with a `force_reindex` hint, exactly like the dimension guard. The guard is fail-open: no configured model, missing metadata, or an unreadable stored value never blocks. Collections indexed before this release are stamped with the current model on first reuse, so protection begins immediately without reindexing (a one-time `last_updated` refresh per legacy collection is the only visible effect).

## [1.3.1] - 2026-06-11

### Changed

- Bumped the shared `vector-core` library to v1.2.1, a fix release (fact batch-read ordering, glossary `entry_hash` staleness, per-batch embedding progress callbacks, blank/duplicate glossary input validation). None of the fixed code paths are exercised by mcp-codesearch at runtime, so this is dependency hygiene with no behavior change for code search.

## [1.3.0] - 2026-05-31

### Added

- **Embedding-dimension mismatch detection.** When an already-indexed codebase is searched or re-indexed and its collection's stored dense vectors were built with a different embedding dimension than the one now configured — e.g. `VECTOR_EMBEDDING_MODEL` was switched to a model with a different output size — indexing now fails fast with a clear, actionable error that points at `force_reindex`, instead of letting a cryptic Qdrant dimension error surface deep inside a search (or, for a same-dimension model swap, silently returning meaningless results). The guard runs whenever an existing collection is reused, so it covers `code_search`, `search_multiple`, and `search_changed` through their shared auto-index path. It is fail-open: when the expected dimension is unknown (auto-detection has not resolved it) or the stored dimension cannot be read, indexing proceeds untouched — only a definite mismatch is refused. `force_reindex` is unaffected, since it deletes and recreates the collection with the current model.

## [1.2.2] - 2026-05-31

### Added

- **Jupyter notebook (`.ipynb`) indexing.** Notebooks are now discovered and made searchable by their *code*: a new `indexer/notebook.py` reduces a notebook to the source of its `code` cells (markdown, raw, and output cells are dropped), joined with `# %%` cell markers, which is then chunked as Python with full tree-sitter AST support — so a notebook's functions and classes are searchable like any other source file. The conversion is applied through a single shared read path used by both full discovery and incremental change-detection, so a notebook's content hash is computed over its extracted code (output-only edits don't trigger re-indexing, and notebooks never re-index perpetually). Code-less or unparseable notebooks are skipped rather than failing a codebase index. Supports both nbformat v4 (`cells`) and legacy v3 (`worksheets[*].cells` with `input`).

### Fixed

- **Incremental change-detection now removes a previously-indexed file that has become unindexable** — e.g. a notebook whose code cells were all deleted, a file that was emptied, or one that became unreadable. The fast scan still reported such a file by path (so it wasn't treated as deleted) while the content read yielded nothing (so it wasn't treated as modified), leaving its chunks stale in the index; it is now treated as a deletion. (Latent before this release; reachable for any file type, newly easy to hit via notebooks.)
- **Output-only notebook edits no longer cause a needless re-index.** Notebooks are excluded from the size-delta "definitely modified" shortcut and always hash-verified against their *extracted code*, so adding large cell outputs (which changes the raw file size but not the code) does not re-index the notebook.
- **A notebook's cell outputs no longer affect whether it is indexed at all.** Notebooks are judged by their (typically small) extracted code rather than their raw JSON size, so large embedded outputs can't push a notebook past `max_file_size_kb` — which would otherwise exclude it from discovery and then delete it from the index on the next incremental scan. (Non-notebook files are still subject to the size limit.)

## [1.2.1] - 2026-05-30

### Changed

- Bumped the `vector-core` dependency to `v1.2.0`.

### Fixed

- Annotated `valid_results` in the incremental-reindex token fetch (`indexing_service.py`) so `mypy src` is clean again; the empty-list initializer was inferred too narrowly.

## [1.2.0] - 2026-05-30

### Added

- **Cross-codebase global ranking for `search_multiple`** — pass `global_ranking=True` to merge results from every searched codebase into a single list ranked *across* codebases, each tagged with its source, answering "across all my repos, where is the best match?". Ranking uses Reciprocal Rank Fusion (via `vector-core`), which fuses by rank position and is therefore robust to the fact that raw similarity scores from different collections are not directly comparable. A result that appears in more than one codebase (e.g. a nested repo indexed both on its own and as part of its parent) is de-duplicated by absolute file location. Supported in `text`, `markdown`, and `json` output.

### Changed

- **`search_multiple` now searches codebases concurrently** instead of sequentially. Indexing and search for each codebase run under `asyncio.gather`, so overall latency is bounded by the slowest codebase rather than the sum of them all. Output ordering, per-codebase `=== path ===` grouping, and per-codebase error isolation are unchanged.

### Fixed

- **`search_multiple` no longer reports cached results as "No results found."** The grouped output keyed off `results_count`, which is left at `0` on a cache hit; it now renders the response's formatted output directly (matching `code_search`), so repeated multi-codebase searches show their results.

## [1.1.1] - 2026-05-30

### Changed

- Bumped the `vector-core` dependency to `v1.1.0`, which adds nested ignore-file support to `FileDiscovery`.

## [1.1.0] - 2026-05-30

### Added

- **Nested `.gitignore` support** — ignore rules are now honored at every directory level, matching git's "deeper files override shallower" semantics (including `!` re-include negations), instead of only the repository-root `.gitignore`.
- **`.codesearchignore` files** — exclude paths from indexing using gitignore syntax without affecting git's own behavior. Honored at any directory level. Useful for vendored code, generated files, or large data that should stay tracked by git but out of the index.
- **`.git/info/exclude` support** — repo-local (uncommitted) excludes are now respected. The user's global `core.excludesFile` is intentionally not consulted, keeping indexing reproducible across machines.

### Changed

- File discovery now prunes ignored directories during traversal (git does not descend into them), avoiding wasted work on excluded subtrees.
- `discover_files()` and `scan_file_metadata()` now share a single traversal, so full indexing and incremental change detection always agree on which files are indexable.

## [1.0.3] - 2026-05-27

### Fixed

- Aligned the runtime package `__version__` constant, project metadata, lockfile package entry, and version regression test.
- Bumped the `vector-core` dependency to `v1.0.5`, where `vector_core.__version__` matches package metadata.

## [1.0.2] - 2026-05-25

### Changed

- Bumped `vector-core` dependency to the reachable `v1.0.4` tag, aligning with corrected Vector Core release metadata.

## [1.0.1] - 2026-05-23

### Changed

- Tagged the first reproducible consumer release after pinning `vector-core` to `v1.0.3`.

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
