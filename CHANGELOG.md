# Changelog

## [1.6.8] - 2026-06-20

### Fixed

- **A force re-index that fails partway through no longer corrupts the shared global vocabulary or leaves a partial collection that double-counts on the next access.** A forced full re-index drops the existing collection and rebuilds it, registering the codebase's full token contribution with the shared global vocabulary in Phase 1, before any points are embedded and upserted in Phase 2. If Phase 2 raised (a transient embedding-service or Qdrant error), the old index was already gone, the new collection held few or no points, and the full vocabulary contribution stayed registered, skewing IDF and therefore sparse ranking for every other indexed codebase. Worse, the surviving partial collection steered the next access onto the incremental path, whose additive vocabulary update counted the same tokens a second time. The force path now rolls back on any failure: it removes the codebase's vocabulary contribution and drops the partial collection, leaving a clean "not indexed" state so the next access performs a fresh full index. The rollback is best-effort and the original indexing error still propagates.

## [1.6.7] - 2026-06-20

### Fixed

- **`scope:test` and `scope:impl` now classify results by tokenized path and name matching instead of a bare `"test"` substring.** The filter kept a result for `scope:test` (and dropped it for `scope:impl`) whenever the literal `test` appeared anywhere in its path or symbol name, so files and symbols like `contest`, `latest`, `attestation`, `protest`, and `fastest` were misclassified as tests. A result now counts as a test only when its path, filename, or symbol name carries a `test` or `spec` word on a token boundary (camelCase humps, snake_case, dots, dashes). This recognizes the common conventions across ecosystems, including `tests/` and `__tests__/` directories, `test_foo.py`, `foo_test.go`, `UserServiceTest.java`, `FooSpec.kt`, `tests.py`, `foo.test.ts`, and `bar.spec.js`, while no longer treating substrings such as `contest` or `latest` as tests.
- **`scope:class` now also matches `enum`, type-alias, and `module` chunks.** The indexer emits `enum`, `type`, and `module` chunk types, but the `scope:class` filter only kept `class`, `class_overview`, `struct`, `interface`, and `impl`, so those definitions were silently excluded. They are now included.
- **`scope:struct`, `scope:enum`, `scope:interface`, `scope:type`, and `scope:module` are now recognized and normalized to `scope:class`.** Previously only `scope:function|class|method|test|impl` were parsed; any other scope token was left untouched in the query, applied no filter (a silent no-op), and leaked the literal `scope:...` text into the embedding query. `scope:method` continues to normalize to `scope:function`.

## [1.6.6] - 2026-06-19

### Fixed

- **`search_changed(since="<branch>")` now reports only what changed since the branch diverged, not files the target branch advanced on afterwards.** `_changed_files_since` ran `git diff --name-only <since>`, a two-dot diff against the *tip* of `<since>`. For a branch revision like `main` that has advanced since you branched off it, that set wrongly included files changed on `main` after the divergence and never touched on your branch, so `search_changed` could return matches in files you never modified, contradicting the documented "since diverging from main" intent. It now diffs the working tree against the merge-base of `<since>` and `HEAD`, so the result is exactly what changed on your side of the divergence, including uncommitted work. Ancestor revisions such as `HEAD~10` are unaffected (the merge-base is the revision itself), and relative-time queries (`3.days.ago`) still use `git log --since`.
- Aligned the vendored `SparseVectorizer` IDF test with vector-core 1.2.8, which recomputes IDF for the whole vocabulary on `extend_vocab` (the test still asserted the older new-tokens-only behavior).

## [1.6.5] - 2026-06-19

### Changed

- Bumped `vector-core` to `v1.2.8`. Pure dependency hygiene: v1.2.8 fixes `SparseVectorizer.extend_vocab` IDF recomputation and the `limit=0` semantics of two library list methods, plus two docstring corrections. mcp-codesearch builds sparse vectors through `GlobalVocabulary` (not the standalone `SparseVectorizer`) and never passes `limit=0` to those stores, so no behavior of this server changes. This keeps the pin current with the shared library.

## [1.6.4] - 2026-06-19

### Fixed

- **`find_references` no longer reports "No references found" for a symbol that is referenced but defined in many places.** The exact-match scan stops scrolling once it has collected 10 high-quality (name/summary) matches. That optimization is only safe when ranking, but `find_references` calls the scan with `rank=False` and then discards the name-equal definitions to surface usages. On a collection larger than one scroll page (1000 points), if the first page held 10 or more same-named definitions, the scan stopped before fetching later pages, so the actual usages (which scroll into those later pages) were never read and the tool returned nothing despite real references existing. The same early stop also affected the path-/scope-filtered exact search (`code_search "sym path:src"`), which likewise post-filters the scroll-order pool. Early termination is now gated on `rank`, so a `rank=False` scan reads the full scroll-order pool up to its cap; `scan_cap` and the scroll-iteration limit still bound the work, and ranked searches are unchanged.

## [1.6.3] - 2026-06-14

### Changed

- Bumped the shared `vector-core` library to v1.2.7 (FactStore: case-insensitive `query()`/`list_summaries()` type filters + rejection of inverted `valid_from`/`valid_to` ranges). mcp-codesearch indexes code and does not use FactStore, so no behavior of this server changes; this keeps the pin current with the shared library.

## [1.6.2] - 2026-06-13

### Changed

- Bumped the shared `vector-core` library to v1.2.6. v1.2.6 fixes `FactIndexer.index_all()`/`_train_vocabulary()` to index the complete fact corpus instead of the 50 most-recently-modified facts and to register the sparse vocabulary from all facts. mcp-codesearch does not use the fact indexer (it indexes code, not facts), so no behavior of this server changes; this keeps the pin current with the shared library.

## [1.6.1] - 2026-06-13

### Fixed

- Exact-match search now returns the highest-scored matches rather than the first ones reached in scroll order. The scan stopped as soon as it had collected `limit` matches, but scroll order is point-id order — unrelated to match quality — so a high-value name match (score 3.0) could be crowded out by lower-value content matches (1.0) that happened to scroll ahead of it. In practice `fn:Name` / `cls:Name` / `class:Name` lookups for a widely-referenced symbol returned files that merely *mention* the symbol while omitting its actual definition (the one chunk whose name equals the query). The scan now collects a candidate pool of `max(limit, 500)` matches, ranks by score, and returns the top `limit`, so the definition is no longer lost to truncation. The high-quality early-termination and per-collection scroll cap are unchanged, so cost stays bounded.

## [1.6.0] - 2026-06-12

### Added

- The `path` payload field now gets a full-text index (same parameters as the existing `content`/`name`/`summary` indexes), created on new collections and added lazily to existing ones. It powers retrieval-layer pushdown of filename constraints and is never used as a should-condition, so exact-match candidate selection on content fields is unchanged.

### Changed

- `file:` filename patterns are now pushed into the retrieval layer as a Qdrant `MatchText` filter on the indexed `path` field, instead of only post-filtering a bounded candidate pool. The pushdown extracts only tokens guaranteed (by fnmatch wildcard analysis under word-tokenizer semantics) to appear as whole path tokens in every matching file, so it can admit extra candidates but never exclude a true match; the precise fnmatch post-filter is unchanged. The constraint joins both the fast-path filter and the exhaustive scan of exact-match search, which makes the scan's early termination safe under filename constraints — previously a query like `fn:init file:db.py` could return nothing because `init` hits in other files exhausted the scan budget before `db.py` was ever reached. Because the tokens are only a superset (`db_pool.py` contains the tokens of `file:db.py`), the precise fnmatch constraint is additionally applied inside the exact-match scan itself, before any match counting, so token-level false positives within the constrained set cannot consume the scan budget either — this also holds when the token pushdown is unavailable. If a constrained search returns nothing, it reruns once without the token filter as a safety net for unforeseen tokenizer edge cases (e.g. unicode filenames). Patterns yielding no guaranteed tokens (such as `*test*`) and collections where index creation failed simply skip the pushdown.
- `search_changed` now pushes the changed-file set into retrieval as an exact `MatchAny` path filter, so ranking happens within the changed files only and a match can no longer rank below the candidate pool and silently disappear. The candidate pool shrinks accordingly (`limit*2`), the post-intersection remains as belt-and-suspenders, and the no-results message states plainly that nothing in the changed files matched. Change sets over 500 files fall back to the previous post-filter behavior (pool of `limit*20` capped at 200) to keep filter payloads small, and only then does the message mention the candidate-pool caveat.
- `path:` and `-path:` deliberately keep post-filtering over their widened candidate pool. Their semantics include substring-within-a-component matching (`path:earch` matches `src/mcp_codesearch/`, `path:src/mcp` matches `src/mcpx/`), so no token-based prefilter is a superset of their matches and a pushdown would drop valid results.

## [1.5.2] - 2026-06-12

### Changed

- Bumped the shared `vector-core` library to v1.2.5. `QdrantStorage.get_metadata` now only JSON-deserializes dict-shaped strings, symmetric with `store_metadata` which only serializes dicts. This improves the precision of mcp-codesearch's embedding-model guard: stored model names that happen to look like non-dict JSON (e.g. `"123"`) now round-trip as strings and are actually compared, instead of deserializing to a non-string and failing open (silently skipping the check).

## [1.5.1] - 2026-06-12

### Fixed

- Path boosting no longer clamps scores to 1.0, which had been collapsing exact-match relevance tiers (name=3.0, summary=2.0, content=1.0) into a tie. Previously the post-boost sort degraded to storage scroll order for exact results, so a `function:` query could truncate away the chunk literally named in the query in favor of incidental content matches, and test-path demotion had no effect on exact results. Tiers now survive boosting; score normalization to 0-1 still runs afterwards, so reported scores are unchanged in scale.
- The `file:pattern` query syntax now actually filters results. It was parsed and stripped from the search text but never applied, so `code_search("connection pooling file:db.py")` silently searched the whole codebase. The pattern is matched case-insensitively against the filename component using glob semantics: `file:db.py` keeps `src/db.py` but drops `src/db_pool.py`, and `file:*.sql` works as expected. It composes with `path:` and `-path:` filters, and is now documented in the README.
- `search_changed` is more honest and less likely to miss matches. It works by intersecting top-ranked whole-codebase results with the changed-file set; the candidate pool grew from `limit*5` to `limit*20` (capped at 200), the no-match message now states that nothing matched within the top-ranked candidates rather than claiming every changed file was searched, and the tool docstring documents the ranking-intersection limitation.

## [1.5.0] - 2026-06-12

### Added

- Exact-match search now uses Qdrant full-text payload indexes (`content`, `name`, `summary`) as a server-side pre-filter, replacing the previous scan of every point in the collection with a scan of only the points containing the query's tokens. Word-boundary matching semantics and field-based scoring are unchanged: a regex hit implies the query's own tokens are present, so the pre-filter cannot drop matches, and the token-anywhere candidates it adds are still rejected by the regex. Queries the index cannot serve safely (no alphanumeric content, or a word longer than the indexed token limit) skip the fast path, and a fast path that finds nothing falls back to the exhaustive scan. One deliberate precision improvement: fields over the regex size cutoff are matched by plain substring in the exhaustive scan, which can surface mid-token hits (`handle_request` inside `xhandle_requestx`) that contradict the word-boundary contract; the pre-filter does not produce these, so such false positives no longer appear alongside genuine word-boundary matches. They still appear, unchanged, when the fast path finds nothing and the exhaustive scan runs. True word-boundary occurrences in oversized fields tokenize at their boundaries and are therefore always visible to the pre-filter.
- Text indexes are created automatically on new collections and added lazily (idempotently) to existing collections on their first exact-match search. Index creation is additive and reversible: it never modifies points, and a failure merely leaves the search unaccelerated.

## [1.4.3] - 2026-06-12

### Changed

- Bumped the shared `vector-core` library to v1.2.4, which makes `FactStore.create()`/`update()` validate their inputs up front: blank subject/predicate/object/fact-type fields and out-of-range confidence values now raise `ValueError` before any database access. mcp-codesearch uses neither the facts store nor the glossary, so this is dependency hygiene with no behavior change for this server.

## [1.4.2] - 2026-06-12

### Changed

- Bumped the shared `vector-core` library to v1.2.3, which makes the glossary store's `update()` uniqueness check self-excluding: renaming an entry's term to a case variant of itself (e.g. "USAF" → "Usaf") or to one of its own aliases no longer fails with a spurious collision against the entry's own rows. mcp-codesearch does not expose glossary tools, so this is dependency hygiene with no behavior change for this server.

## [1.4.1] - 2026-06-12

### Changed

- Bumped the shared `vector-core` library to v1.2.2, which makes glossary store `create()`/`update()` atomic: alias validation (cross-entry collisions and case-normalized intra-list duplicates) now runs before any row is written, with a rollback-on-error backstop. mcp-codesearch does not expose glossary tools, so this is dependency hygiene with no behavior change for this server.

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
