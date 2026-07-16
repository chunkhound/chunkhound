# LanceDB Large-Index Reimplementation

**Branch:** `lance-take2` (based on upstream `chunkhound/chunkhound` main)  
**Reference:** fork `main` worktree at `../chunkhound-fork-lancedb-ref`  
**Status:** Phase 3 in progress (Phases 1–2 complete)  
**Date:** 2026-07-17

## Goal

Bring large-codebase LanceDB + streaming/paginated indexing and embedding behavior
from this fork's `main` onto current upstream-aligned main **without** merging the
divergent history or regressing real-main features (Rust scanner, DuckDB compaction,
MCP daemon, watchman, Matryoshka, git-diff search, websearch, etc.).

## Non-goals

- Mechanical merge/rebase of fork `main`
- Porting LLM/Grok/setup/autodoc experiments from the fork squash
- Changing the default database provider away from DuckDB
- Deleting `get_all_chunks_with_metadata` until all callers are migrated
- Matching fork line-by-line

## Problem statement

Current `EmbeddingService.generate_missing_embeddings` loads **all** chunk metadata
via `get_all_chunks_with_metadata()`, then filters for missing embeddings in process
memory. That works for small projects and OOMs / times out on large indexes
(100k+ files, 1M+ chunks).

The fork solved this with:

1. Provider-level **paginated** "chunks without embeddings" queries
2. Optional schema (`embedding_signature`, `embedding_status`) for cheap filters
3. A streaming embed loop that pages until empty
4. LanceDB native queries (avoid full-table pandas loads)
5. Fragment optimize / dedup under heavy write load

## Architecture principles

1. **Contracts over commits** — reimplement capabilities against current interfaces.
2. **Dual-backend** — every new DB API works on DuckDB and LanceDB.
3. **Respect real-main seams** — async methods, DuckDB compaction, `should_optimize`,
   daemon lock semantics, serial executor timeouts.
4. **Stream hot paths** — anything that can see 10⁵–10⁶ chunks must not full-load.
5. **Ruthless scope** — large-index / embedding pipeline only per PR.

## Provider contract (target)

```python
def get_chunks_without_embeddings_paginated(
    self,
    provider: str,
    model: str,
    *,
    limit: int = 1000,
    after_id: int | None = None,
) -> list[dict[str, Any]]:
    """Return up to `limit` chunks missing embeddings for provider/model.

    Ordered by chunk id ascending. When `after_id` is set, only return rows
    with id > after_id (keyset pagination). When the caller inserts embeddings
    between pages, omitting after_id and re-querying is also valid (streaming
    cursor over a shrinking residual set).

    Each dict is provider-agnostic:
      id, file_id, code, symbol, file_path, start_line, end_line,
      chunk_type, language
    """
```

### Backend notes

| Backend | How "missing embedding" is determined (Phase 1) |
|---------|--------------------------------------------------|
| **DuckDB** | `NOT EXISTS` against all `embeddings_<dims>` tables for (provider, model); `ORDER BY id LIMIT` + optional `id > after_id` |
| **LanceDB** | Chunks table stores embedding/provider/model inline; filter rows that lack a valid embedding for that provider/model; keyset by id |

Phase 2 may add `embedding_signature` / `embedding_status` on LanceDB (and optionally
DuckDB) for cheaper filters and permanent-failure skipping. Phase 1 must work
**without** those columns so existing DBs keep working.

## Phased delivery

### Phase 0 — Inventory (done)

- Worktree: `../chunkhound-fork-lancedb-ref` → fork `main` @ large-index tip
- Capability list documented here
- Approach: redevelop on `lance-take2`, not merge

### Phase 1 — Paginated missing-embeddings API (this phase)

**In scope**

- Protocol method on `DatabaseProvider`
- DuckDB + LanceDB implementations
- Integration tests: empty page, multi-page keyset, residual shrink after insert
- No change to `EmbeddingService` call path yet (old path still default)

**Out of scope**

- Schema migrations (`embedding_signature`, `embedding_status`)
- Embedding service rewrite
- Serial-executor timeout retry
- Fragment-threshold config changes

**Acceptance**

- Both providers return stable dict shape
- Residual-shrink (`after_id=None`) is the large-index-safe streaming mode (bounded by `limit`)
- Multi-page keyset walk with `after_id` is ordered/correct; on LanceDB it may scan remaining candidate ids (not for multi-million residual sets until Phase 2 indexes)
- Query failures raise (empty page means completion, not soft-fail)
- LanceDB “missing” aligns with valid-embedding semantics (null/empty labels, null or invalid/zero vectors)
- Existing smoke tests pass
- DuckDB default path unchanged for CLI/MCP

### Phase 2 — Streaming embed service (current)

**In scope**

- Rewrite `generate_missing_embeddings` to page via Phase 1 API (ordered keyset walk)
- Stop using `get_all_chunks_with_metadata` on the missing-embeddings hot path
- Fix `_get_chunks_by_ids` to fetch by id (no full-table load) for regenerate paths
- Preserve Matryoshka / current provider config wiring and existing return status shape

**Deferred (still optional later / Phase 3 prep)**

- LanceDB `embedding_signature` / `embedding_status` schema + indexes
- Full embedding error-classification / permanent-failure status machine from the fork

**Acceptance**

- `generate_missing_embeddings` never calls `get_all_chunks_with_metadata`
- Ordered keyset walk embeds all missing chunks (fake provider + dual backend)
- Exclude patterns advance without infinite loop or skipped keep-ids
- Failed/partial embed page returns error (does not advance past still-missing work)
- Existing embedding pipeline integration tests still pass when API keys available

### Phase 3 — LanceDB write/read scaling (current)

**In scope**

- Harden `get_existing_embeddings` (no full-table `head().to_pandas()`; batched IN + prefer-embedded)
- Post-write optimize when fragment count ≥ threshold (chunk insert + embedding insert, in-executor)
- Regex: lightweight id scan + full-row fetch for page only; stable totals/pagination
- Fragment-aware dedup already used on read/missing paths (prefer embedded)

**Acceptance**

- `get_existing_embeddings` never full-loads the table on targeted lookups (`head` / unfiltered `to_pandas`); query failures raise
- Empty `chunk_ids` scans provider/model-matching rows only (vectors included; count callers may still be heavy on huge DBs)
- Zero-vector placeholders are not treated as existing embeddings
- Embedding/chunk insert runs optimize when over fragment threshold (in-executor only)
- Regex pagination: disjoint pages, consistent total, ordered ids
- Prior Phase 1–2 tests still green

### Phase 4 — Coordinator / realtime / config

- Indexing coordinator glue (provider-aware batch sizing / optimize)
- Watchman/realtime: store first, then missing-embeddings path
- Config knobs + docs for large LanceDB indexes
- Optional debug scripts from fork

### Phase 5 — Hardening

- Large synthetic / real-repo soak
- Full pytest before any upstream PR

## Explicit non-port list (from fork main)

| Material | Action |
|----------|--------|
| Autodoc / code-mapper experiments in fork squash | Leave |
| Fork LLM/Grok provider churn | Leave (upstream has its own stack) |
| Interactive setup wizard / terminal providers | Leave |
| Kitchen-sink squash commit as a patch | Leave |
| Debug scripts | Optional later under `scripts/` only |
| Delete `get_all_chunks_with_metadata` | Defer until callers migrated |

## Risk register

| Risk | Mitigation |
|------|------------|
| Break DuckDB (default) | Dual-backend + DuckDB tests in Phase 1 |
| Break daemon / compaction timeouts | No serial-executor changes in Phase 1 |
| Schema migration pain | Phase 1 has no schema change |
| Scope creep | PR checklist: large-index only |
| False confidence | Phase 5 soak; Phase 1 contract tests first |

## Reference locations (fork worktree)

| Need | Path under `../chunkhound-fork-lancedb-ref` |
|------|-----------------------------------------------|
| Protocol | `chunkhound/interfaces/database_provider.py` |
| LanceDB page query | `chunkhound/providers/database/lancedb_provider.py` (`get_chunks_without_embeddings_paginated`) |
| DuckDB page query | `chunkhound/providers/database/duckdb_provider.py` |
| Streaming embed loop | `chunkhound/services/embedding_service.py` (`EmbeddingBatchProcessor`) |
| Schema signature | LanceDB `get_chunks_schema` / connect migration |

## Progress log

- **2026-07-16:** Worktree created; design doc written; Phase 1 implementation started.
- **2026-07-16:** Phase 1 API landed:
  - `DatabaseProvider.get_chunks_without_embeddings_paginated`
  - DuckDB + LanceDB implementations (keyset `after_id` + residual-shrink)
  - Tests: `tests/integration/test_paginated_missing_embeddings.py`
  - LanceDB note: `limit` alone is not id-ordered; keyset path selects lightweight
    ids then fetches page rows. Residual-shrink (`after_id=None`) stays O(page).
- **2026-07-17:** Finish-work review fixes: raise on query failure; Lance missing
  predicate includes null/empty labels + null embedding + invalid/zero recovery;
  residual is documented as large-index mode; contract tests expanded.
- **2026-07-17:** Phase 2 streaming: `generate_missing_embeddings` pages via
  Phase 1 API; no full-table metadata load on hot path; tests in
  `tests/integration/test_streaming_missing_embeddings.py`.
- **2026-07-17:** Phase 3 scaling: targeted `get_existing_embeddings`, post-write
  fragment optimize, lighter regex pagination; tests in
  `tests/integration/test_lancedb_large_index_scaling.py`.
