# Rust indexing pipeline (`chunkhound_native`)

This file documents the design decisions and known gaps behind the Rust indexing
crate rooted here. It exists so an agent (or human) touching this code doesn't
have to re-derive *why* things are shaped this way from scattered comments and
git history. Read this before changing anything under `src/`, and treat items 4-5
as relevant even when editing the Python-side files they name.

**Maintenance policy:** update this file when the orchestration/threading model
changes, a new DB backend gains Rust support, a fallback condition changes, or a
"deliberate exclusion" below gets resolved (move it out of the gaps table). Do
NOT log individual bug fixes here — those belong in commit messages and
`CHANGELOG.md`. This doc records *why*, not a changelog. If you notice an entry
below is stale, fix it rather than leaving it.

## 1. Scope: orchestration with native embedding adapters

The Rust pipeline does NOT parse code. It owns file scanning, diffing,
threading/scheduling, DuckDB writes, and native embedding requests for the
OpenAI/Azure OpenAI and VoyageAI providers. All other provider names (or an
unrecognized provider/configuration combination) fall back to the existing
Python embedding callback:

- `IndexingPipeline.run()` takes `parse_batch_callback` and `embed_batch_callback`
  as Python callables and invokes the parse callback from a dedicated Rust
  thread. The embedding callback remains available as the per-run fallback.
- Native adapters live under `embed/`; the Python side of the fallback lives in
  `chunkhound/pipeline_bridge.py` (`embed_batch_callback`).
- Parsing remains entirely in the existing Python implementation: language
  detection and tree-sitter parsing happen in `parse_file_callback`.

This is a deliberate architecture, not a partial parsing migration in
progress. Embedding deduplication/reuse is explicitly out of scope here and
does not currently exist anywhere in the pipeline.

## 2. Module map

| Path | Purpose |
|---|---|
| `lib.rs` | `#[pymodule]` entry point; `scan_files()` — parallel file discovery via the `ignore` crate |
| `error.rs` | `DbError`/`ScanError` → `PyErr` conversions |
| `embed/{mod,callback,factory,openai,voyageai,retry,token}.rs` | Embedding trait, Python fallback, native providers, retries, and token-aware batching |
| `types.rs` | DB-facing serde structs shared across the PyO3 boundary |
| `db/mod.rs` | `DbBackend` trait, `DbConfig`, `create_backend()` |
| `db/duckdb_backend/mod.rs` | `DuckDbHnswBackend` struct, open/close lifecycle |
| `db/duckdb_backend/schema.rs` | DDL for files/chunks/embeddings tables |
| `db/duckdb_backend/write.rs` | Batched upserts/inserts, single-transaction batch writes |
| `db/duckdb_backend/read.rs` | `read_file_states()` (diff-phase snapshot), disk usage check |
| `db/duckdb_backend/hnsw.rs` | HNSW vector index drop/discover/rebuild (VSS extension) |
| `db/duckdb_backend/compaction.rs` | ATTACH+INSERT-SELECT DB compaction |
| `db/duckdb_backend/recovery.rs` | 3-phase swap-intent crash recovery for compaction |
| `pipeline/pipeline.rs` | `IndexingPipeline` `#[pyclass]` — orchestrates diff → parse → embed → store |
| `pipeline/differ.rs` | `compute_diff()` — filesystem vs. DB-snapshot diffing (mtime/hash based) |
| `pipeline/config.rs` | `PipelineConfig::from_py_dict()` |
| `pipeline/types.rs` | Internal `ParsedFile`/`NewChunk` — never exposed to Python |
| `pipeline/report.rs` | `PipelineReport` `#[pyclass]` returned to Python |
| `pipeline/parse_call_config.rs` | `ParseCallConfig` `#[pyclass]` passed into the Python parse callback |

## 3. PyO3 boundary design decisions

Baseline rules (`#![forbid(unsafe_code)]`, no `.unwrap()` at the boundary, no
borrowing `&str` across `py.allow_threads()`, always `allow_threads` for
CPU/IO-bound work) are defined once in the root `AGENTS.md` under `RUST_RULES`
— follow those, don't re-derive them here.

**Threading model** (`pipeline/pipeline.rs:416-440`): three persistent OS
threads — parse, embed, store — connected by two bounded `mpsc` channels
(capacity 2 each). While the store thread writes batch N to DuckDB (and, on the
final batch, rebuilds HNSW indexes and compacts), the embed thread is already
embedding batch N+1 and the parse thread is already parsing batch N+2. The
bounded channels provide backpressure since parsed/embedded batches are
memory-heavy (source text, then float vectors). The caller must release the GIL
before entering `.run()`; each thread re-acquires the GIL independently via
`Python::with_gil()`. The store thread reuses the existing HNSW "bulk mode"
bracket (`drop_all_hnsw_indexes()` → N incremental writes →
`ensure_all_hnsw_indexes()`) that the Python path already used, so no new
DB-layer mechanism was needed for this.

**Error handling**: `DbError`/`ScanError` (`error.rs`) always convert to
`PyRuntimeError`. On the Python side, `chunkhound/pipeline_bridge.py` wraps any
Rust exception as `RustPipelineError` so callers can distinguish native-pipeline
failures from other exceptions. There is **no automatic runtime fallback to
Python on a mid-run Rust failure** — `indexing_coordinator.py` reports
`{"status": "error", "pipeline": "rust", "rust_pipeline_error": True}` and stops;
it does not retry on the Python path.

`PipelineError` covers provider authentication, request, retry, context-length,
and response-validation failures. Its `is_fatal()` method means
non-retryable, not pipeline-aborting: by deliberate choice, a bad provider key
or malformed provider response is recorded in `PipelineReport.errors` and the
affected chunks are written with `embedding=NULL`.


**Fail-closed philosophy**: a scan that hits walk errors and ends up with zero
files must raise, not report an ordinary empty scan — an empty scan is
indistinguishable from every file having been deleted, and downstream cleanup
treats an empty scan as license to delete every DB row. The same fail-closed
pattern is applied to compaction crash recovery (`db/duckdb_backend/recovery.rs`).

## 4. Feature flag & fallback layers

`CHUNKHOUND_USE_RUST` defaults to on (`chunkhound/utils/rust_pipeline_flag.py`).
This is a deliberate, settled decision, not an experimental opt-in —
`chunkhound_native` is imported eagerly as a hard dependency. Don't re-raise
"should this default to off" in review; see the module's own docstring for the
rationale.

There are three independent, narrower fallback layers. The scanning and write
layers are pre-run capability checks, not runtime-exception recovery:

1. **Scanning** (`chunkhound/utils/file_patterns.py`): the Rust fast path
   (`chunkhound_native.scan_files`) only handles patterns that reduce to a pure
   extension, an exact filename, or literal path segments + `**`. It falls back
   to the Python `os.walk`-based scanner for character classes (`[...]`), `?`
   wildcards, non-`**` wildcard directory segments, and patterns anchored into a
   `HEAVY_DIRS`-named directory (Rust's `scan_files` always passes
   `skip_dirs=HEAVY_DIRS` with no anchor-awareness, which would wrongly drop
   those subtrees). A `RuntimeError` from the Rust scanner (the fail-closed
   empty-result case above) is deliberately re-raised, not swallowed as "empty."
   Note: the Python fallback scanner has its own known, *unfixed*
   anchor-pruning bug (wrongly prunes sibling directories when an anchored
   pattern is mixed with an unanchored one) — this is a pre-existing bug in the
   slow path, not something introduced by the Rust/fallback split.

2. **Writes** (`chunkhound/services/indexing_coordinator.py`,
   `resolve_rust_pipeline_decision()`): downgrades to the Python write pipeline
   when the DB provider doesn't declare `supports_rust_pipeline` (LanceDB
   doesn't — only `DuckDBProvider` does), has no `db_path`, or the DB filename
   isn't literally `chunks.db` (the Rust backend hardcodes this). Regression
   test: `tests/integration/test_rust_pipeline_lancedb_fallback.py`.

3. **Embedding provider selection** (`embed/factory.rs`): OpenAI/Azure OpenAI
   and VoyageAI use native Rust adapters; unknown or unsupported provider names
   use the Python callback supplied for the run. Python always supplies both
   the native configuration and that fallback, so this is selected per-run in
   Rust rather than by a Python pre-run capability gate.


Call chain for reference: `IndexingCoordinator.process_directory()` →
`chunkhound/services/rust_pipeline_runner.py::run_rust_indexing_phase()` →
`chunkhound/pipeline_bridge.py::run_rust_pipeline()` →
`chunkhound_native.IndexingPipeline(...).run(...)` (via `asyncio.to_thread`).

## 5. Known gaps / deliberate exclusions

| Gap | Status | Evidence |
|---|---|---|
| Rust pipeline is DuckDB-only | Deliberate. Previously caused a real bug (disconnected a live LanceDB provider and handed Rust a `db_path` it can't use) before the `supports_rust_pipeline` capability gate was added. | `tests/integration/test_rust_pipeline_lancedb_fallback.py` |
| Fast scanner only covers simple glob patterns | Deliberate scope; see §4. | `chunkhound/utils/file_patterns.py` |
| Anchor-pruning bug in the Python fallback scanner | Known, **unfixed** — pre-existing in the slow path, not caused by the Rust split. | `chunkhound/utils/file_patterns.py` comments |
| DuckDB FK / `ON CONFLICT DO UPDATE` two-phase write pattern | Inherent DuckDB limitation (can't `ON CONFLICT DO UPDATE` on an FK parent row), worked around with a two-phase pre-transaction delete/insert; documented atomicity gap between the phases. | `db/duckdb_backend/write.rs` |
| Cross-process DB reopen (open pipeline's DB in a separate process after close) | Verify current status before relying on it — the test is phrased as a hard guard, suggesting this was, or may still be, an unmet requirement. | `tests/contracts/test_reopen_after_close.py` |
| Platform/wheel coverage (Intel macOS, musl/pre-manylinux_2_34, no sdist, air-gapped builds) | Deliberate, documented gaps — not duplicated here. | Root `AGENTS.md`, `RUST_COMMANDS` section |
| Embedding deduplication/reuse | Absent by design; it never existed in this pipeline. Deferred to a future content-hash→vector lookup backed by a chunk-granularity `content_hash` column; a bare bloom filter is insufficient. | Future follow-up plan |
