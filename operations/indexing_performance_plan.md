# Indexing Flow Performance Plan

**Branch:** `lance-index-perf` (from `lance-upgrade-0.34` / LanceDB 0.34)  
**Scope:** Bulk `chunkhound index` throughput and memory for large codebases  
**Non-goals:** Real embedding API latency (use fake provider), search quality, MCP

---

## 1. Goals and constraints

| Goal | Constraint |
|------|------------|
| Maximize **chunks/sec** through parse → store → embed | Memory stays **O(page/batch)**, not O(corpus) |
| Make **DB write path** as small as possible | LanceDB OSS: single serial DB executor thread |
| Keep large-index invariants from lance-take2 | Ordered keyset missing-embed pages; no full-table loads |
| Fake embeddings for profiling | Embed CPU must not dominate wall time |

**Success metric (profiled, not guessed):**  
For N chunks on LanceDB + `FakeEmbeddingProvider`, report wall split:

```
discovery | parse | change-detect | chunk-store (merge_insert) | embed (fake) | embed-store (merge_insert) | optimize
```

Target after improvements: **chunk-store + embed-store dominate less**; ideally wall ≈ parse + fake-embed floor, with DB overhead clearly measured.

---

## 2. Current indexing pipeline (where time goes)

### 2.1 Directory index (`IndexingCoordinator.process_directory`)

```
Discover files (parallel)
  → Cleanup orphans (optional)
  → Change detection (batch meta + per-file mtime/size/hash)
  → Parse batches (ProcessPool / CPU)
  → Store per file (serial DB thread):
        begin_tx
        upsert file
        load existing chunks (if reindex)  ← smart diff
        delete modified/removed chunks
        insert_chunks_batch (merge_insert)  ← WRITE #1 (no/null embedding)
        commit
  → (per file or later) generate embeddings
  → generate_missing_embeddings (paginated residual):
        page missing chunks
        FakeEmbeddingProvider.embed_batch
        insert_embeddings_batch (merge_insert)  ← WRITE #2 (vectors)
        maybe optimize fragments
```

### 2.2 Hot paths already correct for large indexes

- **Missing-embed stream** pages via keyset (`get_chunks_without_embeddings_paginated`) — do not reintroduce full-table scans.
- **Soak script** `scripts/soak_large_index.py` already uses `FakeEmbeddingProvider` for insert + stream-embed timing.
- **`BatchMetricsCollector`** + `perf_analyzer` exist for embed API vs DB insert split within embedding service.

### 2.3 Double write (user idea — primary optimization candidate)

Today a new chunk typically incurs **two Lance `merge_insert`s**:

1. `insert_chunks_batch` — text + metadata, embedding null/empty  
2. `insert_embeddings_batch` — same row id, fill embedding/provider/model  

Each merge_insert:

- Joins on `id` (scalar index helps)
- May create fragments → threshold optimize
- Serializes on the **single DB executor**

**Hypothesis:** With fake embeddings (near-zero IO), wall time is dominated by **#1 + #2 + optimize**, not embed math.

**Proposed direction (defer materialize):**

- Hold parsed chunks in memory **only for the current parse/store batch** (bounded).
- Embed while still in-process (no intermediate DB row, or only file-level bookkeeping).
- **Single** `merge_insert` of fully formed rows (content + vector).

**Must preserve:**

| Concern | Approach |
|---------|----------|
| Crash safety | File not marked fully indexed until commit; resume re-parses incomplete files |
| Smart diff / embed preserve | Diff in memory against existing rows still works; unchanged chunks keep vectors |
| Realtime path | Realtime already often `skip_embeddings=True` then async embed — keep or align carefully |
| Memory | Never accumulate all repo chunks; cap by batch (e.g. `db_batch_size` / parse batch) |
| Partial embed failure | Either all-or-nothing per file batch, or write without vector only for failed subset (document choice) |

---

## 3. Other bottleneck candidates (profile to rank)

Ordered by likelihood under **fake embed + large N**:

| ID | Bottleneck | Why | Memory risk if “fixed wrong” |
|----|------------|-----|------------------------------|
| B1 | **Double merge_insert** | 2× join/write/fragment | Deferred write reduces DB ops; keep batch-bounded |
| B2 | **Serial DB executor** | All Lance ops on one thread | Parallel writers need careful MVCC + locks; prefer fewer ops first |
| B3 | **Per-file transactions** | High commit overhead at small files | Multi-file commit batches (bounded) |
| B4 | **Reindex: load all file chunks** | `get_chunks_by_file_id` full file for smart diff | Diff by content hash index later; keep per-file only |
| B5 | **Schema migration `to_pandas()`** | First fixed-dim embed may rewrite whole table | Ensure dims known at create; never migrate mid-large-index |
| B6 | **Fragment optimize storms** | Post-write optimize on every threshold cross | Coalesce optimize; raise threshold under LSM 0.34 |
| B7 | **Change detection** | Large path maps | Already batch meta; profile only if cold start heavy |
| B8 | **Parse pool** | CPU-bound tree-sitter | 3.14 free-threading may help; measure vs ProcessPool |
| B9 | **Fake embed still copies large text lists** | Batch build overhead | Stream pages; avoid duplicate text buffers |

**Explicitly out of scope for “faster by loading more”:** loading all chunks into RAM, full-table sorts, building giant PyArrow tables spanning the repo.

---

## 4. Profiling methodology

### 4.1 Environment

- Branch: `lance-index-perf`
- DB: LanceDB 0.34 (this fork)
- Embed: `tests.fixtures.fake_providers.FakeEmbeddingProvider` (deterministic, no network)
- Python in current venv: **3.12.12** — free-threading experiments need a separate **3.13t/3.14t** interpreter later
- Prefer **fresh temp DB** for cold profile; optional second run for warm reindex

### 4.2 Workloads

| Workload | Purpose |
|----------|---------|
| **Synthetic soak** (`scripts/soak_large_index.py`) | Isolate seed insert + stream-embed DB cost (no parse) |
| **Full index of this repo** (or a fixed fixture tree) | Real parse + change-detect + store + embed |
| **Scale ladder** | 1k / 10k / 50k chunks (synthetic) — detect non-linear fragment blowups |

### 4.3 Instrumentation plan (Phase 0 — implement first)

1. **Phase timers** in `process_directory` (extend `profile_startup` pattern to full phases):
   - discover, cleanup, change_detect, parse, store, embed_missing, optimize
2. **DB op counters** on Lance provider (thread-local or provider-level):
   - `merge_insert` count + rows + wall
   - `optimize` count + wall
   - `list_indices` / scalar create
3. **Reuse** `BatchMetricsCollector` for embed_api vs db_insert inside embedding service
4. **Optional:** `pyinstrument` or `cProfile` one-shot on `uv run chunkhound index` with fake provider wired via test harness or small CLI flag
5. **Memory:** sample `tracemalloc` or `psutil` RSS at phase boundaries (peak, not continuous)

### 4.4 Fake provider wiring for full CLI index

Today soak uses FakeEmbeddingProvider in-process. For full CLI profile, add one of:

- **Preferred for experiments:** thin harness `scripts/profile_index.py` that builds coordinator + FakeEmbeddingProvider + LanceDB (no API keys), mirrors `chunkhound index` phases  
- Or config provider plugin later — do not block profiling on productized CLI flags

Harness must:

- Stream/page embeddings (existing EmbeddingService path)
- Print phase table + RSS peak
- Exit non-zero if residual missing embeddings remain

### 4.5 What “good profile” looks like

Example target output:

```
chunks=10000 files=100 provider=lancedb embed=fake
discover     0.12s
parse        1.80s
store        4.50s   merge_insert_chunks=100  rows=10000
embed_fake   0.40s
embed_store  4.20s   merge_insert_embed=10   rows=10000
optimize     1.10s   calls=5
TOTAL       12.1s   peak_rss=420MB
```

If `store + embed_store + optimize` ≫ parse + embed_fake → deferred-write and/or fragment policy wins.

---

## 5. Improvement tracks (after profile ranks them)

### Track A — Deferred single write (user idea) — likely highest impact

**Design sketch:**

1. Parse batch → list of `Chunk` models (memory-bounded by batch).
2. Format texts → `FakeEmbeddingProvider.embed_batch` (or real provider).
3. Build PyArrow rows with embedding filled.
4. One `merge_insert` per batch (or per file batch).
5. File row upsert still first (path/mtime/hash) so change detection works; file can be marked incomplete until chunk commit succeeds.

**Incremental rollout:**

| Step | Deliverable | Risk |
|------|-------------|------|
| A0 | Profile harness + baseline numbers | None |
| A1 | Feature flag `indexing.defer_chunk_write` / env | Easy off-ramp |
| A2 | New-file path only (no existing chunks) | Lower |
| A3 | Reindex path with smart diff | Higher |
| A4 | Realtime alignment | Careful |

**Contract tests:**

- Crash mid-batch: re-run indexes cleanly  
- Unchanged file still skips  
- Residual missing empty after successful run  
- Soak: no full-table load mocks still pass  

### Track B — DB write hygiene (safe with 0.34 MemWAL)

- Ensure scalar `id` BTree exists **before** bulk merges (already on connect; verified empty-table fix).
- Coalesce optimize: once per N batches or wall interval, not every threshold trip mid-page.
- Prefer fixed-size embedding schema at table create when dims known from Fake/provider (avoid B5).
- Larger `db_batch_size` only after measuring fragment + RSS (do not default to “huge”).

### Track C — Parallelism (Python free-threading)

| Layer | Today | 3.14 free-threading option |
|-------|-------|----------------------------|
| Parse | ProcessPool | Could try threads on free-threaded build if tree-sitter GIL-free enough |
| Embed API | async concurrent batches | Keep; fake is CPU of hash+vector |
| **DB** | **One serial executor** | Keep serial for Lance correctness; free-threading does **not** magically parallelize Lance OSS safely |

**Rule:** Use free-threading for **CPU-side** work (parse, text prep, fake embed) that can run while the DB thread is busy — **pipeline**, not multi-writer DB.

Concrete idea after A0:

```
[parse batch k+1]  ||  [embed batch k]  ||  [DB write batch k-1]
```

Bounded queues between stages; backpressure when queue depth × batch size exceeds memory budget.

### Track D — Smart diff cost

- Only load existing chunks for files that **failed** cheap mtime/size/hash skip.
- Already mostly true; profile reindex-all (`force_reindex`) separately.

---

## 6. Memory budget rules (non-negotiable)

1. **No** `get_all_chunks_with_metadata` / full `to_pandas` on hot path.
2. Page size for missing embeds ≤ `embedding.batch_size` (or explicit profile page size).
3. Parse result buffer ≤ one process-pool batch (or explicit cap).
4. Deferred-write buffer ≤ one store batch of chunks × (text + vector dims × 4 bytes).
5. Schema migration that loads entire table is **forbidden** mid large index — fail closed with “recreate DB” or preconfigure dims.

Rough vector memory:  
`rows × dims × 4` — e.g. 10k × 1536 × 4 ≈ **60 MB** per in-flight batch; keep concurrent batches small.

---

## 7. Phase plan

### Phase 0 — Measure (this branch first work)

- [x] `scripts/profile_index.py` with phase timers + RSS + merge_insert counters  
- [x] `IndexProfile` / `DbOpStats` + LanceDB op hooks  
- [x] Baseline soak classic vs deferred (see sample below)  
- [ ] Optional: full-repo `mode=index` baseline on large tree  

### Phase 1 — Cheap wins

- [x] Optimize coalesce cooldown (5s) after write-path optimizes  
- [x] Fixed-size schema on empty table when deferred write knows dims  
- [ ] Broader threshold tuning for 0.34 after more soaks  

### Phase 2 — Deferred single write (flagged)

- [x] `indexing.defer_chunk_write` + env `CHUNKHOUND_INDEXING__DEFER_CHUNK_WRITE`  
- [x] New-file path: embed then `insert_chunks_with_embeddings_batch`  
- [x] Tests: residual empty + batch roundtrip  
- [x] Re-measure vs Phase 0 (sample: 2k chunks, fake embed)

**Sample soak (2k chunks, page 100, LanceDB 0.34):**

| Mode | merge_insert calls | rows | TOTAL s | peak RSS MB |
|------|-------------------|------|---------|-------------|
| classic two-write | 40 | 4000 | ~3.9 | ~262 |
| `--defer-write` | 20 | 2000 | ~2.3 | ~225 |

### Phase 3 — Pipeline concurrency

### Phase 3 — Pipeline concurrency

- [ ] Overlap parse / embed / DB with bounded queues  
- [ ] Optional free-threaded Python smoke (if available)  
- [ ] Re-measure  

### Phase 4 — Decide defaults

- [ ] Promote flag if ≥X% faster at same RSS peak on large soak  
- [ ] Document ops knobs in `operations/lancedb_large_indexes.md`  

---

## 8. Risks and open decisions

| Decision | Options | Default lean |
|----------|---------|--------------|
| Crash mid deferred batch | Drop in-memory; re-parse file | Yes |
| Failed embeds in batch | Split write naked chunks vs fail file | Fail file / retry page (match current residual semantics) |
| Realtime | Keep skip_embeddings + background residual | Yes until Phase 2 proven |
| Free-threading | Optional CI matrix later | Measure on 3.12 first; 3.14t optional |

---

## 9. Immediate next commands (when implementing Phase 0)

```powershell
git checkout lance-index-perf
uv sync

# Synthetic DB-bound soak (already FakeEmbeddingProvider)
uv run python scripts/soak_large_index.py --provider lancedb --chunks 10000

# After profile harness exists:
# uv run python scripts/profile_index.py --provider lancedb --root . --chunks-cap none
```

---

## 10. References in tree

- `chunkhound/services/indexing_coordinator.py` — process_directory, store, embed  
- `chunkhound/services/embedding_service.py` — paginated missing embed  
- `chunkhound/providers/database/lancedb_provider.py` — merge_insert, optimize, fragments  
- `chunkhound/core/diagnostics/batch_metrics.py` — embed vs DB timing  
- `tests/fixtures/fake_providers.py` — `FakeEmbeddingProvider`  
- `scripts/soak_large_index.py` — synthetic soak  
- Prior large-index design: `operations/lancedb_large_index_reimplementation.md`
