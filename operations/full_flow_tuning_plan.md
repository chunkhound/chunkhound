# Full-Flow Indexing Tuning Plan

**Branch:** `lance-index-perf`  
**Harness:** `scripts/profile_index.py --mode full` (+ `--full-scale`)  
**Embed path:** always `FakeEmbeddingProvider` (no Voyage/network)  
**Gate metric:** end-to-end `wall_s` (not merge call counts alone)

---

## 1. Why this exists

Earlier soaks (`--mode soak`) proved the **DB write path** under free embeds:

| Path | What it measures | What it misses |
|------|------------------|----------------|
| **soak** | file insert, chunk build, fake embed, Lance append/merge, residual | discovery, change-detect, tree-sitter parse, process-pool, product store loop, hash, timeouts |
| **full** | **entire product cold index** with fake vectors | real Voyage latency (intentionally excluded) |

Product “Handling files” wall includes parse + serial store + in-process fake (or real) embed under `defer_chunk_write`. Soak never ran that pipeline, so **wall wins on soak do not automatically equal product wins**.

Full-flow keeps embeddings **meaningless for search quality** but still:

- allocates real-width vectors (`--dims 1024` default ≈ voyage-code-3)
- exercises `insert_chunks_with_embeddings` / residual `generate_missing`
- stresses Lance fragment/optimize policy on the product batch cadence

---

## 2. Pipeline under test (cold, defer)

```
discover
  → change_detect (all new)
  → parse ∥ workers  ──on_batch──► store (serial DB + optional defer embed)
  → generate_missing (residual; near-empty if defer succeeded)
```

Profile phases (when `IndexProfile` attached):

| Phase | Meaning |
|-------|---------|
| `discover` | file walk / parallel discovery |
| `change_detect` | mtime/size/hash skip logic |
| `parse_store` | wall of parse+store pipeline (overlapping) |
| `store` | **serial** time inside `_store_parsed_results` (+ store batches) |
| `store_optimize` | mid-batch fragment optimize |
| `generate_missing` | residual embed pass |
| `process_directory` | outer coordinator directory wall |
| **`wall_s`** | full harness wall (gate) |

Approx non-store parse share: `parse_store - store` (pipeline-overlapped; use as signal, not exact CPU).

---

## 3. How to run

### 3.1 Baseline cold (synthetic, recommended)

```powershell
# Small smoke (~seconds–tens of seconds)
uv run python scripts/profile_index.py --mode full --corpus synthetic `
  --files 50 --funcs-per-file 15 --defer-write --json

# Medium (primary tuning unit)
uv run python scripts/profile_index.py --mode full --corpus synthetic `
  --files 200 --funcs-per-file 20 --defer-write --keep-dir .bench-full

# Large-ish synthetic
uv run python scripts/profile_index.py --mode full --corpus synthetic `
  --files 1000 --funcs-per-file 20 --defer-write --keep-dir .bench-full
```

### 3.2 Scale ladder (wall gate across sizes)

```powershell
uv run python scripts/profile_index.py --full-scale --defer-write --page-size 128
# Optional: --funcs-per-file 20 --dims 1024 --json
```

### 3.3 Real tree (still fake embeds)

```powershell
uv run python scripts/profile_index.py --mode full --corpus root --root . `
  --defer-write --keep-dir .bench-repo-full
```

Use for “this repo feels slow” confirmation; prefer synthetic for A/B (determinism).

### 3.4 Resume (later track — scaffold ready)

```powershell
# 1) cold
uv run python scripts/profile_index.py --mode full --corpus synthetic `
  --files 200 --scenario cold --defer-write --keep-dir .bench-full

# 2) unchanged resume (should be discover + change_detect dominated)
uv run python scripts/profile_index.py --mode full --corpus synthetic `
  --files 200 --scenario resume --defer-write --keep-dir .bench-full
```

### 3.5 Dims / page-size sensitivity

```powershell
# Vector payload cost (write path)
uv run python scripts/profile_index.py --mode full --corpus synthetic --files 200 `
  --dims 32 --defer-write
uv run python scripts/profile_index.py --mode full --corpus synthetic --files 200 `
  --dims 1024 --defer-write

# Batch sizing (product db_batch_size ≈ page-size here)
uv run python scripts/profile_index.py --mode full --corpus synthetic --files 200 `
  --page-size 64 --defer-write
uv run python scripts/profile_index.py --mode full --corpus synthetic --files 200 `
  --page-size 256 --defer-write
```

### 3.6 Still use soak for pure write A/B

```powershell
uv run python scripts/profile_index.py --mode soak --chunks 50000 --defer-write
uv run python scripts/profile_index.py --scale --page-size 256
```

**Rule:** ship a cold-path change only if **full-flow wall** does not regress at the medium synthetic size (and soak wall does not regress at 50k+ when the change is DB-write-only).

---

## 4. Interpreting a report

1. Sort phases by time. Gate = `wall_s`.
2. If `store` ≫ (`parse_store - store`): **DB / defer embed / Lance** — continue write-path work (append batching, optimize policy). Cross-check with soak.
3. If (`parse_store - store`) large and `store` small: **parse / process pool / I/O** — do **not** optimize merge_insert next; look at batch sizing, worker count, timeouts, language mix.
4. If `discover` or `change_detect` large on **cold**: discovery or hash path; on **resume**: expected skip path cost.
5. If `generate_missing` large with defer on: **bug or residual leak** (defer should leave near-zero missing on new files).
6. `peak_rss_mb` must stay O(batch), not O(corpus). RSS↑ that tracks N is a fail even if wall↓ slightly.
7. Throughput: `chunks_per_s` / `files_per_s` for scale plots; compare same `--dims` and `--funcs-per-file`.

---

## 5. Tuning methodology (repeatable)

### Step A — Establish baseline

| Run | Command intent |
|-----|----------------|
| A1 | full synthetic 200 files, defer, dims=1024, keep-dir |
| A2 | full-scale ladder (50/200/500/1000) |
| A3 | soak 50k defer (write reference) |

Record: `wall_s`, phase table, `db.merge_insert_*`, `optimize_*`, peak RSS, chunks/s.

### Step B — Attribute bottleneck

Pick the largest phase family from A1:

| Dominant | Next experiments |
|----------|------------------|
| `store` | Write batch / append policy; page-size; thr; dims A/B; compare soak |
| parse-heavy | Worker counts, batch sizes in `_process_files_in_batches`, language parsers |
| residual | Residual empty? If not, fix defer/missing page path first |
| discover/hash | Parallel discovery threshold, hash cost on cold |

### Step C — One change at a time

- Single flag or single code path.
- Re-run **A1** (mandatory) + soak if write-related.
- Keep if **wall_s ↓** (or holds with clear RSS/memory win and no wall↑).
- Reject call-count wins that raise wall (L2 lesson).

### Step D — Scale check

Re-run `--full-scale` (or at least 200 + 1000). Wall should not super-linearly explode vs chunk count without explanation.

### Step E — Resume track (after cold is stable)

1. Cold baseline into `--keep-dir`.
2. Resume unchanged — expect `files_processed≈0`, wall ≪ cold.
3. Touch N% of files (scripted mtime/content) — measure incremental wall / file.
4. Plan L5 (smart-diff / reindex) only after cold wall is acceptable.

---

## 6. Improvement backlog (use full-flow to rank)

Priorities below are **hypotheses** until measured under `--mode full`.

| ID | Idea | Measure with | Ship if |
|----|------|--------------|---------|
| **F1** | Cross-file **append** buffer (write batch; L2-style but append-only new ids) | full 200/1000 + soak 50k | wall↓, RSS not O(corpus) |
| **F2** | Parse∥store balance (queue depth, worker count) | full; phase split store vs parse | wall↓ when store was starved or parse oversubscribed |
| **F3** | Fake-embed CPU (dims=1024 hash vectors) vs I/O split | dims 32 vs 1024 full | Informs whether product Voyage will hide DB (it will) — still optimize DB for free-embed and for defer CPU |
| **F4** | `db_batch_size` / page-size ladder on **full** path | full 200 | wall↓ at some size; no fragment storm |
| ~~**F5**~~ | ~~Optimize thr under product batch cadence~~ | Synthetic thr=50 slight; OpenJDK thr=**100**; product default **100** | wall gate |
| **F6** | Change-detect / hash cost on large trees | full root + resume | Only if discover/change_detect share is material |
| **F7** | Timeout / large-file path | full with large synthetic files | No silent skip regression; wall honesty |
| **L5** | Reindex smart-diff | resume + force-reindex | Out of cold-start; later |
| **L6** | Fixed embedding schema | full dims switch mid-run | Product safety, not clean cold wall |

**Done (soak + full revalidated):** L1 defer, File-id, P3 file batch, Write append, L3-empty, F1, F5 (synth thr=50; product thr=100).

---

## 7. Decision rules (non-negotiable)

1. **Gate = `wall_s` on full cold synthetic (medium size).**  
2. Fake embeds only for ranking optimizations.  
3. Real Voyage only for manual UX/quality — never for “is DB faster.”  
4. Reject: fewer Lance calls with higher wall or peak RSS ∝ N.  
5. Soak remains the **microbenchmark** for write internals; full is the **system gate**.  
6. Resume work starts only after cold full-flow is understood and stable.

---

## 8. Baseline established (2026-07-17)

See **`operations/indexing_timing_ledger.md` → Full-flow baseline**.

| Gate | wall_s | Notes |
|------|--------|-------|
| Medium (200 modules, dims=1024) | **~17–23** | store ~47–50%; residual empty ~2.6s |
| Large (1000 modules, ~49k chunks) | **~79.5** | store **57%**; residual empty **15%**; Lance write **7%** |
| Soak 50k (dims=32, write-only) | **~14.1** | reference only |

### Ranked plan from findings

| Priority | Work | Why | Success |
|----------|------|-----|---------|
| ~~**P0**~~ | ~~**L3-empty residual**~~ | **Done:** residual 11.7→0.1s @ 49k; wall 79.5→71 | residual ≪1s |
| ~~**P1**~~ | ~~**F1 cross-file append batch**~~ | **Done:** wall ~70→**61s** @ 49k; batches 1016→254; RSS 429→375 MB; `defer_flush_chunks=1000` | full wall↓ |
| ~~**P2**~~ | ~~**F5 thr under full cadence**~~ | Synthetic thr=50; **OpenJDK thr=100** (opt 1232→621s); product default **100** | thr ladder |
| **P3** | **F2 parse pipeline** | store still dominates; parse secondary | next if wall still needed |
| **Later** | Resume / L5 | scaffold ready | after cold stable |

### L3-empty (done)

- Residual uses SQL missing-clause with **select id only** (no full labeled embedding materialization).  
- `insert_embeddings_batch` / deferred write **refuse zero vectors as complete labels** so zeros stay residual candidates without an O(N)×dims scan.  
- **Legacy DBs** that already store labeled all-zero embeddings are **not** auto-repaired by residual; use **force-reindex** (or rebuild) to clear them. Search still treats zeros as invalid via `_has_valid_embedding`.

### F1 (done)

- Config: `indexing.defer_flush_chunks` (default **1000**; `1` = per-file L1).  
- Cross-file buffer of `(chunk, vector)` for brand-new deferred files; flush via Lance **append**.  
- Per-store-batch remainder flush (directory-wide single flush cut calls but **raised wall** — rejected).  
- Prefer large append batches on flush (`prefer_large_append`).  

### F5 (done) + OpenJDK scale check

- Harness: `--full-optimize-ladder` + config isolation (`model_construct`, no global config)  
- Synthetic 1000: thr=**50** slightly better wall  
- **OpenJDK** (~71k files, ~1.9M chunks): thr=**100** better wall (**4720s** vs **4959s**); thr=50 spent **1232s** optimizing vs **621s** at thr=100  
- **Product default remains 100** for large-tree health  
- Do not drop thr to 50 solely from synthetic evidence  

### Next (P3 F2)

Parse share is secondary after F1/L3; profile before investing.

### Follow-up (correctness, not cold wall)

- **Smart-diff multiplicity:** chunk IDs now include line span so cold append keeps two identical bodies; reindex smart-diff still keys on **content only** and may collapse multiplicity. Fix or contract-test before claiming reindex parity for copy-pasted blocks.

---

## 9. References

- `scripts/profile_index.py` — soak / full / full-scale / resume scaffolding  
- `chunkhound/core/diagnostics/index_profile.py`  
- `chunkhound/services/indexing_coordinator.py` — phases discover / change_detect / parse_store / store  
- `operations/indexing_performance_plan.md` — L1–L4 history, cold-start scope  
- `tests/fixtures/fake_providers.py` — `FakeEmbeddingProvider`
