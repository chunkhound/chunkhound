# Indexing Flow Performance Plan

**Branch:** `lance-index-perf` (from `lance-upgrade-0.34` / LanceDB 0.34)  
**Scope:** Bulk **end-to-end indexing flow** for large codebases (DB is the usual cost center under free embeds, not the goal)  
**Non-goals:** Voyage/API embedding latency (always use **FakeEmbeddingProvider** for tests), search quality, MCP UX

---

## 0. North star (corrected)

**What matters:** overall indexing flow wall time (`wall_s` / full product index path).

DB counters (`merge_insert_*`, optimize) are **diagnostics** — they explain *where* time goes. Reducing DB work while shifting cost elsewhere so **total wall increases is a regression**, even if call counts look better.

We use fake embeds so embed network does not hide the rest of the flow. Under free embed:

1. Prefer fewer **expensive** Lance operations *only when wall falls*  
2. Memory stays **O(batch/page)**, never O(corpus)  
3. Do not optimize a sub-metric at the expense of the full path  

**Primary bottleneck hypothesis (validated for L1):** LanceDB serial `merge_insert` + residual **read-modify-write** re-fetching full rows. **L1 (defer single write)** cut wall ~4× at 500k. **L2 (cross-file flush)** cut merge calls ~10× but **increased wall ~18%** — rejected as a wall win (see §5).

---

## 1. Goals and constraints

| Goal | Constraint |
|------|------------|
| **Lower end-to-end index wall** | Lance OSS: **one serial DB executor** |
| Scale to large codebases (10⁵–10⁶ chunks) | No full-table loads / full `to_pandas` on hot path |
| Keep lance-take2 streaming invariants | Ordered keyset missing-embed pages |
| Profile only with fake embeddings | Embed must not dominate wall time |

**Success metrics (fake embed) — ranked:**

| Rank | Metric | Role |
|------|--------|------|
| **1 (gate)** | `wall_s` / TOTAL phase wall | Ship / keep only if this improves (or holds with clear memory win) |
| **2** | `peak_rss_mb` | Ceiling; must stay O(batch) |
| **3 (diag)** | `merge_insert_s`, calls, rows | Explain DB share of wall |
| **3 (diag)** | `optimize_s` / calls | Fragment tax |
| **3 (diag)** | phase split (`seed_insert`, residual, etc.) | Where non-DB time moved |

A change that improves rank-3 while hurting rank-1 is **not done**.

---

## 2. Current pipeline and DB cost model

```
Discover → change-detect → parse (CPU)
  → per-file serial DB:
        upsert file
        smart-diff load existing chunks (reindex only)
        WRITE A: insert_chunks_batch (merge_insert)     [classic]
        or WRITE A': insert_chunks_with_embeddings      [defer_chunk_write, new files]
  → residual generate_missing (if any):
        keyset page missing rows
        fake embed_batch
        WRITE B: insert_embeddings_batch (merge_insert full rows)
        maybe optimize
```

### Cost per new chunk

| Path | Lance merge_inserts | Notes |
|------|---------------------|--------|
| **Classic** | **2** (text then vector RMW) | Residual re-read was O(table) pain |
| **`defer_chunk_write`** | **1** (text+vector) | New files only; reindex still classic |

### Why residual was the large-N killer

`insert_embeddings_batch` used to:

1. Receive only `(chunk_id, vector)`  
2. **Re-read full rows** from Lance (`id IN (...)`) for every batch  
3. `merge_insert` full schema back  

As the table grows, (2) dominates even with free embeds. Missing-embed **pages already had full row data** and discarded it.

---

## 3. Implemented optimizations

### 3.1 Deferred single write (new files)

- Config: `indexing.defer_chunk_write` / `CHUNKHOUND_INDEXING__DEFER_CHUNK_WRITE`  
- Embed in memory (batch-bounded) → one `insert_chunks_with_embeddings_batch`  
- Reindex / existing-chunk smart-diff: still two-write (preserve embeddings)  

### 3.2 Residual path: no re-read when page fields present

- Missing-embed pages carry `content`, `file_id`, lines, types, metadata, `created_time`  
- `EmbeddingService` passes `row_fields` into embed batches  
- Lance `insert_embeddings_batch` builds merge rows from payload; re-read only if fields incomplete  

### 3.3 Optimize coalesce

- 5s cooldown after optimize to avoid fragment-threshold storms  

### 3.4 Profiling harness (fake only)

```powershell
# Single soak (DB write path only — no parse)
uv run python scripts/profile_index.py --mode soak --chunks 10000 --page-size 256
uv run python scripts/profile_index.py --mode soak --chunks 10000 --page-size 256 --defer-write

# Scale ladder 2k/10k/25k/50k classic + defer
uv run python scripts/profile_index.py --scale --page-size 256

# Full product path with fake embeds (discover+parse+store+residual)
# Prefer synthetic corpus for A/B; dims default 1024 for vector payload cost.
uv run python scripts/profile_index.py --mode full --corpus synthetic `
  --files 200 --funcs-per-file 20 --defer-write
uv run python scripts/profile_index.py --full-scale --defer-write

# Real tree index with fake (no Voyage)
uv run python scripts/profile_index.py --mode full --root . --defer-write
```

**System gate for product wall** = `--mode full` (see `operations/full_flow_tuning_plan.md`).  
Soak remains the write-path microbench. Never use Voyage for these measurements.

---

## 4. Measured results (fake, LanceDB 0.34, Windows)

**Living ledger (append after every soak):** [`operations/indexing_timing_ledger.md`](indexing_timing_ledger.md)

### 2k chunks, page 100

| Path | merge_insert calls | rows | TOTAL s | peak RSS |
|------|-------------------|------|---------|----------|
| classic | 40 | 4000 | ~3.9 | ~262 MB |
| defer | 20 | 2000 | ~2.3 | ~225 MB |

### 10k chunks, page 256 (after residual no-re-read)

| Path | seed | stream_embed | TOTAL s | mi_s | mi_calls | rows | peak RSS |
|------|------|--------------|--------|------|----------|------|----------|
| classic | 2.5s | **4.4s** (was ~6.2 before no-re-read) | ~7.9 | 1.3 | 140 | 20k | ~308 MB |
| defer | 4.4s | — | **~5.4** | 0.9 | 100 | 10k | ~243 MB |

### 50k chunks, page 512, **defer only**

| seed | TOTAL | mi_s | mi_calls | rows | optimize | peak RSS |
|------|-------|------|----------|------|----------|----------|
| 25.8s | ~27.3s | 5.2s | **500** | 50k | 5 calls / 1.4s | ~316 MB |

### 500k chunks, page 512 — **before vs after (headline)**

Fake embed only. Same machine. Synthetic seed (5000 files × 100 chunks).

| | **BEFORE** classic two-write | **AFTER** `defer_chunk_write` | Improvement |
|--|------------------------------|-------------------------------|-------------|
| **Wall clock** | **1220.6 s** (~20.3 min) | **294.8 s** (~4.9 min) | **4.1× faster** |
| TOTAL phases | 1216.6 s | 292.2 s | −76% |
| seed_insert | 250.1 s | 285.7 s | (includes embed+write) |
| stream_embed | **960.3 s** | — (not needed) | **eliminated** |
| merge_insert rows | **1_000_000** | **500_000** | **half** |
| merge_insert calls | 6953 | 5000 | −28% |
| merge_insert wall | 104.7 s | 69.5 s | −34% |
| optimize calls / wall | 82 / 72.4 s | 48 / 44.6 s | −41% time |
| peak RSS | **2080 MB** | **1019 MB** | **~half memory** |
| remaining missing | 0 | 0 | both correct |

**Throughput:** classic ~410 chunks/s → defer (pre-L2) **~1700 chunks/s** (fake embed).

**Interpretation at large-codebase scale:**

1. Residual **stream_embed** is ~79% of classic wall (960/1217) — the real large-N DB tax under free embed.  
2. Defer removes that phase for cold new-file ingest → **wall** wins (L1).  
3. Memory halves (no second full-table RMW pipeline).  
4. Pre-L2 still had **5000** merge_inserts (one per file). L2 tried multi-file batching — see §5 (wall regression).

JSON artifacts (local, not committed): `profile_500k_classic.json`, `profile_500k_defer.json`.

**Takeaways (still true):**

1. Under free embed, residual two-write DB path dominates classic wall — fix that first.  
2. **L1 defer** halves write amplification and **wins overall wall**.  
3. Residual **no-re-read** helps classic residual; avoiding residual via defer is the step-change for cold new files.  
4. Call-count and optimize are useful diagnostics; they are **not** success criteria alone.  
5. Default `defer_chunk_write=true` is correct for cold new-file bulk index.  
6. Do not load the corpus into RAM to go “faster.”

### 500k — L2 vs L2-fix (same harness, defer on)

| | Pre-L2 (historical) | L2 (`flush=1000`, 2 runs) | **L2-fix** (per-file, this run) |
|--|---------------------|---------------------------|----------------------------------|
| **wall_s** | ~295 | ~349 / ~350 | **~336** |
| seed_insert | ~286 | ~339 / ~337 | **~325** |
| merge_insert calls | 5000 | 500 | **5000** |
| merge_insert_s | ~69.5 | ~8.0–8.2 | **~86** |
| optimize calls / s | 48 / ~45 | 10 / ~12.5 | **54 / ~57** |
| peak RSS MB | ~1019 | ~1234–1236 | **~1007** |
| remaining_missing | 0 | 0 | **0** |
| flush_policy | per_file | cross_file | **per_file** |

**Conclusion:** L2 improved DB diagnostics but **raised wall ~18%**. L2-fix restores per-file flush: **wall better than L2** (~350→~336), **RSS back to ~1 GB**, correctness OK. Historical ~295 is not fully recovered on this run (likely machine noise + time since that baseline); gate is “beat L2 wall,” which holds. See re-evaluation below.

### 500k defer — Instr sub-phases (after L2-fix, per-file)

Same flush policy as L2-fix (per-file); **different run** — use for *share of
seed*, not as an absolute wall gate vs the L2-fix ~336s figure.

| Phase | s | % of seed |
|-------|---|-----------|
| **seed_chunk_write** | **154.3** | **51%** |
| └ db.merge_insert_s | 77.6 | (inside write) |
| └ db.optimize_s | 49.9 | (inside write when threshold hits) |
| **seed_file_insert** | **91.3** | **30%** |
| **seed_embed** (fake) | **55.1** | **18%** |
| seed_chunk_build | 1.2 | ~0% |
| seed_insert (sum) | 302.6 | 100% |
| **wall_s** | **313.0** | remaining_missing=0, peak RSS ~1009 MB |

**Implications (wall-first):**

1. **Chunk write path is still the largest single slice** (~half of seed), of which merge+optimize ≈ 128s. **L4** (optimize policy) is the next DB knob that can move wall without L2-style batching.  
2. **`insert_file` is ~30% of seed** (5000 serial file rows) — not free; batching/cheaper file upserts would matter more than cross-file *chunk* buffer.  
3. Fake embed is ~18% — real Voyage will dominate this slice; do not over-optimize it under fake.  
4. Chunk model construction is negligible.

### L4 — optimize fragment threshold A/B (wall gate)

Harness: `--optimize-ladder` / `--optimize-threshold` (soak default now **100** = product).

**50k defer ladder:**

| thr | wall_s | opt_n / opt_s | mi_s | write_s | peak MB |
|-----|--------|---------------|------|---------|---------|
| 50 | 28.6 | 5 / 1.4 | 5.7 | 9.9 | 316 |
| **100** | **26.4** | 4 / 1.2 | 5.9 | 10.1 | 324 |
| 200 | 32.9 | 2 / 0.8 | 7.2 | 12.4 | 350 |
| 500 | 59.5 | 1 / 0.9 | 12.0 | 23.3 | 459 |
| 10000 | 57.8 | 0 / 0 | 11.8 | 22.0 | 452 |

**500k thr=50 vs thr=100 (back-to-back):**

| thr | wall_s | opt_n / opt_s | mi_s | file_s | write_s | remaining |
|-----|--------|---------------|------|--------|---------|-----------|
| 50 | 316.2 | 51 / 51.5 | 78.2 | 91.7 | 156.6 | 0 |
| **100** | **310.6** | 49 / 50.1 | 76.5 | 90.9 | 153.3 | 0 |

**Conclusion:** Raising threshold to “save optimize time” **increases** wall — fragment drag slows merge_insert *and* `insert_file`. Product default **100** is near the sweet spot; thr=50 (old soak hard-code) is no better at 500k. **No product default change.** Next wall lever is **file insert batching**, not fewer optimizes.

---

## 5. Re-evaluation: L2 and remaining ideas (wall-first)

### Decision rule

```
ship / keep change  ⇔  wall_s improves (or holds) AND peak_rss acceptable
                       DB counters only explain why
```

### L1 — defer single write — **KEEP (validated wall win)**

- Wall −76% at 500k classic → defer; peak RSS ~half.  
- Correct default for cold new-file path.

### L2 — cross-file deferred buffer — **REVERTED (L2-fix done)**

| Claim | L2 result | After L2-fix |
|-------|-----------|----------------|
| Fewer merge_inserts | 5000 → 500 | Back to 5000 (per-file) |
| Lower merge_insert_s / optimize_s | Yes | Back to ~86s / ~57s |
| Lower end-to-end wall | **False** (~350s) | **~336s** (beats L2; gate) |
| Lower peak RSS | **False** (~1.2 GB) | **~1.0 GB** (recovered) |

**Why L2 was wrong as shipped:** optimized call count / pure DB write time; **wall** rose. Larger batches + buffer churn + RSS did not repay merge fixed-cost savings under free embed.

**What we did (L2-fix):**

- Product: `_store_parsed_results` uses per-file `_embed_and_store_new_chunks` again (no cross-file buffer).  
- Soak: `flush_policy=per_file` (no cross-file buffer).  
- Tests: L2 call-count assertion removed; multi-file defer contract = no missing embeds.

**If revisiting batching later:** sub-phase timers + flush-size ladder with **wall as gate**. Only ship a size that beats current per-file wall.

### Remaining ideas — re-ranked by expected **end-to-end wall** impact

| Priority | ID | Idea | Wall impact under current defaults | Verdict |
|----------|-----|------|------------------------------------|---------|
| **Done** | **L2-fix** | Revert cross-file buffer; per-file L1 flush | Beat L2 wall (~350→~336); RSS ~1.2→~1.0 GB | **Done** |
| **Done** | **Instr** | Soak sub-phases: `seed_file_insert` / `seed_chunk_build` / `seed_embed` / `seed_chunk_write` (+ existing `db.merge_insert_s` / `optimize_s`) | Attribute wall inside seed | **Done** |
| **P2** | **L1-hold** | Keep defer default; no clever write paths without wall gate | Protects the only proven large win (vs classic) | Hold |
| **Done** | **L4** | Optimize threshold A/B (50/100/200/500/10k) | **Keep product default 100**; thr≥200 **raises** wall (fragment drag on merge + file insert) | **Done** (measure, no product change) |
| **Done** | **File-id** | Skip post-`insert_file` path search; return pre-assigned id | file 91→51s; wall 311→297 at 500k | **Done** |
| **Done** | **File batch (P3)** | `insert_files_batch` multi-file merge | file 51→**0.14s**; wall 297→**214**; peak **0.77 GB** | **Done** |
| **Done** | **Write append** | Deferred chunk write uses `add` (new ids) not merge_insert | wall 214→**138**; write 145→**77**; mi_s 85→**24** | **Done** |
| **P3** | **Write batch** | Cross-file deferred chunk buffer + append (L2-style but append) | write still ~77s / ~60% seed | Optional if wall still needs it |
| **Out of cold-start scope** | **L6** | Fixed-dim schema when dims known | One-shot footgun (O(rows) only if variable schema already full); **does not improve clean cold index wall** when empty/fixed schema | Later product safety |
| **Out of cold-start scope** | **L5** | Reindex smart-diff / hash-only | **Incremental reindex**, not cold bulk | Later reindex profile |
| **Later** | **L3** | Residual missing scan cheaper | Residual empty on cold+defer success | Deprioritize default cold bulk |
| **Later** | **L7** | Parse ∥ pipeline / free-threading | After DB path wall-stable | Profile first |
| **Park** | **L2-retry** | Cross-file batch only if flush ladder shows wall↓ | Only after Instr baseline | Parked |
| **Park** | **Read-backend** | After Lance **write** path is done: if DuckDB is faster for **search/read**, optionally rebuild final Lance tables into DuckDB (similar to existing Duck→Duck compaction) | Search latency, not index wall | **Later — only when evaluating Lance for searches** |

**Cold-start focus:** large-codebase **first index** wall (new files + defer). L5/L6 remain valid work later but are **out of scope** for this cold-start track.

**Do not:**

- Load more of the corpus into RAM to go “faster.”  
- Ship DB-call reductions that raise wall or peak RSS without a clear product reason.  
- Prioritize residual-scan work (L3) for the default cold+defer path where residual is already empty.

---

## 6. Phase plan (revised)

### Done (wall-validated)

- [x] Profile harness + DB counters + RSS (`scripts/profile_index.py`, `IndexProfile`)  
- [x] Deferred single write for **new** files (**L1** — wall win)  
- [x] Residual embed **no re-read** when page fields present  
- [x] Optimize cooldown  
- [x] Fake-only scale soaks (2k / 10k / 500k classic vs defer)  
- [x] Default `defer_chunk_write=true` (**L1**)  

### Reverted (wall regression)

- [x] Cross-file deferred buffer (**L2**) — shipped then **L2-fix** removed (calls↓ wall↑)  
- [x] **L2-fix:** per-file deferred flush restored; 500k wall ~336s (beats L2 ~350s); RSS ~1 GB  

### Next (wall-first)

- [x] **Instr:** soak sub-phases `seed_file_insert` / `seed_chunk_build` / `seed_embed` / `seed_chunk_write`  
- [x] **L4 measure:** optimize threshold A/B — product **100** wins; soak default aligned to 100; do not raise  
- [x] **File-id:** skip post-`insert_file` path search (500k file ~91→51s, wall ~311→297)  
- [x] **File batch (P3):** multi-file `insert_files_batch` (500k file ~51→0.14s, wall ~297→214)  
- [x] **Write append:** deferred `insert_chunks_with_embeddings` uses append (500k wall ~214→138)  
- [ ] **Write batch (optional):** cross-file buffer + append if more wall needed  

### Out of cold-start scope (address later)

- [ ] **L6:** fixed-size embedding schema when dims known — product safety / avoid O(N) rewrite; **not** a clean cold-start wall win  
- [ ] **L5:** reindex smart-diff cost — incremental path only  

### Full-flow gate (product path, fake embeds)

- [x] **Full harness:** `--mode full` / `--full-scale` + synthetic corpus + store phase split  
- [x] **Baseline full-flow (2026-07-17):** 200 wall ~17–23s; 1000 (~49k ch) wall **~79.5s**; ledger updated  
- [x] **Attribute:** **store 57%** (Lance write only **7%**); **empty residual 15%**; parse ~9%  
- [x] **P0 L3-empty residual** — missing-clause id-only; residual 11.7→**0.1s** @ 49k; wall 79.5→**71**  
- [ ] **P1 F1** cross-file **append** batch (not L2 merge) — store still ~84% wall post-L3  
- [ ] **P2 F5** thr ladder under full cadence after F1  
- [ ] **P3 F2** parse only if still material  
- See **`operations/full_flow_tuning_plan.md`** + ledger full-flow section.

### Later (product path / real profiles)

- [ ] **L3:** residual scan only if residual path still appears on measured product runs  
- [ ] Pipeline: parse ∥ embed ∥ DB with bounded queues (one DB writer) — only if wall profile shows idle DB waiting on parse  
- [ ] Free-threaded parse only after above  
- [ ] **Resume track:** `--scenario resume` after cold full-flow is stable  

### Parked

- [ ] Cross-file batching (**L2-retry**) until a flush size beats per-file **wall**  
- [x] **Read-backend util:** Lance→DuckDB materialization + config activation  
  (`chunkhound/utils/lance_to_duckdb.py`, `scripts/convert_lancedb_to_duckdb.py --activate`).  
  Use after cold Lance index when comparing/using DuckDB for search — not on the write hot path.

---

## 7. Testing policy

| Allowed | Forbidden for perf claims |
|---------|---------------------------|
| `FakeEmbeddingProvider` | Voyage / OpenAI / real network embeds |
| `scripts/profile_index.py` | Claiming “index is fast” based on Voyage-less wall alone for product UX |
| Integration tests with fake | Tests that require API keys |

Real Voyage is only for **manual end-to-end** quality, not for deciding DB optimizations.

---

## 8. References

- `scripts/profile_index.py` — soak / scale / **full** / full-scale / resume scaffolding  
- `operations/full_flow_tuning_plan.md` — full-flow methodology & improvement backlog  
- `chunkhound/core/diagnostics/index_profile.py`  
- `chunkhound/services/embedding_service.py` — `row_fields` residual path  
- `chunkhound/providers/database/lancedb_provider.py` — deferred insert + no-re-read merge  
- `chunkhound/services/indexing_coordinator.py` — `defer_chunk_write`, profile phases  
- `tests/fixtures/fake_providers.py` — `FakeEmbeddingProvider`  
- `operations/lancedb_large_index_reimplementation.md` — streaming invariants  

---

## 9. Operator knobs

After L2-fix, **`db_batch_size` is insert/fragment batch sizing only** — it does
**not** control cross-file deferred flush (defer writes one merge_insert per new
file). Tune it for classic/residual batch size and fragment pressure, not for
“fewer merge_inserts across files.”

```json
{
  "database": {
    "provider": "lancedb",
    "lancedb_optimize_fragment_threshold": 100
  },
  "indexing": {
    "defer_chunk_write": true,
    "db_batch_size": 2000
  }
}
```

L4: keep threshold **100** (do not raise to 500 “to avoid optimize”). Soak harness
defaults to 100; use `--optimize-ladder` to re-check on a machine.

```powershell
$env:CHUNKHOUND_INDEXING__DEFER_CHUNK_WRITE = "true"
uv run python scripts/profile_index.py --scale --page-size 256
uv run python scripts/profile_index.py --optimize-ladder --chunks 50000 --page-size 512
```
