# Indexing timing ledger

**Purpose:** Single place to track soak/profile wall times across this branch.  
**Gate metric:** `wall_s` (end-to-end). DB counters are diagnostics only.  
**Harness:** `scripts/profile_index.py` · Fake embed · LanceDB 0.34 · Windows  
**Shape (unless noted):** `files_n = chunks // 100` (e.g. 500k → 5000 files × 100 chunks)

Update this file after every meaningful soak. Prefer new rows over editing old ones.

---

## How to add a run

```powershell
uv run python scripts/profile_index.py --mode soak --chunks N --page-size P --defer-write --optimize-threshold 100 --json
```

Record: date, commit/label, thr, flush policy, wall + sub-phases if present, remaining_missing, peak RSS.

---

## Headline 500k timeline (defer unless noted)

Same machine family; run-to-run noise is real (~±5–10% wall). Use for direction, not micro-claims.

| When | Label | thr | flush | wall_s | seed | mi_calls | mi_s | opt_n/s | peak MB | miss | Notes |
|------|-------|-----|-------|--------|------|----------|------|---------|---------|------|-------|
| Early | **classic** two-write | 50* | n/a | **1220.6** | 250.1 + stream **960.3** | 6953 | 104.7 | 82 / 72.4 | 2080 | 0 | Residual dominates |
| Early | **L1 defer** (pre-L2) | 50* | per_file | **294.8** | 285.7 | 5000 | 69.5 | 48 / 44.6 | 1019 | 0 | **4.1× vs classic** |
| L2 | cross-file flush=1000 r1 | 50* | cross | **~349** | ~339 | 500 | ~8.2 | 10 / ~12 | ~1236 | 0 | calls↓ wall↑ |
| L2 | cross-file flush=1000 r2 | 50* | cross | **350.1** | 337.4 | 500 | 8.0 | 10 / 12.5 | 1234 | 0 | confirmed |
| L2-fix | per-file restore | 50* | per_file | **336.0** | 325.3 | 5000 | 86.3 | 54 / 56.7 | 1007 | 0 | beat L2 wall |
| Instr | sub-phases first | 50* | per_file | **313.0** | 302.6 | 5000 | 77.6 | 50 / 49.9 | 1009 | 0 | see sub-phase table |
| L4 | thr=50 retest | **50** | per_file | **316.2** | 306.1 | 5000 | 78.2 | 51 / 51.5 | 1012 | 0 | old soak default |
| L4 | thr=100 retest | **100** | per_file | **310.6** | 300.6 | 5000 | 76.5 | 49 / 50.1 | 1025 | 0 | product thr |
| **File-id** | skip post-insert path search | **100** | per_file | **296.6** | 285.8 | 5000 | 87.0 | 46 / 55.3 | 1046 | 0 | prior |
| **P3 File batch** | insert_files_batch | **100** | per_file+batch_files | **213.5** | 203.4 | 5000 | 85.5 | 35 / 28.6 | 773 | 0 | prior |
| **Write append** | deferred chunk add() | **100** | per_file+batch_files | **138.3** | 128.1 | 5000 | **24.4** | 21 / 21.7 | 807 | 0 | **current best** |

\* Early soaks hard-coded thr=50 in harness; product default was already 100.

### Best known 500k baseline (current product-aligned)

| Metric | Value |
|--------|-------|
| **wall_s** | **~138** (append deferred chunks + P3 files, thr=100) |
| seed_insert | ~128 |
| seed_file_insert | **~0.14** |
| seed_chunk_write | **~77** (was ~145 with merge_insert) |
| merge write s | **~24** (was ~85) |
| peak RSS | **~0.81 GB** |
| remaining_missing | 0 |
| flush | per_file (L1) |
| optimize_threshold | **100** |

---

## 500k sub-phase share (Instr, thr≈50, per-file)

| Phase | s | % of seed |
|-------|---|-----------|
| seed_chunk_write | 154.3 | **51%** |
| └ merge_insert_s | 77.6 | (inside write) |
| └ optimize_s | 49.9 | (inside write) |
| seed_file_insert | 91.3 | **30%** |
| seed_embed (fake) | 55.1 | **18%** |
| seed_chunk_build | 1.2 | ~0% |
| seed_insert | 302.6 | 100% |
| wall_s | 313.0 | |

**Implication:** next wall levers after L4 = **file insert (~30%)** then write path; fake embed ~18% will be Voyage-dominated in prod.

---

## L4 optimize threshold ladder (50k, defer, page 512)

| thr | wall_s | opt_n / opt_s | mi_s | write_s | peak MB |
|-----|--------|---------------|------|---------|---------|
| 50 | 28.6 | 5 / 1.4 | 5.7 | 9.9 | 316 |
| **100** | **26.4** | 4 / 1.2 | 5.9 | 10.1 | 324 |
| 200 | 32.9 | 2 / 0.8 | 7.2 | 12.4 | 350 |
| 500 | 59.5 | 1 / 0.9 | 12.0 | 23.3 | 459 |
| 10000 | 57.8 | 0 / 0 | 11.8 | 22.0 | 452 |

**Takeaway:** thr≥200 **raises** wall (fragment drag). Keep product default **100**.

---

## Smaller soaks (historical / sanity)

### 2k, page 100

| Path | mi_calls | rows | TOTAL s | peak RSS |
|------|----------|------|---------|----------|
| classic | 40 | 4k | ~3.9 | ~262 MB |
| defer | 20 | 2k | ~2.3 | ~225 MB |

### 10k, page 256 (post residual no-re-read)

| Path | seed | stream | TOTAL | mi_s | mi_calls | peak |
|------|------|--------|-------|------|----------|------|
| classic | 2.5 | **4.4** | ~7.9 | 1.3 | 140 | ~308 |
| defer | 4.4 | — | **~5.4** | 0.9 | 100 | ~243 |

### 10k defer after L2-fix / Instr (per-file, thr varies)

| Label | wall_s | seed | mi_calls | mi_s | opt | peak | miss |
|-------|--------|------|----------|------|-----|------|------|
| L2-fix sanity | ~7.1 | 4.1 | 100 | 1.0 | 1 / 0.1 | 244 | 0 |
| Instr thr default | ~7.3 | 4.1 | 100 | 1.0 | 1 / 0.1 | 243 | 0 |

### 50k, page 512, defer (pre-L4 historical)

| seed | TOTAL | mi_s | mi_calls | opt | peak |
|------|-------|------|----------|-----|------|
| 25.8 | ~27.3 | 5.2 | 500 | 5 / 1.4 | ~316 |

---

## Decision log (tied to timings)

| Decision | Evidence | Outcome |
|----------|----------|---------|
| **L1** defer_chunk_write default true | classic 1221s → defer 295s | **Keep** |
| **L2** cross-file buffer | wall 295 → 350 | **Revert** |
| **L2-fix** per-file | wall 350 → 336, RSS 1.2→1.0 GB | **Keep** |
| **Instr** sub-phases | write 51% / file 30% / embed 18% | **Keep**; drives priority |
| **L4** raise thr to cut optimize | thr 200+ doubles wall at 50k; 500k thr 50≈100 | **No product change**; thr=100 |
| **File-id** skip path lookup after insert | file 91→51s; wall 311→297 | **Keep** (product Lance path) |
| **P3 File batch** | file 51→0.14s; wall 297→214; RSS 1.0→0.77 GB | **Keep** |
| **Write append** | deferred path uses `add` not merge; wall 214→138; write 145→77 | **Keep** |
| **Full-flow baseline** | store 57% wall; residual empty 15%; Lance write only 7% @ 49k | Led to L3-empty |
| **L3-empty residual** | missing-clause id-only; no labeled vector scan; refuse zero labels | residual 11.7→0.1s; wall 79.5→71 @ 49k; **Keep** |

---

## Full-flow baseline (2026-07-17) — post L1/P3/append/etc.

**Harness:** `scripts/profile_index.py --mode full` · fake embed · dims=**1024** · thr=100 · defer · Windows  
**Corpus:** synthetic Python (`--funcs-per-file 20`) · cold · artifacts under `.bench-full-baseline/`  
**Note:** All prior soak write wins are **already in place**. This is the new **product-path** baseline.

### Medium gate (standalone, files=200 → 208 paths incl. `__init__`)

| Metric | dims=1024 | dims=32 A/B |
|--------|-----------|-------------|
| **wall_s** | **22.6** | **14.9** |
| process_directory | 14.3 | 11.5 |
| parse_store | 13.0 | 10.3 |
| **store** | **10.6** (47% wall) | **8.0** |
| store_optimize | 0.17 | 0.37 |
| generate_missing (gen=0) | **2.63** | **0.12** |
| change_detect | 0.67 | 0.66 |
| discover | 0.60 | 0.55 |
| mi_calls / mi_s | 208 / 1.30 | 208 / 1.32 |
| chunks / ch/s | 9808 / 434 | 9808 / 656 |
| peak RSS MB | 270 | 273 |
| remaining_missing | 0 | 0 |

**dims A/B takeaway:** Lance write time (~mi_s) **unchanged**; wall drop is mostly residual scan + some store overhead when vectors are narrower. Fake embed at 1024 is **not** the whole store cost (store still ~8s at dims=32).

### Full-scale ladder (cold, defer, dims=1024, page=128)

| files* | chunks | wall_s | parse_store | store | residual | mi_s | opt_n/s | peak MB | ch/s |
|--------|--------|--------|-------------|-------|----------|------|---------|---------|------|
| 54 | 2454 | **8.7** | 3.9 | 2.0 | 0.66 | 0.27 | 0 / 0 | 235 | 283 |
| 208 | 9808 | **16.8** | 11.1 | 8.7 | 2.27 | 1.16 | 2 / 0.32 | 276 | 584 |
| 516 | 24516 | **38.1** | 24.5 | 21.0 | 5.74 | 2.69 | 5 / 1.32 | 335 | 643 |
| 1016 | 49016 | **79.5** | 52.6 | **45.1** | **11.7** | 5.79 | 10 / 4.74 | 445 | 617 |

\*file count includes package `__init__.py` (e.g. 200 modules → 208 paths).

### Phase share @ ~49k chunks (1000 modules)

| Phase | s | % wall |
|-------|---|--------|
| **store** (serial product store loop) | 45.1 | **57%** |
| └ of which Lance write (mi_s / append) | 5.8 | **7%** |
| └ of which store_optimize | 5.4 | **7%** |
| └ remainder (defer fake embed + row build + file/chunk orchestration) | ~34 | **~43%** |
| generate_missing **empty residual** | 11.7 | **15%** |
| parse (approx parse_store − store) | ~7.5 | **~9%** |
| change_detect | 2.8 | 4% |
| discover | 0.8 | 1% |

### Soak write reference (same day, dims=**32**, not 1024)

| chunks | wall_s | seed_embed | seed_write | mi_s | ch/s | miss |
|--------|--------|------------|------------|------|------|------|
| 50k | **14.1** | 4.5 | 5.8 | 2.6 | 3542 | 0 |

Full ~49k chunks @ 617 ch/s vs soak 50k @ 3542 ch/s → product path ~**5.7×** slower per chunk than pure DB soak (parse + per-file product store + residual + wider vectors).

### Findings → next work (ranked)

1. ~~**Empty residual is a first-class full-flow tax**~~ → **L3-empty done (2026-07-17)**  
2. **Store dominates (~57% pre-L3; still ~84% of wall post-L3 @ 49k)** — Lance append only ~10% of wall; next is **F1 cross-file append batch** (per-file product overhead).  
3. **Parse is secondary** — F2 only after F1.  
4. **Optimize mid-store** — re-check thr under product cadence after F1.  
5. **change_detect** — not top priority yet.

### L3-empty residual (2026-07-17) — measured

Root cause: residual loaded **all labeled rows with full embedding vectors** (O(N)×dims) to recover rare zero placeholders.  
Fix: missing-clause **id-only** query; refuse to label zero vectors on insert.

| Size | residual before | residual after | wall before | wall after |
|------|----------------:|---------------:|------------:|-----------:|
| 200 modules (~9.8k ch) | **2.63s** | **0.022s** | 22.6 | **18.3** |
| 1000 modules (~49k ch) | **11.7s** | **0.096s** | 79.5 | **71.0** |

remaining_missing=0 both sizes.

---

## Next measurement targets

| ID | Hypothesis | Protocol |
|----|------------|----------|
| ~~**L3-empty**~~ | ~~Empty residual scan too heavy~~ | **Done** — residual 11.7→0.1s @ 49k |
| **F1 write batch** | Cross-file append buffer cuts store wall | full 200/1000 + soak 50k; wall gate |
| **F5 thr@full** | thr=100 still best under product store cadence | full 1000 thr ladder after F1 |
| **F2 parse** | only if store still leaves parse material | full phase share recheck |
| (later) L6 / L5 | Out of cold-start scope | product safety / reindex |

---

## Run log (append-only)

| Date | Commit / label | Command knobs | wall_s | seed | file | build | embed | write | mi_s | opt_s | peak | miss | Notes |
|------|----------------|---------------|--------|------|------|-------|-------|-------|------|-------|------|------|-------|
| (session) | classic baseline | 500k classic thr50* | 1220.6 | 250.1 | — | — | — | — | 104.7 | 72.4 | 2080 | 0 | stream 960.3 |
| (session) | L1 defer | 500k defer thr50* | 294.8 | 285.7 | — | — | — | — | 69.5 | 44.6 | 1019 | 0 | |
| (session) | L2 r1/r2 | 500k flush1000 thr50* | 349–350 | 337–339 | — | — | — | — | ~8 | ~12.5 | ~1235 | 0 | |
| (session) | L2-fix | 500k per_file thr50* | 336.0 | 325.3 | — | — | — | — | 86.3 | 56.7 | 1007 | 0 | |
| (session) | Instr | 500k per_file thr50* | 313.0 | 302.6 | 91.3 | 1.2 | 55.1 | 154.3 | 77.6 | 49.9 | 1009 | 0 | first sub-phases |
| (session) | L4 thr50 | 500k thr=50 | 316.2 | 306.1 | 91.7 | 1.2 | 55.8 | 156.6 | 78.2 | 51.5 | 1012 | 0 | |
| (session) | L4 thr100 | 500k thr=100 | **310.6** | 300.6 | 90.9 | 1.1 | 54.5 | 153.3 | 76.5 | 50.1 | 1025 | 0 | prior baseline |
| (session) | File-id skip search | 500k thr=100 | **296.6** | 285.8 | **51.2** | 1.3 | 57.8 | 174.7 | 87.0 | 55.3 | 1046 | 0 | prior |
| (session) | P3 file batch | 500k thr=100 batch | **213.5** | 203.4 | **0.14** | 1.1 | 56.7 | 144.9 | 85.5 | 28.6 | 773 | 0 | prior |
| (session) | Write append | 500k thr=100 append | **138.3** | 128.1 | 0.14 | 1.0 | 49.7 | **76.7** | **24.4** | 21.7 | 807 | 0 | **current soak baseline** |
| 2026-07-17 | **full baseline** | full synth 200 fp20 dims1024 thr100 | **22.6** | — | — | — | — | store 10.6 | 1.30 | 0.38 | 270 | 0 | residual 2.6 gen=0 |
| 2026-07-17 | full dims32 A/B | full synth 200 fp20 dims32 thr100 | **14.9** | — | — | — | — | store 8.0 | 1.32 | 0.33 | 273 | 0 | residual 0.12 |
| 2026-07-17 | full-scale 50 | full synth 50 fp20 dims1024 | **8.7** | — | — | — | — | store 2.0 | 0.27 | 0 | 235 | 0 | 2.5k ch |
| 2026-07-17 | full-scale 200 | full synth 200 fp20 dims1024 | **16.8** | — | — | — | — | store 8.7 | 1.16 | 0.32 | 276 | 0 | ladder |
| 2026-07-17 | full-scale 500 | full synth 500 fp20 dims1024 | **38.1** | — | — | — | — | store 21.0 | 2.69 | 1.32 | 335 | 0 | residual 5.7 |
| 2026-07-17 | full-scale 1000 | full synth 1000 fp20 dims1024 | **79.5** | — | — | — | — | store 45.1 | 5.79 | 4.74 | 445 | 0 | residual 11.7; **full gate** |
| 2026-07-17 | soak 50k ref | soak 50k page128 defer dims32 | **14.1** | 10.5 | 0.01 | 0.11 | 4.5 | 5.8 | 2.6 | 0.33 | 286 | 0 | write ref |
| 2026-07-17 | **L3-empty** | full synth 200 dims1024 | **18.3** | — | — | — | — | store 11.3 | 1.43 | 0.41 | 269 | 0 | residual **0.022** (was 2.63) |
| 2026-07-17 | **L3-empty** | full synth 1000 dims1024 | **71.0** | — | — | — | — | store 59.6 | 7.15 | 6.04 | 422 | 0 | residual **0.096** (was 11.7); **new full gate** |
| 2026-07-17 | **F1** flush=1000 | full 1000 dims1024 | **61.3** | — | — | — | — | store 49.2 | 2.16 | 0.77 | 375 | 0 | batches **254** (was 1016) |
| 2026-07-17 | L1 flush=1 A/B | full 1000 dims1024 | **69.9** | — | — | — | — | store 57.9 | 6.85 | 5.88 | 429 | 0 | batches 1016 control |
| 2026-07-17 | F1 flush=1000 | full 200 dims1024 | **15.3** | — | — | — | — | store 8.6 | 0.45 | 0 | 262 | 0 | batches 52 |
| 2026-07-17 | L1 flush=1 A/B | full 200 dims1024 | **17.6** | — | — | — | — | store 10.8 | 1.36 | 0.40 | 271 | 0 | batches 208 |
| 2026-07-17 | F1 dir-end flush | full 1000 (rejected) | **73.0** | — | — | — | — | store 60.1 | 0.66 | 0 | 397 | 0 | batches 50; **wall↑** |

*Add new rows below as work continues.*
