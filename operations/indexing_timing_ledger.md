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

---

## Next measurement targets

| ID | Hypothesis | Protocol |
|----|------------|----------|
| **Write batch (optional)** | Cross-file chunk buffer + append after Write append | only if wall still needs cut |
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
| (session) | Write append | 500k thr=100 append | **138.3** | 128.1 | 0.14 | 1.0 | 49.7 | **76.7** | **24.4** | 21.7 | 807 | 0 | **current baseline** |

*Add new rows below as work continues.*
