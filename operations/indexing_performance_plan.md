# Indexing Flow Performance Plan

**Branch:** `lance-index-perf` (from `lance-upgrade-0.34` / LanceDB 0.34)  
**Scope:** Bulk indexing **DB path** for large codebases  
**Non-goals:** Voyage/API embedding latency (always use **FakeEmbeddingProvider** for tests), search quality, MCP UX

---

## 0. North star (updated)

We are **not** optimizing Voyage wall time. External embed IO will dominate real runs and that is fine.

We optimize what still hurts at large N **even when embed is free**:

1. How many Lance **writes** happen per chunk  
2. How expensive each write is as the table grows (re-reads, fragments, optimize)  
3. Memory staying **O(batch/page)**, never O(corpus)

**Primary bottleneck hypothesis (validated):** LanceDB serial `merge_insert` + residual **read-modify-write** that re-fetched full rows while the table grew. Fake embed makes this visible; real Voyage would hide it under network time but the DB cost still adds.

---

## 1. Goals and constraints

| Goal | Constraint |
|------|------------|
| Cut DB work per indexed chunk | Lance OSS: **one serial DB executor** |
| Scale to large codebases (10⁵–10⁶ chunks) | No full-table loads / full `to_pandas` on hot path |
| Keep lance-take2 streaming invariants | Ordered keyset missing-embed pages |
| Profile only with fake embeddings | Embed must not dominate wall time |

**Success metrics (fake embed):**

| Metric | Meaning |
|--------|---------|
| `merge_insert_calls` / `merge_insert_rows` | Write amplification |
| `merge_insert_s` | Pure DB write wall |
| `stream_embed` phase | Residual path (page + fake embed + DB write) |
| `peak_rss_mb` | Memory ceiling |
| chunks/s at 2k → 50k | Superlinear DB slowdown detection |

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
# Single soak
uv run python scripts/profile_index.py --mode soak --chunks 10000 --page-size 256
uv run python scripts/profile_index.py --mode soak --chunks 10000 --page-size 256 --defer-write

# Scale ladder 2k/10k/25k/50k classic + defer
uv run python scripts/profile_index.py --scale --page-size 256

# Full tree index with fake (no Voyage)
uv run python scripts/profile_index.py --mode index --root . --defer-write
```

Never use Voyage for these measurements.

---

## 4. Measured results (fake, LanceDB 0.34, Windows)

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

~1.8k chunks/s end-to-end; pure `merge_insert` ~10k rows/s.  
**500 merge_insert calls** = still one write per synthetic file (100 chunks) — fixed per-call overhead scales with file count, not just row count → **L2 batching across files** is the next DB win.

**Takeaways:**

1. **DB is the bottleneck** with free embeds (`stream_embed` ≫ fake compute).  
2. **Defer halves write amplification** and wins overall wall at 10k.  
3. Residual **no-re-read** cut classic `stream_embed` ~30% at 10k; still loses to defer because of second write + page scan.  
4. At 50k, **call count** (many small merge_inserts) and **optimize** matter; not just row count.  
5. For large cold indexes of **new** codebases, **default lean is `defer_chunk_write=true`** once product-ready; residual path remains for reindex / crash recovery / realtime.  
6. Peak RSS stayed ~O(batch) (~300 MB) even at 50k — memory model is OK; do not “go faster” by loading the corpus.

---

## 5. Remaining large-N bottlenecks (priority order)

| ID | Issue | Why it hurts large repos | Next step |
|----|--------|---------------------------|-----------|
| **L1** | Classic two-write still default | 2× merge_insert always | Product default / flag docs; promote defer for cold index |
| **L2** | Seed/store **many tiny merge_inserts** (per file / small db_batch) | Fixed cost × files | Batch inserts across files; raise `db_batch_size` under defer |
| **L3** | Residual missing scan (`search().where` over growing table) | Pages still scan candidates | Indexed signature / better filter; keep keyset |
| **L4** | Fragment growth + optimize | Spikes wall mid-run | Cooldown done; tune threshold vs LSM 0.34 |
| **L5** | Reindex smart-diff loads all file chunks | Large files | Hash-only skip more often; optional chunk-level signatures |
| **L6** | Schema migration full table rewrite | One-time disaster at first embed | Always create fixed-dim schema when dims known |
| **L7** | Parse/discovery | Real but secondary when DB is free-embed bottleneck | Free-threading / pipeline after L1–L4 |

**Do not:** load more of the corpus into RAM to go “faster.”

---

## 6. Phase plan (revised)

### Done

- [x] Profile harness + DB counters + RSS (`scripts/profile_index.py`, `IndexProfile`)  
- [x] Deferred single write for **new** files (flagged)  
- [x] Residual embed **no re-read** when page carries row fields  
- [x] Optimize cooldown  
- [x] Fake-only scale soaks (2k / 10k)  

### Next (DB scale)

- [ ] Run `--scale` ladder to 50k/100k; record in this doc if superlinear  
- [ ] Multi-file **batched** deferred insert (reduce per-file merge_insert count)  
- [ ] Prefer fixed-size embedding schema at first connect when fake/real dims known  
- [ ] Consider default `defer_chunk_write=true` after more soak confidence  
- [ ] Residual candidate scan: avoid loading `embedding` column in id candidate pass where possible  

### Later

- [ ] Pipeline: parse ∥ embed ∥ DB with bounded queues (still one DB writer)  
- [ ] Optional free-threaded Python for parse only (not multi-writer DB)  
- [ ] Reindex path smart-diff cost  

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

- `scripts/profile_index.py` — soak / scale / index modes  
- `chunkhound/core/diagnostics/index_profile.py`  
- `chunkhound/services/embedding_service.py` — `row_fields` residual path  
- `chunkhound/providers/database/lancedb_provider.py` — deferred insert + no-re-read merge  
- `chunkhound/services/indexing_coordinator.py` — `defer_chunk_write`  
- `tests/fixtures/fake_providers.py` — `FakeEmbeddingProvider`  
- `operations/lancedb_large_index_reimplementation.md` — streaming invariants  

---

## 9. Operator knobs

```json
{
  "database": {
    "provider": "lancedb",
    "lancedb_optimize_fragment_threshold": 50
  },
  "indexing": {
    "defer_chunk_write": true,
    "db_batch_size": 2000
  }
}
```

```powershell
$env:CHUNKHOUND_INDEXING__DEFER_CHUNK_WRITE = "true"
uv run python scripts/profile_index.py --scale --page-size 256
```
