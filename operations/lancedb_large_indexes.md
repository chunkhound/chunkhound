# Large indexes with LanceDB

Guidance for running ChunkHound against very large codebases when
`database.provider` is `lancedb`.

## Recommended config (`.chunkhound.json`)

```json
{
  "database": {
    "provider": "lancedb",
    "path": ".chunkhound",
    "lancedb_optimize_fragment_threshold": 100,
    "lancedb_index_type": "ivf_hnsw_sq"
  },
  "indexing": {
    "db_batch_size": 2000,
    "exclude": ["**/node_modules/**", "**/dist/**", "**/.git/**"]
  },
  "embedding": {
    "batch_size": 256
  }
}
```

| Knob | Role |
|------|------|
| `lancedb_optimize_fragment_threshold` | Min chunk fragments before optimize (0 = always). Product default **100** (OpenJDK-scale preferred thr=100 over 50; thr=50 over-optimizes). thr≥200 raises wall on soaks. Env: `CHUNKHOUND_DATABASE__LANCEDB_OPTIMIZE_FRAGMENT_THRESHOLD`. CLI: `--lancedb-optimize-fragment-threshold`. |
| `indexing.db_batch_size` | Base chunk insert batch size (coordinator splits large per-file inserts). Further reduced when LanceDB fragment count is high. Env: `CHUNKHOUND_DB_BATCH_SIZE`. |
| `embedding.batch_size` | Page size for streaming `generate_missing_embeddings` keyset walk. |

## Indexing flow

1. **Parse/store** — directory indexing stores chunks in fragment-aware sub-batches;
   after each parse-store batch the coordinator may call `should_optimize` /
   `optimize_tables` (LanceDB only). Chunk and embedding inserts also compact when
   over the fragment threshold (both layers are threshold-gated).
2. **Embeddings** — bulk embedding uses ordered keyset pagination so chunk content
   is never full-loaded into process memory. Prefer `chunkhound index` then let
   missing-embedding generation run (or realtime `embed` pass).
3. **Realtime** — file change events store first (`skip_embeddings=True`), then
   queue an `embed` mutation that calls `generate_missing_embeddings` with
   `indexing.exclude` patterns applied.

## Operational tips

- Prefer an absolute `database.path` when the CWD is not the project root.
- Keep fragment threshold near the product default (100); thr=50 can over-optimize large trees (OpenJDK evidence).
- DuckDB remains the default provider; LanceDB is optional for this scale path.
- See `operations/lancedb_large_index_reimplementation.md` for the phased design.

## Local soak (optional)

CI covers a multi-page synthetic soak (~800 chunks). For larger local runs
without API keys:

```bash
uv run python scripts/soak_large_index.py --provider lancedb --chunks 5000 --page-size 100
uv run python scripts/soak_large_index.py --provider duckdb --chunks 10000
```

Exit code 0 means insert + stream-embed completed with zero residual missing
chunks, no `get_all_chunks_with_metadata` calls, and a multi-page paginated walk.
