# Database Concurrency Operations

This note captures the current, experiment-backed stance on database concurrency for ChunkHound. It is informational for operators and maintainers; the production rule remains **single-owner, single-threaded access per database path** enforced via `SerialDatabaseProvider`.

## Summary

- **DuckDB**: Designed for a single owning process per database file. When a second process attempts to connect to the same `.duckdb` file, DuckDB fails fast with a locking error.
- **LanceDB**: The library is designed to support concurrent access (multiple readers and at least one writer) and, in practice, can tolerate one writer process plus multiple reader processes under our test conditions, provided:
  - Each process uses its own LanceDB connection.
  - Processes are started with the `spawn` start method (not `fork`).
  - Only one process performs writes for a given database directory at any time.
- **ChunkHound policy**: ChunkHound deliberately treats each database path as owned by a single process and routes all access through `SerialDatabaseProvider`. This is a project-level simplification for predictability and consistency across backends, not a statement that LanceDB itself is unable to handle concurrency.

## Experiments

Two standalone scripts under `operations/experiments/` probe the backends with ChunkHound-like chunk + embedding data:

- `operations/experiments/lancedb_concurrency_probe.py`
  - Seeds a LanceDB database with `files` and `chunks` tables, where `chunks.embedding` is a fixed-size float vector and `provider`/`model` columns mirror real usage.
  - Spawns **1 writer + N readers** using `multiprocessing.get_context("spawn")`.
  - Writer:
    - Re-embeds existing rows.
    - Inserts new rows with embeddings (new chunk IDs).
  - Readers:
    - Run vector search queries on the embedding column filtered by `provider`/`model`.
    - Count how often they see rows with `id > seed_rows` (freshly inserted data).
  - Observed behavior for representative runs (e.g., `--rows 5000 --readers 4 --writers 1 --duration 10`):
    - Non-zero `reads_ok` and `writes_ok`.
    - `fresh_read_iterations` and `fresh_read_results` > 0, meaning readers see new data while the writer runs.
    - `reads_err` and `writes_err` remained 0 in these tests.

- `operations/experiments/duckdb_concurrency_probe.py`
  - Seeds a DuckDB database file with a `chunks` table using an `embedding FLOAT[dims]` column and similar metadata columns.
  - Attempts the same **1 writer + N readers** pattern, again with `spawn`.
  - Writer connects and performs updates/inserts successfully.
  - Reader processes fail at connect time with:
    - `_duckdb.IOException: IO Error: Could not set lock on file "...": Conflicting lock is held ...`
  - Result: no successful concurrent reads while the writer holds the file lock.

These experiments are intentionally isolated: they do **not** use ChunkHound providers or services and should not be treated as production patterns.

## Operational Guidance

### Backend reality (from experiments)

- DuckDB:
  - A second process attempting to connect to a database file already opened by another process fails immediately with a locking error, even in read-only mode.
  - Practically, this means “one OS process at a time per DuckDB file” for the scenarios we tested.
- LanceDB:
  - With `spawn`-started processes, separate connections, and a single writer, we observed stable 1-writer + N-reader behavior on chunk+embedding workloads.
  - Readers not only stayed error-free, they also saw newly inserted rows while the writer was active.

### How ChunkHound uses them today

- ChunkHound is intentionally conservative and treats every configured database path as owned by a **single process**, regardless of backend.
- All database access in the core system flows through `SerialDatabaseProvider` and its single-threaded executor to:
  - Serialize operations inside the owning process.
  - Keep the concurrency model identical for DuckDB and LanceDB and avoid backend-specific surprises.
- If you need multiple MCP endpoints or CLIs around a single index in a production-like setting:
  - Either route through the same owning process (e.g., via HTTP/RPC).
  - Or give each process its own independent database path rather than sharing one DuckDB/LanceDB directory.
- When experimenting with LanceDB **outside** the core ChunkHound stack, the 1-writer + N-reader pattern demonstrated by the probe is a valid option, but it is currently **out of scope** for the supported ChunkHound architecture.
