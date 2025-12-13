Operations experiments and diagnostics
======================================

The ``operations`` directory contains ad-hoc experiments and operational
documentation that are **not** part of the main ChunkHound library API.

- ``database_concurrency.md`` – notes and conclusions from DuckDB/LanceDB
  concurrency probes, including the single-owner / SerialDatabaseProvider
  policy enforced by ChunkHound.
- ``experiments/duckdb_concurrency_probe.py`` – standalone script that
  demonstrates DuckDB lock failures under multi-process access.
- ``experiments/lancedb_concurrency_probe.py`` – standalone script that
  explores LanceDB behaviour under one-writer / many-reader scenarios.

These scripts are intended for local investigation and validation of
operational assumptions; they should not be imported by production code.

