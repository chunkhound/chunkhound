"""Prepare real row waste without changing source files or live database rows."""

from pathlib import Path

import duckdb

from tests.contracts.pipeline_harness import hnsw_index_names, index_with_rust


def prepare_compaction(
    fixture_dir: Path, db_dir: Path, *, enabled: bool = True
) -> None:
    """Create a real store, then add waste before the measured indexing run."""
    result = index_with_rust(
        fixture_dir,
        db_dir,
        duckdb_hnsw_enabled=enabled,
        compaction_threshold=1.0,
    )
    assert not result.errors, result.errors
    assert result.chunks_written > 0
    seed_compaction_waste(db_dir)


def seed_compaction_waste(db_dir: Path) -> None:
    """Checkpoint scratch chunks before deletion so zero-threshold compaction is due.

    Only chunks are touched. Load installed VSS only for existing HNSW indexes:
    checkpointing those indexes requires it, but this helper must not download it.
    """
    has_hnsw = bool(hnsw_index_names(db_dir))
    with duckdb.connect(str(db_dir / "chunks.db")) as conn:
        if has_hnsw:
            conn.execute("LOAD vss")
        before = _live_rows(conn)
        _write_deleted_chunks(conn)
        assert _live_rows(conn) == before, "Waste preparation changed live data"
    assert dead_row_count(db_dir) > 0, "Waste preparation left no dead rows"


def dead_row_count(db_dir: Path, table: str = "chunks") -> int:
    """Rows stored in *table* that are no longer live — pre-compaction waste.

    Reads the DuckDB file directly so the same probe works for any writer
    (Python provider or Rust pipeline) and survives rewriting either.
    Counts the scalar ``id`` column only: a ``FLOAT[dims]`` column also emits
    child validity/segment rows whose count is the child-vector length, not the
    row count. MAX per row group because one column can span several segments
    inside one group. Callers must ``CHECKPOINT`` first — on-disk row groups
    are what this measures.
    """
    escaped = table.replace("'", "''")
    conn = duckdb.connect(str(db_dir / "chunks.db"))
    try:
        stored = conn.execute(
            "SELECT coalesce(sum(cnt), 0) FROM ("
            "SELECT row_group_id, MAX(count) AS cnt "
            f"FROM pragma_storage_info('{escaped}') "
            "WHERE column_name = 'id' GROUP BY row_group_id)"
        ).fetchone()[0]
        live = conn.execute(f'SELECT count(*) FROM "{escaped}"').fetchone()[0]
    finally:
        conn.close()
    return max(int(stored) - int(live), 0)


def _write_deleted_chunks(conn: duckdb.DuckDBPyConnection) -> None:
    start = conn.execute("SELECT coalesce(max(id), 0) + 1 FROM chunks").fetchone()[0]
    conn.execute(
        "INSERT INTO chunks (id, chunk_type, code) "
        "SELECT ? + i, 'function', 'compaction scratch' FROM range(100) t(i)",
        [start],
    )
    conn.execute("CHECKPOINT")
    conn.execute("DELETE FROM chunks WHERE id >= ? AND id < ?", [start, start + 100])
    conn.execute("CHECKPOINT")


def _live_rows(conn: duckdb.DuckDBPyConnection) -> dict[str, list[tuple]]:
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name IN ('files', 'chunks') OR table_name LIKE 'embeddings_%'"
    ).fetchall()
    return {
        table: conn.execute(f'SELECT * FROM "{table}" ORDER BY id').fetchall()
        for (table,) in tables
    }
