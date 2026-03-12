"""Graph storage provider – DuckDB schema and queries for the edges table.

This module manages the ``edges`` table that sits alongside the existing
``files``, ``chunks``, and ``embeddings`` tables.  It follows the same
patterns used by ``DuckDBProvider``:

* Schema creation is idempotent (``CREATE TABLE IF NOT EXISTS``).
* All writes go through the caller's ``SerialDatabaseProvider`` executor.
* Reads are plain SQL queries returning dicts.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from chunkhound.graph.models import Edge, EdgeType


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

EDGES_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS edges (
    id              BIGINT PRIMARY KEY DEFAULT nextval('edges_id_seq'),
    source_chunk_id BIGINT NOT NULL,
    target_chunk_id BIGINT NOT NULL,
    edge_type       TEXT   NOT NULL,
    confidence      FLOAT  NOT NULL DEFAULT 1.0,
    metadata        JSON
);
"""

EDGES_SEQUENCE_DDL = """\
CREATE SEQUENCE IF NOT EXISTS edges_id_seq START 1;
"""

EDGES_INDEXES_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source_chunk_id);",
    "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target_chunk_id);",
    "CREATE INDEX IF NOT EXISTS idx_edges_type   ON edges (edge_type);",
]


def create_edges_schema(execute_fn: Any) -> None:
    """Create the edges table and indexes idempotently.

    Args:
        execute_fn: A callable that accepts a SQL string and optional params,
                    e.g. ``provider.execute_query``.
    """
    execute_fn(EDGES_SEQUENCE_DDL)
    execute_fn(EDGES_TABLE_DDL)
    for ddl in EDGES_INDEXES_DDL:
        execute_fn(ddl)
    logger.debug("Graph: edges schema ensured.")


# ------------------------------------------------------------------
# Write operations
# ------------------------------------------------------------------

def insert_edges_batch(
    execute_fn: Any,
    edges: list[Edge],
    batch_size: int = 500,
) -> int:
    """Batch-insert edges.  Returns the number of rows inserted."""
    if not edges:
        return 0

    total = 0
    for i in range(0, len(edges), batch_size):
        batch = edges[i : i + batch_size]
        values_clause = ", ".join(
            [
                f"({e.source_chunk_id}, {e.target_chunk_id}, "
                f"'{e.edge_type.value}', {e.confidence}, "
                f"'{_escape_json(e.metadata)}')"
                for e in batch
            ]
        )
        sql = (
            "INSERT INTO edges (source_chunk_id, target_chunk_id, "
            "edge_type, confidence, metadata) VALUES " + values_clause
        )
        execute_fn(sql)
        total += len(batch)

    logger.debug(f"Graph: inserted {total} edges.")
    return total


def delete_edges_for_chunks(execute_fn: Any, chunk_ids: list[int]) -> int:
    """Delete all edges referencing any of the given chunk IDs."""
    if not chunk_ids:
        return 0
    id_list = ", ".join(str(c) for c in chunk_ids)
    sql = (
        f"DELETE FROM edges WHERE source_chunk_id IN ({id_list}) "
        f"OR target_chunk_id IN ({id_list})"
    )
    execute_fn(sql)
    return len(chunk_ids)


# ------------------------------------------------------------------
# Read operations
# ------------------------------------------------------------------

def get_edges_from(
    execute_fn: Any,
    chunk_id: int,
    edge_types: list[EdgeType] | None = None,
) -> list[dict[str, Any]]:
    """Get outgoing edges (downstream) from a chunk."""
    type_filter = _edge_type_filter(edge_types)
    sql = (
        "SELECT e.*, c.symbol AS target_symbol, f.path AS target_file_path "
        "FROM edges e "
        "LEFT JOIN chunks c ON c.id = e.target_chunk_id "
        "LEFT JOIN files f ON f.id = c.file_id "
        f"WHERE e.source_chunk_id = {chunk_id} {type_filter} "
        "ORDER BY e.confidence DESC"
    )
    return execute_fn(sql)


def get_edges_to(
    execute_fn: Any,
    chunk_id: int,
    edge_types: list[EdgeType] | None = None,
) -> list[dict[str, Any]]:
    """Get incoming edges (upstream) to a chunk."""
    type_filter = _edge_type_filter(edge_types)
    sql = (
        "SELECT e.*, c.symbol AS source_symbol, f.path AS source_file_path "
        "FROM edges e "
        "LEFT JOIN chunks c ON c.id = e.source_chunk_id "
        "LEFT JOIN files f ON f.id = c.file_id "
        f"WHERE e.target_chunk_id = {chunk_id} {type_filter} "
        "ORDER BY e.confidence DESC"
    )
    return execute_fn(sql)


def find_chunk_ids_by_symbol(
    execute_fn: Any,
    symbol: str,
    path_filter: str | None = None,
) -> list[int]:
    """Resolve a symbol name to chunk IDs."""
    path_clause = ""
    if path_filter:
        safe_path = path_filter.replace("'", "''")
        path_clause = f" AND f.path LIKE '{safe_path}%'"
    sql = (
        "SELECT c.id FROM chunks c "
        "LEFT JOIN files f ON f.id = c.file_id "
        f"WHERE c.symbol = '{_escape(symbol)}' {path_clause}"
    )
    rows = execute_fn(sql)
    return [int(r["id"]) for r in rows]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _edge_type_filter(edge_types: list[EdgeType] | None) -> str:
    if not edge_types:
        return ""
    types = ", ".join(f"'{et.value}'" for et in edge_types)
    return f"AND e.edge_type IN ({types})"


def _escape(value: str) -> str:
    return value.replace("'", "''")


def _escape_json(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data).replace("'", "''")