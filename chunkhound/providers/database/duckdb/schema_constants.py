"""Shared DDL definitions and naming helpers for the DuckDB provider.

Single source of truth for table schemas, index names, and column lists.
Used by duckdb_provider.py (create/migrate/compact) and
embedding_repository.py (upsert fallback path).
"""

from __future__ import annotations

from typing import Any

# ── Schema version ──────────────────────────────────────────────────────

_SCHEMA_VERSION_TABLE_COLUMNS: str = """\
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
"""

# Canonical statically-named tables that compaction must preserve.
# Embedding tables (embeddings_<dims>) are discovered dynamically from
# information_schema and are NOT included here.  When adding a new
# statically-named canonical table, add it here AND update
# _compact_copy_data in duckdb_provider.py to CREATE + INSERT it.
CANONICAL_TABLE_NAMES: set[str] = {"schema_version", "files", "chunks"}


# ── Files table ─────────────────────────────────────────────────────────

_FILES_TABLE_COLUMNS: str = """\
    id INTEGER PRIMARY KEY DEFAULT nextval('files_id_seq'),
    path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    extension TEXT,
    size INTEGER,
    modified_time TIMESTAMP,
    content_hash TEXT,
    language TEXT,
    skip_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
"""

# ── Chunks table ────────────────────────────────────────────────────────

_CHUNKS_TABLE_COLUMNS: str = """\
    id INTEGER PRIMARY KEY DEFAULT nextval('chunks_id_seq'),
    file_id INTEGER REFERENCES files(id),
    chunk_type TEXT NOT NULL,
    symbol TEXT,
    code TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    start_byte INTEGER,
    end_byte INTEGER,
    language TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
"""


# ── Embedding table helpers ─────────────────────────────────────────────

# SIMILAR TO pattern for matching canonical embedding table names.
# Only matches dimension-suffixed tables (embeddings_384) and excludes
# non-canonical names like embeddings_backup or embeddings_staging.
EMBEDDING_TABLE_SIMILAR_PATTERN = "embeddings_[0-9]+"
_DUCKDB_DEFAULT_SCHEMA = "main"

# Catalogs and schemas that the WHERE clause helpers accept.
# Input validation prevents SQL injection from malformed identifiers.
# If adding a new catalog or schema, add it here AND update the
# docstring of _executor_get_existing_vector_indexes_from_catalog
# in duckdb_provider.py to list valid catalogs explicitly.
VALID_CATALOGS: set[str] = {"main", "src"}
VALID_SCHEMAS: set[str] = {"main"}


def _assert_allowed_identifier(value: str, allowed: set[str], kind: str) -> None:
    """Raise ValueError when *value* is not in the *allowed* set."""
    if value not in allowed:
        raise ValueError(
            f"Invalid {kind} {value!r}; expected one of {sorted(allowed)}"
        )


def _embedding_tables_where_clause(
    *,
    catalog: str | None = None,
    schema: str = _DUCKDB_DEFAULT_SCHEMA,
) -> str:
    """Return the shared information_schema filter for embedding tables."""
    _assert_allowed_identifier(schema, VALID_SCHEMAS, "schema")
    filters = [f"table_schema = '{schema}'"]
    if catalog is not None:
        _assert_allowed_identifier(catalog, VALID_CATALOGS, "catalog")
        filters.append(f"table_catalog = '{catalog}'")
    filters.append(
        f"table_name SIMILAR TO '{EMBEDDING_TABLE_SIMILAR_PATTERN}'"
    )
    return " AND ".join(filters)


def _embedding_indexes_where_clause(
    *,
    catalog: str | None = None,
    schema: str = _DUCKDB_DEFAULT_SCHEMA,
) -> str:
    """Return the shared duckdb_indexes() scope for embedding tables."""
    _assert_allowed_identifier(schema, VALID_SCHEMAS, "schema")
    filters = [f"schema_name = '{schema}'"]
    if catalog is not None:
        _assert_allowed_identifier(catalog, VALID_CATALOGS, "catalog")
        filters.append(f"database_name = '{catalog}'")
    filters.append(f"table_name SIMILAR TO '{EMBEDDING_TABLE_SIMILAR_PATTERN}'")
    return " AND ".join(filters)


def is_hnsw_index(index_name: str, create_sql: str | None) -> bool:
    """Return True when index metadata describes an HNSW index.

    Checks ``USING HNSW`` in the DDL first (handles custom-named indexes
    like ``alt_live_idx``), then falls back to canonical name prefixes.
    """
    if create_sql and "USING HNSW" in create_sql.upper():
        return True
    return index_name.startswith("hnsw_") or index_name.startswith("idx_hnsw_")


def fallback_embedding_tables(connection: Any) -> list[str]:
    """Discover embedding tables from the default catalog/schema.

    Used as a fallback when the provider is not available (e.g. tests).
    """
    rows = connection.execute(
        "SELECT table_name FROM information_schema.tables "
        f"WHERE {_embedding_tables_where_clause()}"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _embedding_table_name(dims: int) -> str:
    """Return the canonical embedding table name for one dimension count."""
    return f"embeddings_{dims}"


def _embedding_hnsw_index_name(dims: int) -> str:
    """Return the canonical shared HNSW index name for one embedding table."""
    return f"idx_hnsw_{dims}"


def _embedding_chunk_id_index_name(dims: int) -> str:
    """Return the canonical chunk-id lookup index name for one embedding table.

    Legacy compat: the 1536-dimension index was originally named
    ``idx_embeddings_1536_chunk_id`` (with the ``embeddings_`` prefix before
    ``1536``), matching the original schema.  Newer dimension indexes use
    the compact ``idx_{dims}_chunk_id`` convention without the redundant
    ``embeddings_`` prefix.  The 1536 shim preserves backwards compatibility
    with databases created before the naming was normalized.
    """
    if dims == 1536:
        return "idx_embeddings_1536_chunk_id"
    return f"idx_{dims}_chunk_id"


def _embedding_provider_model_index_name(dims: int) -> str:
    """Return the canonical provider/model lookup index name for one embedding table."""
    return f"idx_{dims}_provider_model"


def _embedding_unique_index_name(dims: int) -> str:
    """Return the canonical unique upsert-contract index name for one embedding table."""
    return f"idx_{dims}_chunk_provider_model_unique"


def _embedding_table_columns(dims: int) -> str:
    """Return column-definition fragment for an embedding table of given dimensions."""
    return f"""\
    id INTEGER PRIMARY KEY DEFAULT nextval('embeddings_id_seq'),
    chunk_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    embedding FLOAT[{dims}],
    dims INTEGER NOT NULL DEFAULT {dims},
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
"""


def _create_embedding_table_sql(dims: int) -> str:
    """Return canonical CREATE TABLE SQL for one embedding table."""
    return (
        f"CREATE TABLE {_embedding_table_name(dims)} ({_embedding_table_columns(dims)})"
    )


def _embedding_dims_from_table_name(table_name: str) -> int | None:
    """Extract dimension count from an embedding table name.

    Returns None if the table name doesn't match the ``embeddings_{dims}``
    pattern (e.g. a non-embedding or unexpected table).
    """
    if not table_name.startswith("embeddings_"):
        return None
    try:
        return int(table_name[11:])
    except ValueError:
        return None


# ── Column name extraction ──────────────────────────────────────────────


def _extract_column_names(cols_sql: str) -> list[str]:
    """Parse column names from a DDL column-definition fragment.

    Input like ``"id INTEGER PRIMARY KEY ..., path TEXT ..."`` → ``["id", "path", ...]``.
    Only extracts the first token of each comma-separated entry.
    """
    names: list[str] = []
    for part in cols_sql.split(","):
        token = part.strip().split()[0] if part.strip() else ""
        if token and token.isidentifier():
            names.append(token)
    return names


_FILES_COLUMN_NAMES: list[str] = _extract_column_names(_FILES_TABLE_COLUMNS)
_CHUNKS_COLUMN_NAMES: list[str] = _extract_column_names(_CHUNKS_TABLE_COLUMNS)


def _embeddings_column_names(dims: int) -> list[str]:
    """Return column names for an embedding table of given dimensions."""
    return _extract_column_names(_embedding_table_columns(dims))
