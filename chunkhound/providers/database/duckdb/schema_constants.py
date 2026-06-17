"""Shared DDL definitions and naming helpers for the DuckDB provider.

Single source of truth for table schemas, index names, and column lists.
Used by duckdb_provider.py (create/migrate/compact) and
embedding_repository.py (upsert fallback path).
"""

# ── Schema version ──────────────────────────────────────────────────────

_SCHEMA_VERSION_TABLE_COLUMNS: str = """\
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
"""

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
