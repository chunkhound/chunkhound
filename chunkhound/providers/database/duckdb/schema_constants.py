"""Centralized DuckDB schema constants for ChunkHound tables and indexes.

These constants are used by the compaction logic to identify which tables
belong to ChunkHound (and should be preserved during a rebuild) vs. unknown
tables that may have been added externally (and are dropped during compaction).
"""

FILES_TABLE = "files"
CHUNKS_TABLE = "chunks"
SCHEMA_VERSION_TABLE = "schema_version"

FILES_SEQUENCE = "files_id_seq"
CHUNKS_SEQUENCE = "chunks_id_seq"
EMBEDDINGS_SEQUENCE = "embeddings_id_seq"

# Tables that always exist in a ChunkHound database (not the dynamic embedding tables)
CHUNKHOUND_CORE_TABLES = (FILES_TABLE, CHUNKS_TABLE, SCHEMA_VERSION_TABLE)

# Prefix for all dynamic embedding tables (e.g. embeddings_1536, embeddings_256)
EMBEDDINGS_TABLE_PREFIX = "embeddings_"


def embedding_table_name(dims: int) -> str:
    """Return the embedding table name for a given dimension count."""
    return f"{EMBEDDINGS_TABLE_PREFIX}{dims}"


def hnsw_index_name(table_name: str) -> str:
    """Return the canonical HNSW index name for a given embedding table."""
    return f"hnsw_{table_name}"


def is_chunkhound_table(table_name: str) -> bool:
    """Return True if table_name is a ChunkHound-managed table."""
    return table_name in CHUNKHOUND_CORE_TABLES or table_name.startswith(EMBEDDINGS_TABLE_PREFIX)
