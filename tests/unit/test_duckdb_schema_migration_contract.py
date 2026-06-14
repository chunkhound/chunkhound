"""Unit test for DuckDB schema auto-creation contract.

Tests the external invariant that opening a DB with no embeddings table
creates one automatically. Data-preservation during migration is covered
by existing tests in test_duckdb_embedding_batch_upsert.py.
"""

from pathlib import Path

import duckdb
import pytest

from chunkhound.core.models import Embedding
from chunkhound.providers.database.duckdb_provider import DuckDBProvider

pytest.importorskip("duckdb")


class TestSchemaAutoCreate:
    """A DB without embeddings tables gets one created on connect."""

    def test_missing_embeddings_table_creates_on_connect(self, tmp_path: Path) -> None:
        """DB with no embeddings table gets one created on connect."""
        db_path = tmp_path / "no_embeddings.duckdb"

        conn = duckdb.connect(str(db_path))
        try:
            conn.execute("""
                CREATE TABLE files (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL,
                    name TEXT,
                    extension TEXT,
                    size INTEGER,
                    modified_time TIMESTAMP,
                    language TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER,
                    symbol TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    code TEXT,
                    chunk_type TEXT,
                    language TEXT
                )
            """)
            conn.execute("""
                INSERT INTO files (
                    id, path, name, extension, size, modified_time, language
                ) VALUES (
                    1, 'helper.py', 'helper.py', '.py',
                    10, CURRENT_TIMESTAMP, 'python'
                )
            """)
            conn.execute("""
                INSERT INTO chunks (
                    id, file_id, symbol, start_line, end_line,
                    code, chunk_type, language
                ) VALUES (
                    1, 1, 'helper', 1, 1,
                    'def helper(): pass', 'function', 'python'
                )
            """)
        finally:
            conn.close()

        provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        provider.connect()
        try:
            tables = provider.execute_query(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 'embeddings%'"
            )
            table_names = [r["table_name"] for r in tables]
            assert any(t.startswith("embeddings_") for t in table_names), (
                f"Expected embeddings_N table, got: {table_names}"
            )

            provider.insert_embedding(
                Embedding(
                    chunk_id=1,
                    provider="test",
                    model="fake",
                    dims=8,
                    vector=[0.1] * 8,
                )
            )

            results, _ = provider.search_semantic(
                query_embedding=[0.1] * 8,
                provider="test",
                model="fake",
                page_size=10,
                offset=0,
            )
            assert len(results) == 1
            assert results[0]["chunk_id"] == 1
        finally:
            provider.disconnect()
