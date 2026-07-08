"""Unit test for DuckDB schema auto-creation contract.

Tests five external invariants:
1. A missing embeddings table gets created lazily on first insert_embedding call.
2. An existing database without canonical columns (created_at, updated_at,
   start_byte, end_byte) has them added on connect (via rebuild).  Data
   preservation during the rebuild is verified inline.
3. A database with correct sequence defaults gets canonical columns added
   via ALTER TABLE (without rebuild).
4. Migration is idempotent — connecting twice produces identical schema
   and data for both files and chunks tables.
5. Legacy ''size'' and ''signature'' columns on the chunks table are
   removed during migration.

Embedding-table data-preservation during migration is covered by existing
tests in test_duckdb_embedding_batch_upsert.py.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from chunkhound.core.models import Embedding
from chunkhound.providers.database.duckdb_provider import DuckDBProvider

pytest.importorskip("duckdb")

# ── Backward-compatibility schema fixtures ──────────────────────────────


@pytest.fixture
def old_schema_db(tmp_path: Path) -> Path:
    """DuckDB with old-style tables (no sequences, no canonical columns)."""
    db_path = tmp_path / "old_schema.duckdb"
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
    return db_path


@pytest.fixture
def canonical_sequence_db(tmp_path: Path) -> Path:
    """DuckDB with correct sequences but missing canonical columns."""
    db_path = tmp_path / "canonical_sequence.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("CREATE SEQUENCE files_id_seq")
        conn.execute("""
            CREATE TABLE files (
                id INTEGER PRIMARY KEY DEFAULT nextval('files_id_seq'),
                path TEXT NOT NULL,
                name TEXT,
                extension TEXT,
                size INTEGER,
                modified_time TIMESTAMP,
                content_hash TEXT,
                language TEXT,
                skip_reason TEXT
                -- NOTE: intentionally omitting created_at, updated_at
            )
        """)
        conn.execute("CREATE SEQUENCE chunks_id_seq")
        conn.execute("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY DEFAULT nextval('chunks_id_seq'),
                file_id INTEGER REFERENCES files(id),
                chunk_type TEXT NOT NULL,
                symbol TEXT,
                code TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                language TEXT,
                metadata TEXT
                -- NOTE: intentionally omitting
                -- start_byte, end_byte,
                -- created_at, updated_at
            )
        """)
        conn.execute("""
            INSERT INTO files (id, path, name, extension, size, modified_time, language)
            VALUES (1, 'helper.py', 'helper.py', '.py', 10, CURRENT_TIMESTAMP, 'python')
        """)
        conn.execute("""
            INSERT INTO chunks (id, file_id, symbol,
                start_line, end_line, code,
                chunk_type, language)
            VALUES (1, 1, 'helper', 1, 1,
                'def helper(): pass',
                'function', 'python')
        """)
    finally:
        conn.close()
    return db_path


# ── Tests ────────────────────────────────────────────────────────────────


class TestSchemaAutoCreate:
    """A DB without embeddings tables gets one created on first insert."""

    def test_missing_embeddings_table_creates_on_insert(
        self, old_schema_db: Path, tmp_path: Path
    ) -> None:
        """DB with no embeddings table gets one created on first insert_embedding."""
        db_path = old_schema_db

        provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        provider.connect()
        try:
            # Verify pre-condition: no embedding table exists yet (lazy creation)
            tables_before = provider.execute_query(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 'embeddings%'"
            )
            assert len(tables_before) == 0, (
                f"Expected no embedding tables before insert, "
                f"got: {[r['table_name'] for r in tables_before]}"
            )

            # Embedding table should be auto-created lazily on first insert
            provider.insert_embedding(
                Embedding(
                    chunk_id=1,
                    provider="test",
                    model="fake",
                    dims=8,
                    vector=[0.1] * 8,
                )
            )

            # Verify embedding table was created lazily
            tables = provider.execute_query(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 'embeddings%'"
            )
            table_names = [r["table_name"] for r in tables]
            assert any(t.startswith("embeddings_") for t in table_names), (
                f"Expected embeddings_N table after insert, got: {table_names}"
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

    def test_old_schema_gets_canonical_columns(
        self, old_schema_db: Path, tmp_path: Path
    ) -> None:
        """Existing DB without canonical columns has them added on connect."""
        db_path = old_schema_db

        provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        provider.connect()
        try:
            # Verify files table has canonical columns after migration
            files_columns = {
                r["column_name"]
                for r in provider.execute_query(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'files'"
                )
            }
            assert "created_at" in files_columns, (
                f"Missing created_at, got: {files_columns}"
            )
            assert "updated_at" in files_columns, (
                f"Missing updated_at, got: {files_columns}"
            )

            # Verify chunks table has canonical columns after migration
            chunks_columns = {
                r["column_name"]
                for r in provider.execute_query(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'chunks'"
                )
            }
            assert "created_at" in chunks_columns, (
                f"Missing created_at, got: {chunks_columns}"
            )
            assert "updated_at" in chunks_columns, (
                f"Missing updated_at, got: {chunks_columns}"
            )
            assert "start_byte" in chunks_columns, (
                f"Missing start_byte, got: {chunks_columns}"
            )
            assert "end_byte" in chunks_columns, (
                f"Missing end_byte, got: {chunks_columns}"
            )

            # Verify existing rows have non-NULL timestamps (DEFAULT CURRENT_TIMESTAMP)
            row = provider.execute_query(
                "SELECT created_at, updated_at FROM chunks WHERE id = 1"
            )
            assert len(row) == 1
            assert row[0]["created_at"] is not None, (
                "created_at should be non-NULL after migration"
            )
            assert row[0]["updated_at"] is not None, (
                "updated_at should be non-NULL after migration"
            )

            # Verify original row data survived the rebuild
            file_row = provider.execute_query(
                "SELECT path, name, extension, language FROM files WHERE id = 1"
            )
            assert len(file_row) == 1
            assert file_row[0]["path"] == "helper.py"
            assert file_row[0]["name"] == "helper.py"
            assert file_row[0]["language"] == "python"

            chunk_row = provider.execute_query(
                "SELECT symbol, code, start_line, end_line FROM chunks WHERE id = 1"
            )
            assert len(chunk_row) == 1
            assert chunk_row[0]["symbol"] == "helper"
            assert chunk_row[0]["code"] == "def helper(): pass"
            assert chunk_row[0]["start_line"] == 1
            assert chunk_row[0]["end_line"] == 1

            # Verify round-trip: insert and read back a chunk with start_byte/end_byte
            conn2 = duckdb.connect(str(db_path))
            try:
                conn2.execute("""
                    INSERT INTO chunks (
                        id, file_id, symbol, start_line, end_line,
                        start_byte, end_byte, code, chunk_type, language
                    ) VALUES (
                        2, 1, 'helper2', 5, 10,
                        100, 200, 'def helper2(): pass', 'function', 'python'
                    )
                """)
            finally:
                conn2.close()

            result = provider.execute_query(
                "SELECT start_byte, end_byte FROM chunks WHERE id = 2"
            )
            assert len(result) == 1
            assert result[0]["start_byte"] == 100
            assert result[0]["end_byte"] == 200
        finally:
            provider.disconnect()

    def test_migration_is_idempotent(self, old_schema_db: Path, tmp_path: Path) -> None:
        """Connecting twice produces identical schema and preserves data."""
        db_path = old_schema_db

        # First connect — runs migration
        provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        provider.connect()
        files_schema_after_first = {
            r["column_name"]
            for r in provider.execute_query(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'files'"
            )
        }
        chunks_schema_after_first = {
            r["column_name"]
            for r in provider.execute_query(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'chunks'"
            )
        }
        files_data_after_first = provider.execute_query(
            "SELECT id, path, name, extension, language FROM files ORDER BY id"
        )
        chunks_data_after_first = provider.execute_query(
            "SELECT id, symbol, code FROM chunks ORDER BY id"
        )
        provider.disconnect()

        # Second connect — must not fail or alter schema/data
        provider2 = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        provider2.connect()
        try:
            files_schema_after_second = {
                r["column_name"]
                for r in provider2.execute_query(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'files'"
                )
            }
            chunks_schema_after_second = {
                r["column_name"]
                for r in provider2.execute_query(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'chunks'"
                )
            }
            files_data_after_second = provider2.execute_query(
                "SELECT id, path, name, extension, language FROM files ORDER BY id"
            )
            chunks_data_after_second = provider2.execute_query(
                "SELECT id, symbol, code FROM chunks ORDER BY id"
            )

            assert files_schema_after_first == files_schema_after_second, (
                f"Files schema changed between connects: "
                f"added={files_schema_after_second - files_schema_after_first}, "
                f"removed={files_schema_after_first - files_schema_after_second}"
            )
            assert chunks_schema_after_first == chunks_schema_after_second, (
                f"Chunks schema changed between connects: "
                f"added={chunks_schema_after_second - chunks_schema_after_first}, "
                f"removed={chunks_schema_after_first - chunks_schema_after_second}"
            )
            assert files_data_after_first == files_data_after_second, (
                f"Files data changed between connects: "
                f"before={files_data_after_first}, after={files_data_after_second}"
            )
            assert chunks_data_after_first == chunks_data_after_second, (
                f"Chunks data changed between connects: "
                f"before={chunks_data_after_first}, after={chunks_data_after_second}"
            )
        finally:
            provider2.disconnect()

    def test_canonical_columns_added_without_rebuild(
        self, canonical_sequence_db: Path, tmp_path: Path
    ) -> None:
        """Existing DB with correct sequence defaults
        gets canonical columns via ALTER TABLE only.

        This exercises the ALTER TABLE path in _executor_materialize_files_table
        and _executor_materialize_chunks_table — the rebuild path is tested by
        test_old_schema_gets_canonical_columns.
        """
        db_path = canonical_sequence_db

        provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        provider.connect()
        try:
            # Verify files table has canonical columns after migration
            files_columns = {
                r["column_name"]
                for r in provider.execute_query(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'files'"
                )
            }
            assert "created_at" in files_columns, (
                f"Missing created_at, got: {files_columns}"
            )
            assert "updated_at" in files_columns, (
                f"Missing updated_at, got: {files_columns}"
            )

            # Verify chunks table has canonical columns after migration
            chunks_columns = {
                r["column_name"]
                for r in provider.execute_query(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'chunks'"
                )
            }
            assert "created_at" in chunks_columns, (
                f"Missing created_at, got: {chunks_columns}"
            )
            assert "updated_at" in chunks_columns, (
                f"Missing updated_at, got: {chunks_columns}"
            )
            assert "start_byte" in chunks_columns, (
                f"Missing start_byte, got: {chunks_columns}"
            )
            assert "end_byte" in chunks_columns, (
                f"Missing end_byte, got: {chunks_columns}"
            )

            # Verify existing rows have non-NULL timestamps (DuckDB backfills
            # ALTER TABLE ADD COLUMN … DEFAULT for existing rows)
            row = provider.execute_query(
                "SELECT created_at, updated_at FROM chunks WHERE id = 1"
            )
            assert len(row) == 1
            assert row[0]["created_at"] is not None, (
                "created_at should be non-NULL after ALTER TABLE"
            )
            assert row[0]["updated_at"] is not None, (
                "updated_at should be non-NULL after ALTER TABLE"
            )

            # Verify round-trip: insert and read back a chunk with start_byte/end_byte
            conn2 = duckdb.connect(str(db_path))
            try:
                conn2.execute("""
                    INSERT INTO chunks (
                        id, file_id, symbol, start_line, end_line,
                        start_byte, end_byte, code, chunk_type, language
                    ) VALUES (
                        2, 1, 'helper2', 5, 10,
                        100, 200, 'def helper2(): pass', 'function', 'python'
                    )
                """)
            finally:
                conn2.close()

            result = provider.execute_query(
                "SELECT start_byte, end_byte FROM chunks WHERE id = 2"
            )
            assert len(result) == 1
            assert result[0]["start_byte"] == 100
            assert result[0]["end_byte"] == 200
        finally:
            provider.disconnect()

    def test_legacy_size_signature_columns_are_removed(self, tmp_path: Path) -> None:
        """Legacy 'size' and 'signature' columns
        on chunks are removed during migration."""
        db_path = tmp_path / "legacy_size_sig.duckdb"

        conn = duckdb.connect(str(db_path))
        try:
            conn.execute("CREATE SEQUENCE files_id_seq")
            conn.execute("""
                CREATE TABLE files (
                    id INTEGER PRIMARY KEY DEFAULT nextval('files_id_seq'),
                    path TEXT NOT NULL,
                    name TEXT,
                    extension TEXT,
                    size INTEGER,
                    modified_time TIMESTAMP,
                    content_hash TEXT,
                    language TEXT,
                    skip_reason TEXT
                )
            """)
            conn.execute("CREATE SEQUENCE chunks_id_seq")
            conn.execute("""
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY DEFAULT nextval('chunks_id_seq'),
                    file_id INTEGER REFERENCES files(id),
                    chunk_type TEXT NOT NULL,
                    symbol TEXT,
                    code TEXT NOT NULL,
                    start_line INTEGER,
                    end_line INTEGER,
                    language TEXT,
                    metadata TEXT,
                    size INTEGER,
                    signature TEXT
                    -- NOTE: intentionally omitting
                    -- start_byte, end_byte,
                    -- created_at, updated_at
                )
            """)
            conn.execute("""
                INSERT INTO files (id, path, name,
                    extension, size,
                    modified_time, language)
                VALUES (1, 'helper.py', 'helper.py',
                    '.py', 10,
                    CURRENT_TIMESTAMP, 'python')
            """)
            conn.execute("""
                INSERT INTO chunks (
                    id, file_id, symbol, start_line, end_line,
                    code, chunk_type, language, size, signature
                ) VALUES (
                    1, 1, 'helper', 1, 1,
                    'def helper(): pass', 'function', 'python', 42, 'abc123'
                )
            """)
        finally:
            conn.close()

        provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        provider.connect()
        try:
            # Verify legacy columns are gone from chunks
            chunks_columns = {
                r["column_name"]
                for r in provider.execute_query(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'chunks'"
                )
            }
            assert "size" not in chunks_columns, (
                f"Legacy 'size' should have been removed from chunks, "
                f"got: {chunks_columns}"
            )
            assert "signature" not in chunks_columns, (
                f"Legacy 'signature' should have been removed from chunks, "
                f"got: {chunks_columns}"
            )

            # Verify canonical columns are present
            assert "start_byte" in chunks_columns
            assert "end_byte" in chunks_columns
            assert "created_at" in chunks_columns
            assert "updated_at" in chunks_columns

            # Verify original row data survived the drop+recreate
            chunk_row = provider.execute_query(
                "SELECT id, symbol, code, start_line, end_line FROM chunks WHERE id = 1"
            )
            assert len(chunk_row) == 1
            assert chunk_row[0]["symbol"] == "helper"
            assert chunk_row[0]["code"] == "def helper(): pass"

            # Verify round-trip: insert and read back a chunk with start_byte/end_byte
            conn2 = duckdb.connect(str(db_path))
            try:
                conn2.execute("""
                    INSERT INTO chunks (
                        id, file_id, symbol, start_line, end_line,
                        start_byte, end_byte, code, chunk_type, language
                    ) VALUES (
                        2, 1, 'helper2', 5, 10,
                        100, 200, 'def helper2(): pass', 'function', 'python'
                    )
                """)
            finally:
                conn2.close()

            result = provider.execute_query(
                "SELECT start_byte, end_byte FROM chunks WHERE id = 2"
            )
            assert len(result) == 1
            assert result[0]["start_byte"] == 100
            assert result[0]["end_byte"] == 200
        finally:
            provider.disconnect()
