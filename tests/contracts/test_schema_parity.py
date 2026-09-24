"""Schema parity: Rust-created tables must match schema_constants.py.

This is the CI guard for the DDL duplication between
``duckdb_backend.rs::setup_schema`` and
``chunkhound/providers/database/duckdb/schema_constants.py``. When a column
is added or renamed in schema_constants.py, these tests fail until
duckdb_backend.rs is updated to match.

Runs the Rust pipeline against the standard fixture directory purely to
trigger schema creation, rather than instantiating the Rust DB writer
directly (that PyO3 class has been removed; the pipeline is the only
remaining production entry point into the Rust DB backend). A run with an
empty file list against a not-yet-existing DB short-circuits before ever
opening the backend (see `IndexingPipeline::run()`'s `no_db_yet` check), so
at least one real file is required to force the open()/setup_schema() path.
"""

from pathlib import Path

import duckdb
import pytest

from tests.contracts.pipeline_harness import index_with_rust

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def _create_schema_via_rust(db_dir: Path) -> None:
    index_with_rust(FIXTURE_DIR, db_dir, skip_embeddings=True)


class TestSchemaParity:
    @pytest.mark.asyncio
    async def test_files_and_chunks_columns_match_schema_constants(self, tmp_path):
        """Rust-created tables must have the same columns (in order) as
        schema_constants.py."""
        # pylint: disable=protected-access  (accessing private module constants intentionally)
        from chunkhound.providers.database.duckdb.schema_constants import (
            _CHUNKS_COLUMN_NAMES,
            _FILES_COLUMN_NAMES,
        )

        db_dir = tmp_path / "db"
        _create_schema_via_rust(db_dir)

        conn = duckdb.connect(str(db_dir / "chunks.db"))
        try:
            rust_files_cols = [
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'files' ORDER BY ordinal_position"
                ).fetchall()
            ]
            rust_chunks_cols = [
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'chunks' ORDER BY ordinal_position"
                ).fetchall()
            ]
        finally:
            conn.close()

        assert rust_files_cols == _FILES_COLUMN_NAMES, (
            f"files columns mismatch.\n"
            f"  Rust  : {rust_files_cols}\n"
            f"  Python: {_FILES_COLUMN_NAMES}\n"
            f"Update duckdb_backend.rs::setup_schema to match schema_constants.py."
        )
        assert rust_chunks_cols == _CHUNKS_COLUMN_NAMES, (
            f"chunks columns mismatch.\n"
            f"  Rust  : {rust_chunks_cols}\n"
            f"  Python: {_CHUNKS_COLUMN_NAMES}\n"
            f"Update duckdb_backend.rs::setup_schema to match schema_constants.py."
        )

    @pytest.mark.asyncio
    async def test_column_types_match_python_schema(self, tmp_path):
        """Rust DDL column types must match what DuckDB reports for the Python
        canonical DDL.

        Catches type/constraint drift that
        test_files_and_chunks_columns_match_schema_constants cannot detect — e.g.
        changing INTEGER to BIGINT, or TEXT to VARCHAR explicitly. Uses the Python
        DDL as the authoritative reference: both sides are created via DuckDB and
        compared through information_schema, so no hardcoded type strings are
        needed and the test stays valid across DuckDB versions.
        """
        from chunkhound.providers.database.duckdb.schema_constants import (
            _CHUNKS_TABLE_COLUMNS,
            _FILES_TABLE_COLUMNS,
        )

        def _col_types(conn: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
            return {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    f"WHERE table_name = '{table}' ORDER BY ordinal_position"
                ).fetchall()
            }

        # Reference: create tables using the Python canonical DDL.
        ref_conn = duckdb.connect(str(tmp_path / "python_ref.duckdb"))
        try:
            ref_conn.execute("CREATE SEQUENCE IF NOT EXISTS files_id_seq START 1")
            ref_conn.execute("CREATE SEQUENCE IF NOT EXISTS chunks_id_seq START 1")
            ref_conn.execute(f"CREATE TABLE files ({_FILES_TABLE_COLUMNS})")
            ref_conn.execute(f"CREATE TABLE chunks ({_CHUNKS_TABLE_COLUMNS})")
            py_files_types = _col_types(ref_conn, "files")
            py_chunks_types = _col_types(ref_conn, "chunks")
        finally:
            ref_conn.close()

        # Subject: create tables using the Rust DDL via the Rust pipeline's open().
        db_dir = tmp_path / "db"
        _create_schema_via_rust(db_dir)

        rust_conn = duckdb.connect(str(db_dir / "chunks.db"))
        try:
            rust_files_types = _col_types(rust_conn, "files")
            rust_chunks_types = _col_types(rust_conn, "chunks")
        finally:
            rust_conn.close()

        assert rust_files_types == py_files_types, (
            f"files column types mismatch (Rust DDL vs Python DDL).\n"
            f"  Rust  : {rust_files_types}\n"
            f"  Python: {py_files_types}\n"
            "Update duckdb_backend.rs::setup_schema to match schema_constants.py."
        )
        assert rust_chunks_types == py_chunks_types, (
            f"chunks column types mismatch (Rust DDL vs Python DDL).\n"
            f"  Rust  : {rust_chunks_types}\n"
            f"  Python: {py_chunks_types}\n"
            "Update duckdb_backend.rs::setup_schema to match schema_constants.py."
        )
