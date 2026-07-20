"""Tests for the Phase 0 Rust DB writer (RustDbWriter PyO3 extension).

Requires the native extension to be built:
    maturin develop --release

All tests that open a DuckDB connection for verification do so ONLY after
writer.close() — DuckDB enforces single-writer.
"""

from __future__ import annotations

import os

import duckdb
import pytest

chunkhound_native = pytest.importorskip(
    "chunkhound_native",
    reason="native extension not built — run: maturin develop --release",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_writer(db_path: str, threshold: int = 50):
    return chunkhound_native.RustDbWriter(
        {"db_path": db_path, "compaction_batch_threshold": threshold}
    )


def _chunk(
    code: str = "def foo(): pass",
    *,
    dims: int = 0,
    symbol: str = "foo",
) -> dict:
    return {
        "chunk_type": "function",
        "symbol": symbol,
        "code": code,
        "start_line": 1,
        "end_line": 2,
        "start_byte": None,
        "end_byte": None,
        "language": "python",
        "metadata": None,
        "embedding": [0.1] * dims if dims else None,
        "provider": "test" if dims else None,
        "model": "test-model" if dims else None,
    }


def _file(
    path: str = "a.py",
    *,
    chunks: list | None = None,
    existing_file_id: int | None = None,
) -> dict:
    return {
        "existing_file_id": existing_file_id,
        "path": path,
        "mtime": 1.0,
        "size_bytes": 100,
        "content_hash": "abc123",
        "language": "python",
        "chunks": chunks or [],
    }


def _batch(*, files: list | None = None, delete_paths: list | None = None) -> dict:
    return {"files": files or [], "delete_paths": delete_paths or []}


def _count(db_path: str, table: str) -> int:
    """Count rows in a table — call only after writer.close()."""
    conn = duckdb.connect(db_path)
    try:
        return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_open_close_creates_schema(self, tmp_path):
        """open() followed by close() must create the files/chunks tables."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db)
        w.open()
        w.close()

        assert _count(db, "files") == 0
        assert _count(db, "chunks") == 0

    def test_write_before_open_raises(self, tmp_path):
        """write_batch() without open() must raise RuntimeError containing 'not open'."""
        w = _make_writer(str(tmp_path / "t.duckdb"))
        with pytest.raises(RuntimeError, match="not open"):
            w.write_batch(_batch(files=[_file()]))

    def test_double_open_is_safe(self, tmp_path):
        """Calling open() twice must not raise or corrupt state."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db)
        w.open()
        w.open()
        w.close()
        assert _count(db, "files") == 0


# ---------------------------------------------------------------------------
# Basic write tests
# ---------------------------------------------------------------------------

class TestBasicWrite:
    def test_write_single_file_no_embedding(self, tmp_path):
        """A single file with one chunk (no embedding) is persisted correctly."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db)
        w.open()
        result = w.write_batch(_batch(files=[_file("a.py", chunks=[_chunk()])]))
        w.close()

        assert len(result["file_ids"]) == 1
        assert result["chunks_written"] == 1
        assert result["embeddings_written"] == 0
        assert _count(db, "files") == 1
        assert _count(db, "chunks") == 1

    def test_batch_result_schema(self, tmp_path):
        """write_batch return value must have file_ids, chunks_written, embeddings_written."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db)
        w.open()
        result = w.write_batch(_batch(files=[_file(chunks=[_chunk()])]))
        w.close()

        assert "file_ids" in result
        assert "chunks_written" in result
        assert "embeddings_written" in result
        assert isinstance(result["file_ids"], list)
        assert isinstance(result["chunks_written"], int)
        assert isinstance(result["embeddings_written"], int)

    def test_write_with_embedding(self, tmp_path):
        """A chunk with a 384-dim embedding is persisted in embeddings_384."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db)
        w.open()
        result = w.write_batch(
            _batch(files=[_file(chunks=[_chunk(dims=384)])])
        )
        w.close()

        assert result["embeddings_written"] == 1
        assert _count(db, "embeddings_384") == 1

    def test_write_multiple_files(self, tmp_path):
        """A batch with three files must persist all three."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db)
        w.open()
        files = [_file(f"{i}.py", chunks=[_chunk()]) for i in range(3)]
        result = w.write_batch(_batch(files=files))
        w.close()

        assert len(result["file_ids"]) == 3
        assert result["chunks_written"] == 3
        assert _count(db, "files") == 3
        assert _count(db, "chunks") == 3


# ---------------------------------------------------------------------------
# Upsert / idempotency
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_upsert_idempotent_no_duplication(self, tmp_path):
        """Writing the same file twice must not duplicate rows in files or chunks."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db)
        w.open()
        batch = _batch(files=[_file("a.py", chunks=[_chunk()])])
        w.write_batch(batch)
        w.write_batch(batch)
        w.close()

        assert _count(db, "files") == 1
        assert _count(db, "chunks") == 1

    def test_upsert_replaces_chunks_on_update(self, tmp_path):
        """Re-indexing a file with existing_file_id set must replace old chunks."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db)
        w.open()

        result1 = w.write_batch(_batch(files=[_file("a.py", chunks=[_chunk("def v1(): pass")])]))
        fid = result1["file_ids"][0]

        # Second write with existing_file_id → old chunks deleted first
        w.write_batch(_batch(files=[_file("a.py", chunks=[_chunk("def v2(): pass"), _chunk("def v3(): pass")], existing_file_id=fid)]))
        w.close()

        assert _count(db, "files") == 1
        assert _count(db, "chunks") == 2


# ---------------------------------------------------------------------------
# Deletion path
# ---------------------------------------------------------------------------

class TestDeletion:
    def test_delete_path_removes_file_and_chunks(self, tmp_path):
        """delete_paths must remove files and associated chunks."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db)
        w.open()
        w.write_batch(_batch(files=[_file("a.py", chunks=[_chunk(), _chunk(code="def bar(): pass")])]))
        w.write_batch(_batch(delete_paths=["a.py"]))
        w.close()

        assert _count(db, "files") == 0
        assert _count(db, "chunks") == 0

    def test_delete_nonexistent_path_is_safe(self, tmp_path):
        """Deleting a path that was never indexed must not raise."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db)
        w.open()
        w.write_batch(_batch(delete_paths=["does_not_exist.py"]))
        w.close()

        assert _count(db, "files") == 0


# ---------------------------------------------------------------------------
# Compaction counter
# ---------------------------------------------------------------------------

class TestCompaction:
    def test_needs_compaction_false_initially(self, tmp_path):
        """needs_compaction() must return False before threshold is reached."""
        w = _make_writer(str(tmp_path / "t.duckdb"), threshold=3)
        w.open()
        assert w.needs_compaction() is False
        w.close()

    def test_needs_compaction_true_at_threshold(self, tmp_path):
        """needs_compaction() must return True once write_count >= threshold."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db, threshold=2)
        w.open()
        w.write_batch(_batch(files=[_file("a.py")]))
        w.write_batch(_batch(files=[_file("b.py")]))
        assert w.needs_compaction() is True
        w.close()

    def test_run_compaction_resets_counter(self, tmp_path):
        """run_compaction() must reset the write counter so needs_compaction() returns False."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db, threshold=1)
        w.open()
        w.write_batch(_batch(files=[_file()]))
        assert w.needs_compaction() is True
        w.run_compaction()
        assert w.needs_compaction() is False
        w.close()


# ---------------------------------------------------------------------------
# Crash recovery (Invariant 17)
# ---------------------------------------------------------------------------

class TestCrashRecovery:
    def test_pre_swap_intent_cleared_on_open(self, tmp_path):
        """Intent file containing 'pre-swap' must be removed; DB opens normally."""
        db_path = tmp_path / "t.duckdb"
        intent_path = tmp_path / "t.duckdb.swap_intent"
        intent_path.write_text("pre-swap")

        w = _make_writer(str(db_path))
        w.open()
        w.close()

        assert not intent_path.exists()
        assert _count(str(db_path), "files") == 0

    def test_phase1_intent_restores_old_file(self, tmp_path):
        """Intent 'phase1' must restore the .old backup to the main DB path."""
        db_path = tmp_path / "t.duckdb"
        old_path = tmp_path / "t.duckdb.old"
        intent_path = tmp_path / "t.duckdb.swap_intent"

        # Simulate a DB that was backed up but not yet swapped
        # Create a valid DB at old_path by opening/closing it there first
        bootstrap = _make_writer(str(old_path))
        bootstrap.open()
        bootstrap.write_batch(_batch(files=[_file("seed.py")]))
        bootstrap.close()

        intent_path.write_text("phase1")

        # open() should detect phase1, rename old → db_path
        w = _make_writer(str(db_path))
        w.open()
        w.close()

        assert not intent_path.exists()
        assert not old_path.exists()
        assert _count(str(db_path), "files") == 1  # seed.py was recovered

    def test_phase2_intent_removes_old_file(self, tmp_path):
        """Intent 'phase2' must delete the .old backup and intent file."""
        db_path = tmp_path / "t.duckdb"
        old_path = tmp_path / "t.duckdb.old"
        intent_path = tmp_path / "t.duckdb.swap_intent"

        old_path.write_text("stale backup marker")
        intent_path.write_text("phase2")

        # Create a normal DB at db_path so open() succeeds
        bootstrap = _make_writer(str(db_path))
        bootstrap.open()
        bootstrap.close()

        w = _make_writer(str(db_path))
        w.open()
        w.close()

        assert not intent_path.exists()
        assert not old_path.exists()

    def test_pre_swap_intent_cleared_on_open_db_extension(self, tmp_path):
        """Crash-recovery intent path must be correct for a .db-extension path (not .duckdb).

        Regression guard for the set_extension() bug: PathBuf::set_extension() on
        'chunks.db' would produce 'chunks.duckdb.swap_intent' instead of
        'chunks.db.swap_intent'. The correct implementation uses string concatenation.
        """
        db_path = tmp_path / "chunks.db"
        # Intent path must be exactly "<db_path>.swap_intent"
        intent_path = tmp_path / "chunks.db.swap_intent"
        intent_path.write_text("pre-swap")

        w = _make_writer(str(db_path))
        w.open()
        w.close()

        assert not intent_path.exists(), (
            "intent file must be removed; wrong path construction would leave it untouched"
        )
        assert _count(str(db_path), "files") == 0

    def test_hnsw_crash_recovery_restores_missing_indexes(self, tmp_path):
        """Crash between HNSW drop (Step 2) and recreate (Step 5) is recovered on open().

        When a write_batch drops HNSW indexes before BEGIN and the process terminates
        before Step 5 recreates them, the DB is left with embeddings_N tables but no
        HNSW indexes — queries silently fall back to brute-force vector scan.

        open() must detect this and call ensure_all_hnsw_indexes() to restore them.
        """
        db = str(tmp_path / "t.duckdb")

        # Session 1: write >= 50 embeddings to trigger HNSW index creation
        w1 = _make_writer(db)
        w1.open()
        files = [_file(f"{i}.py", chunks=[_chunk(dims=128)]) for i in range(50)]
        w1.write_batch(_batch(files=files))
        w1.close()

        # Verify the HNSW index exists after session 1
        conn = duckdb.connect(db)
        try:
            try:
                conn.execute("LOAD vss")
                has_vss = True
            except Exception:
                has_vss = True  # VSS already loaded from prior test runs
            if has_vss:
                rows = conn.execute(
                    "SELECT index_name FROM duckdb_indexes() "
                    "WHERE table_name = 'embeddings_128' "
                    "AND schema_name = 'main'"
                ).fetchall()
                hnsw_names = [r[0] for r in rows if "hnsw" in r[0].lower()]
                assert hnsw_names, (
                    f"Expected HNSW index on embeddings_128 after session 1, "
                    f"got: {rows}"
                )
        finally:
            conn.close()

        # Simulate crash: manually drop the HNSW index outside the writer's lifecycle.
        # This mirrors the state after write_batch Step 2 (DROP HNSW) without Step 5.
        conn = duckdb.connect(db)
        try:
            try:
                conn.execute("LOAD vss")
            except Exception:
                pass
            conn.execute("DROP INDEX IF EXISTS idx_hnsw_128")
            conn.execute("CHECKPOINT")
            # Confirm the index is gone
            rows = conn.execute(
                "SELECT index_name FROM duckdb_indexes() "
                "WHERE table_name = 'embeddings_128' "
                "AND schema_name = 'main'"
            ).fetchall()
            hnsw_names = [r[0] for r in rows if "hnsw" in r[0].lower()]
            assert not hnsw_names, (
                f"Expected HNSW index to be absent after manual drop, "
                f"got: {rows}"
            )
        finally:
            conn.close()

        # Session 2: open() must detect the missing HNSW index and recreate it via
        # ensure_all_hnsw_indexes(), which is called unconditionally during open().
        w2 = _make_writer(db)
        w2.open()
        w2.close()

        # Verify the HNSW index has been restored
        conn = duckdb.connect(db)
        try:
            try:
                conn.execute("LOAD vss")
            except Exception:
                pass
            rows = conn.execute(
                "SELECT index_name FROM duckdb_indexes() "
                "WHERE table_name = 'embeddings_128' "
                "AND schema_name = 'main'"
            ).fetchall()
            hnsw_names = [r[0] for r in rows if "hnsw" in r[0].lower()]
            assert hnsw_names, (
                f"HNSW index not restored after crash recovery: {rows}"
            )
            assert _count(db, "embeddings_128") == 50, (
                "embeddings_128 data must survive the simulated crash"
            )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# HNSW threshold boundary (Invariant 14)
# ---------------------------------------------------------------------------

class TestHnswBoundary:
    def test_below_threshold_no_hnsw_lifecycle(self, tmp_path):
        """Batches with < 50 embeddings must not trigger HNSW drop/recreate path."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db)
        w.open()
        # 49 embeddings — below 50 threshold
        files = [_file(f"{i}.py", chunks=[_chunk(dims=128)]) for i in range(49)]
        result = w.write_batch(_batch(files=files))
        w.close()

        assert result["embeddings_written"] == 49
        assert _count(db, "embeddings_128") == 49

    def test_at_threshold_triggers_hnsw_lifecycle(self, tmp_path):
        """Batches with >= 50 embeddings must complete successfully through HNSW lifecycle."""
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db)
        w.open()
        # 50 embeddings — at threshold
        files = [_file(f"{i}.py", chunks=[_chunk(dims=128)]) for i in range(50)]
        result = w.write_batch(_batch(files=files))
        w.close()

        assert result["embeddings_written"] == 50
        assert _count(db, "embeddings_128") == 50

    def test_new_dims_mid_session_gets_hnsw(self, tmp_path):
        """A second embedding dimension introduced mid-session must get its HNSW index.

        Regression for the hnsw_cache staleness bug: once the cache is primed for
        dims=128, a batch with a new dims=64 must (a) invalidate the cache so Step 2
        rediscovers all indexes next time, and (b) create an HNSW index for
        embeddings_64 after the commit via Step 5+.  Without the fix, embeddings_64
        would never receive an HNSW index and queries would fall back to a table scan.
        """
        db = str(tmp_path / "t.duckdb")
        w = _make_writer(db)
        w.open()

        # Batch 1: 50 × 128-dim — above threshold, HNSW lifecycle fires, cache primed.
        files1 = [_file(f"a{i}.py", chunks=[_chunk(dims=128)]) for i in range(50)]
        r1 = w.write_batch(_batch(files=files1))

        # Batch 2: 50 × 64-dim — new dimension; writer must invalidate cache and
        # create an HNSW index for embeddings_64 after committing.
        files2 = [_file(f"b{i}.py", chunks=[_chunk(dims=64)]) for i in range(50)]
        r2 = w.write_batch(_batch(files=files2))

        # Batch 3: another 50 × 64-dim — exercises the normal per-batch HNSW cycle
        # for the newly-indexed dimension (drop existing idx_hnsw_64, write, recreate).
        files3 = [_file(f"c{i}.py", chunks=[_chunk(dims=64)]) for i in range(50)]
        r3 = w.write_batch(_batch(files=files3))

        w.close()

        assert r1["embeddings_written"] == 50
        assert r2["embeddings_written"] == 50
        assert r3["embeddings_written"] == 50
        assert _count(db, "embeddings_128") == 50
        assert _count(db, "embeddings_64") == 100

        # When VSS is available verify both tables have an HNSW index.
        conn = duckdb.connect(db)
        try:
            try:
                conn.execute("LOAD vss")
                has_vss = True
            except Exception:
                has_vss = False

            if has_vss:
                rows = conn.execute(
                    "SELECT table_name, index_name FROM duckdb_indexes() "
                    "WHERE table_name SIMILAR TO 'embeddings_[0-9]+' "
                    "AND schema_name = 'main'"
                ).fetchall()
                tables_with_hnsw = {
                    r[0] for r in rows if "hnsw" in r[1].lower()
                }
                assert "embeddings_128" in tables_with_hnsw, (
                    f"HNSW missing for embeddings_128: {rows}"
                )
                assert "embeddings_64" in tables_with_hnsw, (
                    f"HNSW missing for embeddings_64 (cache staleness regression): {rows}"
                )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Feature flag (pipeline_bridge)
# ---------------------------------------------------------------------------

class TestFeatureFlag:
    def test_rust_disabled_via_env(self, monkeypatch):
        """CHUNKHOUND_USE_RUST=0 must make _get_use_rust() return False."""
        monkeypatch.setenv("CHUNKHOUND_USE_RUST", "0")

        from chunkhound.providers.database.pipeline_bridge import _get_use_rust

        assert _get_use_rust() is False

    def test_rust_enabled_by_default_when_native_present(self, monkeypatch):
        """Without override, _get_use_rust() mirrors native extension availability."""
        monkeypatch.delenv("CHUNKHOUND_USE_RUST", raising=False)

        from chunkhound.providers.database.pipeline_bridge import _get_use_rust

        try:
            import chunkhound_native  # noqa: F401
            native_present = True
        except ImportError:
            native_present = False

        assert _get_use_rust() == native_present


# ---------------------------------------------------------------------------
# Schema parity (guards against DDL drift between Rust and schema_constants.py)
# ---------------------------------------------------------------------------

class TestSchemaParity:
    def test_files_and_chunks_columns_match_schema_constants(self, tmp_path):
        """Rust-created tables must have the same columns (in order) as schema_constants.py.

        This is the CI guard for the DDL duplication between duckdb_backend.rs::setup_schema
        and chunkhound/providers/database/duckdb/schema_constants.py.  When a column is
        added or renamed in schema_constants.py, this test will fail until duckdb_backend.rs
        is updated to match.
        """
        # pylint: disable=protected-access  (accessing private module constants intentionally)
        from chunkhound.providers.database.duckdb.schema_constants import (
            _FILES_COLUMN_NAMES,
            _CHUNKS_COLUMN_NAMES,
        )

        db = str(tmp_path / "parity.duckdb")
        w = _make_writer(db)
        w.open()
        w.close()

        conn = duckdb.connect(db)
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

    def test_column_types_match_python_schema(self, tmp_path):
        """Rust DDL column types must match what DuckDB reports for the Python canonical DDL.

        Catches type/constraint drift that test_files_and_chunks_columns_match_schema_constants
        cannot detect — e.g. changing INTEGER to BIGINT, or TEXT to VARCHAR explicitly.
        Uses the Python DDL as the authoritative reference: both sides are created via
        DuckDB and compared through information_schema, so no hardcoded type strings are
        needed and the test stays valid across DuckDB versions.
        """
        from chunkhound.providers.database.duckdb.schema_constants import (
            _FILES_TABLE_COLUMNS,
            _CHUNKS_TABLE_COLUMNS,
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
        py_db = str(tmp_path / "python_ref.duckdb")
        ref_conn = duckdb.connect(py_db)
        try:
            ref_conn.execute("CREATE SEQUENCE IF NOT EXISTS files_id_seq START 1")
            ref_conn.execute("CREATE SEQUENCE IF NOT EXISTS chunks_id_seq START 1")
            ref_conn.execute(f"CREATE TABLE files ({_FILES_TABLE_COLUMNS})")
            ref_conn.execute(f"CREATE TABLE chunks ({_CHUNKS_TABLE_COLUMNS})")
            py_files_types = _col_types(ref_conn, "files")
            py_chunks_types = _col_types(ref_conn, "chunks")
        finally:
            ref_conn.close()

        # Subject: create tables using the Rust DDL via RustDbWriter.open().
        rust_db = str(tmp_path / "rust_parity.duckdb")
        w = _make_writer(rust_db)
        w.open()
        w.close()

        rust_conn = duckdb.connect(rust_db)
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
