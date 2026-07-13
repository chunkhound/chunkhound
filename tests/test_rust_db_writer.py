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
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
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
    def test_rust_disabled_via_env(self, monkeypatch, tmp_path):
        """CHUNKHOUND_USE_RUST=0 must make RustWriterBridge.available() return False.

        _get_use_rust() is evaluated at instantiation time, so monkeypatching the
        env var before constructing the bridge is sufficient — no module reload needed.
        """
        monkeypatch.setenv("CHUNKHOUND_USE_RUST", "0")

        from chunkhound.services.pipeline_bridge import RustWriterBridge

        b = RustWriterBridge(
            {"db_path": str(tmp_path / "t.duckdb"), "compaction_batch_threshold": 50}
        )
        assert b.available() is False

    def test_rust_enabled_by_default_when_native_present(self, monkeypatch, tmp_path):
        """With native extension installed and no override, bridge must be available."""
        monkeypatch.delenv("CHUNKHOUND_USE_RUST", raising=False)

        from chunkhound.services.pipeline_bridge import RustWriterBridge

        b = RustWriterBridge(
            {"db_path": str(tmp_path / "t.duckdb"), "compaction_batch_threshold": 50}
        )
        # available() iff native extension is importable AND CHUNKHOUND_USE_RUST != "0"
        try:
            import chunkhound_native  # noqa: F401
            native_present = True
        except ImportError:
            native_present = False

        assert b.available() == native_present

        if b.available():
            b.finalize()
