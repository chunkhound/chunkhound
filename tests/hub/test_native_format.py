"""Native pipeline format ownership cannot bypass Python's format policy."""

import shutil

import duckdb
import pytest

from chunkhound.core.index_format import INDEX_FORMAT_VERSION
from tests.contracts.pipeline_harness import index_with_rust
from tests.hub.test_index_format import snapshot


def test_native_writer_rejects_future_format_before_ddl(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("def retained(): return 7\n")
    db = tmp_path / "db"
    index_with_rust(source, db, skip_embeddings=True)
    path = db / "chunks.db"
    with duckdb.connect(str(path)) as conn:
        conn.execute("UPDATE schema_version SET version=2")
    before = snapshot(tmp_path)
    try:
        report = index_with_rust(source, db, skip_embeddings=True)
    except Exception as exc:
        assert "index format" in str(exc).lower()
    else:
        assert any("index format" in error.lower() for error in report.errors)
    assert snapshot(tmp_path) == before


def test_native_pipeline_rejects_future_candidate_before_diff_recovery(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("def retained(): return 7\n")
    db = tmp_path / "db"
    report = index_with_rust(source, db, skip_embeddings=True)
    assert not report.errors
    path = db / "chunks.db"
    candidate = db / "chunks.db.old"
    shutil.copy2(path, candidate)
    with duckdb.connect(str(candidate)) as conn:
        conn.execute("UPDATE schema_version SET version=2")
    (db / "chunks.db.swap_intent").write_text("phase2")
    before = snapshot(tmp_path)

    try:
        report = index_with_rust(source, db, skip_embeddings=True)
    except Exception as exc:
        errors = [str(exc)]
    else:
        errors = report.errors

    assert snapshot(tmp_path) == before, "diff recovery changed the future candidate"
    assert any("unsupported index format" in error.lower() for error in errors)


@pytest.mark.parametrize(
    "metadata",
    [
        "SELECT 0 AS version",
        "SELECT -1 AS version",
        "SELECT NULL::INTEGER AS version",
        "SELECT '1' AS version",
        "SELECT true AS version",
        "SELECT 1.0 AS version",
        "SELECT 1.5 AS version",
        "SELECT 1 AS missing_version",
        "SELECT unnest([1, 2]) AS version",
    ],
)
def test_native_writer_refuses_invalid_metadata_without_changes(tmp_path, metadata):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("def retained(): return 7\n")
    db = tmp_path / "db"
    index_with_rust(source, db, skip_embeddings=True)
    with duckdb.connect(str(db / "chunks.db")) as conn:
        conn.execute("DROP TABLE schema_version")
        conn.execute(
            f"CREATE TABLE schema_version AS SELECT *, 'retained' AS description "
            f"FROM ({metadata})"
        )
    before = snapshot(tmp_path)
    try:
        report = index_with_rust(source, db, skip_embeddings=True)
    except Exception as exc:
        assert "index format" in str(exc) or "format compatibility is unknown" in str(
            exc
        )
    else:
        assert any("index format" in error.lower() for error in report.errors)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("state", ["new", "absent", "empty", "supported"])
def test_native_writer_stamps_only_new_or_legacy_metadata(tmp_path, state):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("def retained(): return 7\n")
    db = tmp_path / "db"
    db.mkdir()
    path = db / "chunks.db"
    if state != "new":
        index_with_rust(source, db, skip_embeddings=True)
        with duckdb.connect(str(path)) as conn:
            if state == "absent":
                conn.execute("DROP TABLE schema_version")
            elif state == "empty":
                conn.execute("DELETE FROM schema_version")
            else:
                conn.execute("UPDATE schema_version SET description='retained'")
    report = index_with_rust(source, db, skip_embeddings=True)
    assert not report.errors
    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [
            (INDEX_FORMAT_VERSION,)
        ]
        expected = "retained" if state == "supported" else "ChunkHound index format"
        assert conn.execute("SELECT description FROM schema_version").fetchone() == (
            expected,
        )
