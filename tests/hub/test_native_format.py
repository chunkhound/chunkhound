"""Native pipeline format ownership cannot bypass Python's format policy."""

import shutil

import duckdb
import pytest

from chunkhound.core.index_format import INDEX_FORMAT_VERSION, IndexFormatError
from chunkhound.providers.database.index_format import preflight_duckdb, read_format
from tests.contracts.pipeline_harness import index_with_rust
from tests.hub.test_index_format import snapshot


@pytest.mark.parametrize("suffix", ["", ".old"])
@pytest.mark.parametrize(
    "metadata", ["2", "0", "'1'", "view", "external-view", "generated"]
)
def test_native_case_variant_metadata_refused_before_recovery(
    tmp_path, suffix, metadata
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("def retained(): return 7\n")
    db = tmp_path / "db"
    assert not index_with_rust(source, db, skip_embeddings=True).errors
    path = db / "chunks.db"
    candidate = path.with_name(path.name + suffix)
    if suffix:
        shutil.copy2(path, candidate)
        (db / "chunks.db.swap_intent").write_text("phase2")
    outside = tmp_path / "outside.csv"
    outside.write_text("version\n1\n")
    with duckdb.connect(str(candidate)) as conn:
        conn.execute("DROP TABLE schema_version")
        if metadata == "generated":
            conn.execute(
                'CREATE TABLE "ScHeMa_VeRsIoN"(seed INTEGER, version INTEGER '
                "GENERATED ALWAYS AS (CASE WHEN current_database()='memory' "
                "THEN 1 ELSE 2 END), description VARCHAR)"
            )
            conn.execute(
                "INSERT INTO schema_version(seed, description) VALUES (1, 'generated')"
            )
            assert conn.execute("SELECT version FROM schema_version").fetchone() == (2,)
        elif "view" in metadata:
            query = (
                f"SELECT version FROM read_csv('{outside}', header=true)"
                if metadata == "external-view"
                else "SELECT 1 AS version"
            )
            conn.execute(f'CREATE VIEW "ScHeMa_VeRsIoN" AS {query}')
        else:
            conn.execute(
                f'CREATE TABLE "ScHeMa_VeRsIoN" AS SELECT {metadata} AS version, '
                "'retained' AS description"
            )
    if metadata == "external-view":
        outside.unlink()
    (source / "b.py").write_text("def added(): return 8\n")
    before = snapshot(tmp_path)
    try:
        report = index_with_rust(source, db, skip_embeddings=True)
    except Exception as exc:
        errors = [str(exc)]
    else:
        errors = report.errors
    expected = (
        "stored column"
        if metadata == "generated"
        else "physical table"
        if "view" in metadata
        else "index format"
    )
    assert any(expected in error.lower() for error in errors), errors
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("suffix", ["", ".old"])
@pytest.mark.parametrize("version", [1, 2])
def test_native_pipeline_ignores_persistent_format_helper_macros(
    tmp_path, suffix, version
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("def retained(): return 7\n")
    db = tmp_path / "db"
    assert not index_with_rust(source, db, skip_embeddings=True).errors
    path = db / "chunks.db"
    candidate = path.with_name(path.name + suffix)
    if suffix:
        shutil.copy2(path, candidate)
        (db / "chunks.db.swap_intent").write_text("phase2")
    with duckdb.connect(str(candidate)) as conn:
        conn.execute(f"UPDATE schema_version SET version={version}")
        conn.execute("CREATE MACRO translate(a,b,c) AS 'hidden'")
        conn.execute("CREATE MACRO current_database() AS 'absent'")
        conn.execute("CREATE MACRO typeof(a) AS 'INTEGER'")
        conn.execute("CREATE MACRO union_tag(a) AS 'i'")
        conn.execute("CREATE MACRO union_extract(a,b) AS 1")
    (source / "b.py").write_text("def added(): return 8\n")
    before = snapshot(tmp_path)
    if version == 1:
        report = index_with_rust(source, db, skip_embeddings=True)
        assert not report.errors
        assert "b.py" in {chunk[0] for chunk in report.chunk_tuples}
    else:
        try:
            report = index_with_rust(source, db, skip_embeddings=True)
        except Exception as exc:
            errors = [str(exc)]
        else:
            errors = report.errors
        assert any("unsupported index format" in error.lower() for error in errors)
        assert snapshot(tmp_path) == before


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


@pytest.mark.parametrize(
    "member, supported",
    [
        pytest.param("1", True, id="integer"),
        pytest.param("1::HUGEINT", True, id="hugeint"),
        pytest.param("1::UHUGEINT", True, id="uhugeint"),
        pytest.param('union_value("quo\'ted" := 1)', True, id="nested-quoted-tag"),
        pytest.param("true", False, id="boolean"),
        pytest.param("1::DECIMAL(38,0)", False, id="scale-zero-decimal"),
        pytest.param("1.5", False, id="fractional-decimal"),
        pytest.param("'1'", False, id="text"),
        pytest.param("NULL::INTEGER", False, id="null-member"),
        pytest.param("0", False, id="zero"),
        pytest.param("-1", False, id="negative"),
        pytest.param("2", False, id="future"),
        pytest.param(
            "340282366920938463463374607431768211455::UHUGEINT",
            False,
            id="full-width-future",
        ),
    ],
)
def test_native_pipeline_matches_python_union_values(tmp_path, member, supported):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("def retained(): return 7\n")
    db = tmp_path / "db"
    assert not index_with_rust(source, db, skip_embeddings=True).errors
    path = db / "chunks.db"
    with duckdb.connect(str(path)) as conn:
        conn.execute("DROP TABLE schema_version")
        # Refusals mix a valid first row with an invalid/future second row.
        # VARCHAR stays inactive so it must not invalidate supported members.
        type_row = conn.execute(f"SELECT typeof({member})").fetchone()
        assert type_row is not None
        member_type = type_row[0]
        union_type = f"UNION(inactive VARCHAR, other {member_type}"
        if supported:
            first = f"union_value(other := {member})"
        else:
            first = "union_value(i := 1)"
            union_type += ", i INTEGER"
        # Existing schema-stamp SQL binds an INTEGER even for nonempty tables;
        # provide its implicit target when UHUGEINT cannot accept that cast.
        if supported and member_type == "UHUGEINT":
            union_type += ", stamp INTEGER"
        union_type += ")"
        conn.execute(
            "CREATE TABLE schema_version AS "
            "SELECT version, 'retained' AS description FROM ("
            f"SELECT {first}::{union_type} AS version UNION ALL "
            f"SELECT union_value(other := {member})::{union_type})"
        )
    (source / "b.py").write_text("def added(): return 8\n")
    before = snapshot(tmp_path)

    if supported:
        assert read_format(path, "duckdb") == INDEX_FORMAT_VERSION
        preflight_duckdb(path)
    else:
        with pytest.raises(IndexFormatError, match="index format"):
            preflight_duckdb(path)
    assert snapshot(tmp_path) == before, "Python inspection mutated the index"

    if supported:
        report = index_with_rust(source, db, skip_embeddings=True)
        assert not report.errors
        assert "b.py" in {chunk[0] for chunk in report.chunk_tuples}
        with duckdb.connect(str(path), read_only=True) as conn:
            assert conn.execute(
                "SELECT version, description FROM schema_version"
            ).fetchall() == [(1, "retained"), (1, "retained")]
    else:
        try:
            report = index_with_rust(source, db, skip_embeddings=True)
        except Exception as exc:
            errors = [str(exc)]
        else:
            errors = report.errors
        assert any("index format" in error.lower() for error in errors)
        assert snapshot(tmp_path) == before, "native refusal mutated the index"


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
