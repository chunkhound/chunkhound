"""Real-backend contracts for owned index formats and nonmutating inspection."""

import hashlib
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import duckdb
import lancedb
import pyarrow as pa
import pytest

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.index_format import IndexFormatError
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.providers.database.index_format import read_format
from chunkhound.providers.database.lancedb_provider import LanceDBProvider


def snapshot(
    path: Path,
    *,
    live_database_paths: tuple[Path | str, ...] = (),
    live_wal_paths: tuple[Path | str, ...] = (),
) -> dict[str, tuple[str, int] | tuple[str, int, int, int, int]]:
    """Hash strictly, except explicitly named live Windows primaries and WALs.

    DuckDB's Windows writer denies reads of its primary and WAL. An opted-in
    WAL must be adjacent to a declared live primary. Metadata plus callers'
    live-state assertions does NOT prove byte identity: timestamps may lag,
    and same-size, timestamp-preserved writes can go undetected. Unowned WALs,
    recovery files, closed databases and all non-Windows files retain hashes.
    """
    live_paths = {Path(p).resolve() for p in live_database_paths}
    live_wals = {Path(p).resolve() for p in live_wal_paths}
    adjacent_wals = {Path(str(p) + ".wal").resolve() for p in live_database_paths}
    if not live_wals <= adjacent_wals:
        raise ValueError(
            "Each live WAL must be the adjacent WAL of a declared live database"
        )
    files = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )
    result: dict[str, tuple[str, int] | tuple[str, int, int, int, int]] = {}
    for p in files:
        if p.name.endswith(".chunkhound.lock"):
            continue
        if os.name == "nt" and p.resolve() in live_paths | live_wals:
            stat = p.stat()
            result[str(p)] = (
                "live-windows-wal-metadata"
                if p.resolve() in live_wals
                else "live-windows-primary-metadata",
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_dev,
                stat.st_ino,
            )
        else:
            result[str(p)] = (
                hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns
            )
    return result


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("version", [2, 0, -1, None, "1", True, 1.5])
def test_unsupported_format_rejected_without_mutating_database(
    tmp_path, backend, version
):
    path = tmp_path / ("chunks.db" if backend == "duckdb" else "lancedb.lancedb")
    if backend == "duckdb":
        with duckdb.connect(str(path)) as conn:
            conn.execute(
                "CREATE TABLE schema_version AS SELECT ? AS version", [version]
            )
    else:
        conn = lancedb.connect(str(path))
        conn.create_table("schema_version", data=pa.table({"version": [version]}))
        del conn
    before = snapshot(tmp_path)
    cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
    provider = cls(path, base_directory=tmp_path)
    with pytest.raises(IndexFormatError):
        provider.connect()
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("empty", [False, True])
def test_unstamped_format_is_pinned_to_legacy_one(
    tmp_path, monkeypatch, backend, empty
):
    import chunkhound.core.index_format as versions

    path = tmp_path / ("chunks.db" if backend == "duckdb" else "lancedb.lancedb")
    if backend == "duckdb":
        with duckdb.connect(str(path)) as conn:
            conn.execute("CREATE TABLE files(id INTEGER)")
            if empty:
                conn.execute("CREATE TABLE schema_version(version INTEGER)")
    else:
        conn = lancedb.connect(str(path))
        if empty:
            conn.create_table(
                "schema_version", schema=pa.schema([("version", pa.int64())])
            )
        del conn
    monkeypatch.setattr(versions, "INDEX_FORMAT_VERSION", 2)
    before = snapshot(tmp_path)
    assert read_format(path, backend) == 1
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_new_provider_records_owned_format(tmp_path, backend):
    path = tmp_path / ("chunks.db" if backend == "duckdb" else "lancedb.lancedb")
    cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
    provider = cls(path, base_directory=tmp_path)
    provider.connect()
    provider.disconnect()
    assert read_format(path, backend) == 1
    if backend == "duckdb":
        with duckdb.connect(str(path), read_only=True) as conn:
            assert (
                conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
                == 1
            )
    else:
        conn = lancedb.connect(str(path))
        assert conn.open_table("schema_version").to_arrow()["version"].to_pylist() == [
            1
        ]


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("empty", [False, True])
def test_legacy_metadata_is_stamped_only_by_writable_open(tmp_path, backend, empty):
    path = tmp_path / ("chunks.db" if backend == "duckdb" else "lancedb.lancedb")
    cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
    db = cls(path, base_directory=tmp_path)
    db.connect()
    db.disconnect()
    if backend == "duckdb":
        with duckdb.connect(str(path)) as conn:
            conn.execute(
                "DELETE FROM schema_version" if empty else "DROP TABLE schema_version"
            )
        before = snapshot(tmp_path)
        reader = DuckDBProvider(
            path, base_directory=tmp_path, config=DatabaseConfig(read_only=True)
        )
        reader.connect()
        reader.disconnect()
        assert snapshot(tmp_path) == before
    else:
        conn = lancedb.connect(str(path))
        if empty:
            conn.open_table("schema_version").delete("true")
        else:
            conn.drop_table("schema_version")
        del conn
    before = snapshot(tmp_path)
    assert read_format(path, backend) == 1
    assert snapshot(tmp_path) == before
    db = cls(path, base_directory=tmp_path)
    db.connect()
    db.disconnect()
    if backend == "duckdb":
        with duckdb.connect(str(path), read_only=True) as conn:
            assert conn.execute("SELECT version FROM schema_version").fetchall() == [
                (1,)
            ]
    else:
        conn = lancedb.connect(str(path))
        assert conn.open_table("schema_version").to_arrow()["version"].to_pylist() == [
            1
        ]


@pytest.mark.parametrize(
    "suffix", [".compact_backup", ".compact_new", ".old", ".compact"]
)
def test_future_recovery_artifact_rejected_before_any_file_changes(tmp_path, suffix):
    path = tmp_path / "chunks.db"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE schema_version(version INTEGER)")
        conn.execute("INSERT INTO schema_version VALUES (1)")
    with duckdb.connect(str(path) + suffix) as conn:
        conn.execute("CREATE TABLE schema_version(version INTEGER)")
        conn.execute("INSERT INTO schema_version VALUES (1), (2)")
    before = snapshot(tmp_path)
    provider = DuckDBProvider(path, base_directory=tmp_path)
    with pytest.raises(IndexFormatError):
        provider.connect()
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_inspection_does_not_create_missing_paths(tmp_path, backend):
    path = tmp_path / "missing" / "database"
    if backend == "duckdb":
        with pytest.raises(IndexFormatError, match="format compatibility is unknown"):
            read_format(path, backend)
    else:
        assert read_format(path, backend) == 1
    assert not path.parent.exists()


@pytest.mark.parametrize("name", ["SCHEMA_VERSION", "sChEmA_vErSiOn"])
@pytest.mark.parametrize("version", [2, 0, "1"])
@pytest.mark.parametrize("suffix", ["", ".old"])
def test_case_variant_metadata_blocks_provider_before_mutation(
    tmp_path, name, version, suffix
):
    path = tmp_path / "chunks.db"
    with duckdb.connect(str(path) + suffix) as conn:
        conn.execute(
            f'CREATE TABLE "{name}" AS SELECT ? AS version, '
            "'retained' AS description",
            [version],
        )
    before = snapshot(tmp_path)
    provider = DuckDBProvider(path, base_directory=tmp_path)
    try:
        with pytest.raises(IndexFormatError, match="index format"):
            provider.connect()
    finally:
        provider.disconnect(skip_checkpoint=True)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("external", [False, True])
def test_metadata_view_is_rejected_before_evaluation(tmp_path, external):
    path = tmp_path / "chunks.db"
    outside = tmp_path / "outside.csv"
    outside.write_text("version\n1\n")
    with duckdb.connect(str(path)) as conn:
        query = (
            f"SELECT version FROM read_csv('{outside}', header=true)"
            if external
            else "SELECT 1 AS version"
        )
        conn.execute(f'CREATE VIEW "ScHeMa_VeRsIoN" AS {query}')
    # A missing dependency would raise an engine error if the view were bound.
    # The contract requires our object-kind refusal instead, before evaluation.
    if external:
        outside.unlink()
    before = snapshot(tmp_path)
    with pytest.raises(IndexFormatError, match="physical table") as caught:
        read_format(path, "duckdb")
    assert caught.value.__cause__ is None
    provider = DuckDBProvider(path, base_directory=tmp_path)
    try:
        with pytest.raises(IndexFormatError, match="physical table"):
            provider.connect()
    finally:
        provider.disconnect(skip_checkpoint=True)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("suffix", ["", ".old"])
@pytest.mark.parametrize("wal", [False, True])
def test_generated_format_refused_before_python_recovery(
    tmp_path, read_only, suffix, wal
):
    path = tmp_path / "chunks.db"
    with duckdb.connect(str(path) + ".compact_backup") as conn:
        conn.execute("CREATE TABLE schema_version AS SELECT 1 AS version")
    with duckdb.connect(str(path) + suffix) as conn:
        conn.execute(
            "CREATE TABLE schema_version(seed INTEGER, version INTEGER "
            "GENERATED ALWAYS AS (CASE WHEN current_database()='memory' "
            "THEN 1 ELSE 2 END), description VARCHAR)"
        )
        conn.execute(
            "INSERT INTO schema_version(seed, description) VALUES (1, 'generated')"
        )
        assert conn.execute("SELECT version FROM schema_version").fetchone() == (2,)
    if wal:
        run_python(
            """
            import os, sys, duckdb
            conn = duckdb.connect(sys.argv[1])
            conn.execute("UPDATE schema_version SET seed=2")
            os._exit(0)
            """,
            Path(str(path) + suffix),
        )
        assert Path(str(path) + suffix + ".wal").stat().st_size > 0
    (tmp_path / "chunks.db.swap_intent").write_text("phase2")
    before = snapshot(tmp_path)
    with pytest.raises(IndexFormatError, match="stored column") as error:
        read_format(Path(str(path) + suffix), "duckdb")
    assert error.value.__cause__ is None, "generated expression was evaluated"
    provider = DuckDBProvider(
        path, base_directory=tmp_path, config=DatabaseConfig(read_only=read_only)
    )
    try:
        with pytest.raises(IndexFormatError, match="stored column"):
            provider.connect()
        assert snapshot(tmp_path) == before
    finally:
        provider.disconnect(skip_checkpoint=True)


def test_live_preflight_refuses_handle_for_another_database(tmp_path):
    from chunkhound.providers.database.index_format import preflight_duckdb

    path = tmp_path / "owned.db"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE schema_version AS SELECT 2 AS version")
    unrelated = tmp_path / "unrelated.db"
    with duckdb.connect(str(unrelated)) as other:
        other.execute("CREATE TABLE schema_version AS SELECT 1 AS version")
        before = snapshot(
            tmp_path,
            live_database_paths=(unrelated,),
            live_wal_paths=(str(unrelated) + ".wal",),
        )
        with pytest.raises(IndexFormatError, match="catalog identity is ambiguous"):
            preflight_duckdb(path, live_connection=other)
        assert snapshot(
            tmp_path,
            live_database_paths=(unrelated,),
            live_wal_paths=(str(unrelated) + ".wal",),
        ) == before
        assert other.execute("SELECT version FROM schema_version").fetchone() == (1,)


@pytest.mark.parametrize("version", [1, 2])
def test_live_preflight_checks_attached_catalog_with_embedded_quote(tmp_path, version):
    from chunkhound.providers.database.index_format import preflight_duckdb

    path = tmp_path / "owned'quoted.db"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE schema_version AS SELECT ? AS version", [version])
    before = snapshot(tmp_path)
    with duckdb.connect(":memory:") as conn:
        escaped_path = str(path).replace("'", "''")
        conn.execute(f'''ATTACH '{escaped_path}' AS "owned""quoted" (READ_ONLY)''')
        conn.execute("CREATE TEMP TABLE retained_session AS SELECT 42 AS value")
        if version == 2:
            with pytest.raises(IndexFormatError, match="Unsupported index format"):
                preflight_duckdb(path, live_connection=conn)
        else:
            preflight_duckdb(path, live_connection=conn)
        assert conn.execute(
            'SELECT version FROM "owned""quoted".main.schema_version'
        ).fetchone() == (version,)
        assert conn.execute("SELECT value FROM retained_session").fetchone() == (42,)
        assert snapshot(tmp_path) == before
    assert snapshot(tmp_path) == before


def test_stored_format_defaults_are_not_generated_columns(tmp_path):
    path = tmp_path / "chunks.db"
    with duckdb.connect(str(path)) as conn:
        conn.execute(
            "CREATE TABLE schema_version(version INTEGER DEFAULT 1, "
            "description VARCHAR DEFAULT 'GENERATED ALWAYS AS', "
            '"GENERATED ALWAYS AS" INTEGER DEFAULT 3)'
        )
        conn.execute("INSERT INTO schema_version DEFAULT VALUES")
    before = snapshot(tmp_path)
    assert read_format(path, "duckdb") == 1
    assert snapshot(tmp_path) == before


def test_case_variant_supported_metadata_is_read_not_restamped(tmp_path):
    path = tmp_path / "chunks.db"
    with duckdb.connect(str(path)) as conn:
        conn.execute(
            'CREATE TABLE "ScHeMa_VeRsIoN" AS '
            "SELECT 1 AS version, 'retained' AS description"
        )
    before = snapshot(tmp_path)
    assert read_format(path, "duckdb") == 1
    assert snapshot(tmp_path) == before
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    provider.disconnect()
    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute("SELECT * FROM schema_version").fetchall() == [
            (1, "retained")
        ]


def run_python(script: str, *args: object) -> None:
    """Contain native WAL replay in a process, without installing dependencies."""
    result = subprocess.run(
        ["uv", "run", "python", "-c", textwrap.dedent(script), *map(str, args)],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "UV_NO_SYNC": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def make_wal_database(path: Path, kind: str, future: bool = False) -> None:
    """Create committed ordinary/HNSW WAL without close-time checkpointing."""
    run_python(
        """
        import os, sys
        import duckdb

        path, kind, future = sys.argv[1:]
        conn = duckdb.connect(path, config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
        })
        conn.execute("SET wal_autocheckpoint = '1GB'")
        conn.execute("CREATE TABLE schema_version(version INTEGER)")
        conn.execute("INSERT INTO schema_version VALUES (1)")
        conn.execute("CREATE TABLE ordinary(id INTEGER)")
        if kind.startswith("hnsw"):
            conn.execute("LOAD vss")
            conn.execute("SET hnsw_enable_experimental_persistence = true")
            conn.execute("CREATE TABLE vectors(id INTEGER, vec FLOAT[3])")
            conn.execute("INSERT INTO vectors VALUES (0, [0, 0, 0])")
            if kind != "hnsw_create":
                conn.execute("CREATE INDEX hnsw_idx ON vectors USING HNSW(vec)")
        conn.execute("CHECKPOINT")
        if kind.endswith("checkpoint"):
            conn.close()
            sys.exit(0)
        conn.execute("INSERT INTO ordinary VALUES (1)")
        if kind.startswith("hnsw"):
            if kind == "hnsw_create":
                conn.execute("CREATE INDEX hnsw_idx ON vectors USING HNSW(vec)")
            conn.execute("INSERT INTO vectors VALUES (1, [1, 1, 1])")
        if future == "True":
            conn.execute("INSERT INTO schema_version VALUES (2)")
        os._exit(0)
        """,
        path,
        kind,
        future,
    )
    wal = Path(str(path) + ".wal")
    if kind.endswith("checkpoint"):
        assert not wal.exists()
    else:
        assert wal.stat().st_size > 0


@pytest.mark.parametrize("kind", ["ordinary", "hnsw_insert"])
def test_committed_wal_format_is_read_without_changing_files(tmp_path, kind):
    path = tmp_path / "chunks' quoted.db"
    make_wal_database(path, kind, future=True)
    before = snapshot(tmp_path)
    run_python(
        """
        import sys
        from pathlib import Path
        from chunkhound.providers.database.index_format import read_format

        assert read_format(Path(sys.argv[1]), "duckdb") == 2
        """,
        path,
    )
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("suffix", ["", ".compact_backup"])
@pytest.mark.parametrize("read_only", [False, True])
def test_future_wal_rejects_open_before_recovery(tmp_path, suffix, read_only):
    path = tmp_path / "chunks.db"
    if suffix:
        make_wal_database(path, "ordinary_checkpoint")
    make_wal_database(Path(str(path) + suffix), "ordinary", future=True)
    before = snapshot(tmp_path)
    run_python(
        """
        import sys
        from pathlib import Path
        import pytest
        from chunkhound.core.config.database_config import DatabaseConfig
        from chunkhound.core.index_format import IndexFormatError
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        path = Path(sys.argv[1])
        db = DuckDBProvider(path, base_directory=path.parent,
            config=DatabaseConfig(read_only=sys.argv[2] == "True"))
        with pytest.raises(IndexFormatError, match="Unsupported index format"):
            db.connect()
        """,
        path,
        read_only,
    )
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "kind", ["ordinary_checkpoint", "hnsw_checkpoint", "ordinary", "unreadable"]
)
def test_missing_local_vss_only_blocks_wal_inspection(tmp_path, kind):
    path = tmp_path / "chunks.db"
    make_wal_database(path, kind)
    if kind == "unreadable":
        # Failure must still be at local VSS setup, not ATTACH parsing the file.
        path.write_bytes(b"unreadable database with a WAL")
    before = snapshot(tmp_path)
    missing_extensions = tmp_path / "absent_extensions"
    run_python(
        """
        import sys
        from pathlib import Path
        import duckdb
        import pytest
        from chunkhound.core.index_format import IndexFormatError
        from chunkhound.providers.database.index_format import read_format

        path, missing, kind = sys.argv[1:]
        connect = duckdb.connect
        # Isolate the real external connection's extension directory. No fake
        # connection/query results and no changes to the shared VSS installation.
        def isolated_connect(*args, **kwargs):
            config = dict(kwargs.pop("config", {}))
            config["extension_directory"] = missing
            return connect(*args, config=config, **kwargs)
        duckdb.connect = isolated_connect
        if kind.endswith("checkpoint"):
            assert read_format(Path(path), "duckdb") == 1
        else:
            with pytest.raises(IndexFormatError) as caught:
                read_format(Path(path), "duckdb")
            message = str(caught.value)
            assert "WAL" in message and "VSS" in message
            assert "explicit setup" in message
            assert "format compatibility is unknown" in message
            # A real failed LOAD, not a later ATTACH/replay failure.
            assert isinstance(caught.value.__cause__, duckdb.Error)
            assert "vss" in str(caught.value.__cause__).lower()
            assert "replaying WAL" not in str(caught.value.__cause__)
        assert not Path(missing).exists()
        """,
        path,
        missing_extensions,
        kind,
    )
    assert not missing_extensions.exists()
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("read_only", [False, True])
def test_healthy_hnsw_creation_wal_refused_without_recovery(tmp_path, read_only):
    originals = tmp_path / "originals"
    originals.mkdir()
    path = originals / "chunks.db"
    make_wal_database(path, "hnsw_create", future=True)
    before = snapshot(originals)
    copy = tmp_path / "disposable.db"
    shutil.copyfile(path, copy)
    shutil.copyfile(Path(str(path) + ".wal"), Path(str(copy) + ".wal"))
    # Writable replay is ONLY a test-fixture health proof on a disposable copy.
    run_python(
        """
        import sys
        from pathlib import Path
        import duckdb
        import pytest
        from chunkhound.core.config.database_config import DatabaseConfig
        from chunkhound.core.index_format import IndexFormatError
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        path, copy, read_only = sys.argv[1:]
        with duckdb.connect(":memory:", config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "temp_directory": "",
        }) as conn:
            conn.execute("LOAD vss")
            conn.execute("SET hnsw_enable_experimental_persistence = true")
            escaped = copy.replace("'", "''")
            conn.execute(f"ATTACH '{escaped}' AS healthy")
            assert conn.execute(
                "SELECT MAX(version) FROM healthy.main.schema_version"
            ).fetchone() == (2,)
            assert conn.execute(
                "SELECT count(*) FROM healthy.main.vectors"
            ).fetchone() == (2,)
            assert conn.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE database_name='healthy'"
            ).fetchall() == [("hnsw_idx",)]
        db = DuckDBProvider(Path(path), base_directory=Path(path).parent,
            config=DatabaseConfig(read_only=read_only == "True"))
        with pytest.raises(IndexFormatError) as caught:
            db.connect()
        message = str(caught.value)
        assert "format compatibility is unknown" in message
        assert "Preserve the database" in message
        assert "recover a copy" in message
        assert "Reopening writable does not bypass" in message
        assert "corrupt" not in message.lower()
        assert "Unsupported index format" not in message
        """,
        path,
        copy,
        read_only,
    )
    assert snapshot(originals) == before
