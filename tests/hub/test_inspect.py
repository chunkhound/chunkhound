"""Read-only hub inventory contracts against real backend storage."""

import hashlib
import shutil
import tarfile
from pathlib import Path

import duckdb
import lance
import pyarrow as pa
import pytest

from chunkhound.core.index_format import IndexFormatError
from chunkhound.hub.inspect import inspect_database
from tests.hub.artifacts import (
    build_database,
    collate_identities,
    pack_database,
    replace_with_generated_column,
    replace_with_view,
)
from tests.hub.test_index_format import run_python, snapshot


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_inventory_reads_real_vectors_without_mutation(tmp_path, backend):
    path = build_database(tmp_path / "published", backend)
    before = snapshot(tmp_path)
    inventory = inspect_database(path, backend)
    assert inventory.format_version == 1
    assert [(g.provider, g.model, g.dims, g.count) for g in inventory.groups] == [
        ("openai", "hub-test", 8, 100)
    ]
    assert inventory.groups[0].latest > 0
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("table", ["schema_version", "files", "chunks", "embeddings_8"])
@pytest.mark.parametrize("external", [False, True])
def test_inventory_refuses_views_before_evaluating_them(tmp_path, table, external):
    path = build_database(tmp_path / "published")
    outside = replace_with_view(path, table, external)
    if outside:
        outside.unlink()
    before = snapshot(tmp_path)
    with pytest.raises(
        (ValueError, IndexFormatError), match="physical table"
    ) as caught:
        inspect_database(path, "duckdb")
    assert caught.value.__cause__ is None, "view was bound before its kind was checked"
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "table,column", [("files", "path"), ("embeddings_8", "provider")]
)
def test_inventory_refuses_generated_columns_without_evaluation(
    tmp_path, table, column
):
    path = build_database(tmp_path / "published")
    replace_with_generated_column(path, table, column)
    before = snapshot(tmp_path)
    with pytest.raises(IndexFormatError, match="stored column") as caught:
        inspect_database(path, "duckdb")
    assert caught.value.__cause__ is None
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("collation", ["nocase", "noaccent"])
def test_inventory_groups_exact_identities_independent_of_collation(
    tmp_path, collation
):
    path = build_database(tmp_path / "published")
    dominant = collate_identities(path, collation)
    before = snapshot(tmp_path)
    groups = inspect_database(path, "duckdb").groups
    assert {(g.provider, g.model, g.dims, g.count) for g in groups} == {
        ("openai", "hub-test", 8, 1),
        (*dominant, 8, 100),
    }
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_publisher_archive_contains_only_portable_database(tmp_path, backend):
    path = build_database(tmp_path / "published", backend)
    archive = tmp_path / "index.tar.gz"
    result = pack_database(path, archive)
    assert result == {
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        assert members
        assert all(Path(member.name).parts[0] == path.name for member in members)
        assert all(
            member.mode == member.uid == member.gid == member.mtime == 0
            for member in members
        )
        assert all(member.uname == member.gname == "" for member in members)


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_inventory_rejects_unusable_source_tables(tmp_path, backend):
    path = build_database(tmp_path / "published", backend)
    if backend == "duckdb":
        with duckdb.connect(str(path)) as conn:
            # Remove the referencing constraint in this deliberately broken fixture.
            conn.execute("CREATE TABLE chunks_copy AS SELECT * FROM chunks")
            conn.execute("DROP TABLE chunks")
            conn.execute("ALTER TABLE chunks_copy RENAME TO chunks")
            conn.execute("CREATE TABLE files_copy AS SELECT * EXCLUDE(path) FROM files")
            conn.execute("DROP TABLE files")
            conn.execute("ALTER TABLE files_copy RENAME TO files")
    else:
        shutil.rmtree(path / "files.lance")
        (path / "files.lance").mkdir()
    before = snapshot(tmp_path)
    with pytest.raises(Exception):
        inspect_database(path, backend)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_inventory_requires_chunk_source_columns(tmp_path, backend):
    path = build_database(tmp_path / "published", backend)
    if backend == "duckdb":
        with duckdb.connect(str(path)) as conn:
            conn.execute(
                "CREATE TABLE chunks_copy AS SELECT * EXCLUDE(code) FROM chunks"
            )
            conn.execute("DROP TABLE chunks")
            conn.execute("ALTER TABLE chunks_copy RENAME TO chunks")
    else:
        table = lance.dataset(str(path / "chunks.lance")).to_table().drop(["content"])
        lance.write_dataset(table, str(path / "chunks.lance"), mode="overwrite")
    before = snapshot(tmp_path)
    with pytest.raises(Exception):
        inspect_database(path, backend)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize(
    "defect", ["identity", "dimensions", "vector_type", "chunk_link"]
)
def test_inventory_rejects_unusable_embedding_storage(tmp_path, backend, defect):
    path = build_database(tmp_path / "published", backend)
    if backend == "duckdb":
        with duckdb.connect(str(path)) as conn:
            # CTAS avoids updating an HNSW-bearing table without loading VSS.
            projection = {
                "identity": "* REPLACE('' AS provider)",
                "dimensions": "* REPLACE(7 AS dims)",
                "chunk_link": "* EXCLUDE(chunk_id)",
                "vector_type": "* REPLACE(CAST(embedding AS VARCHAR[8]) AS embedding)",
            }[defect]
            conn.execute(
                f"CREATE TABLE vectors_copy AS SELECT {projection} FROM embeddings_8"
            )
            conn.execute("DROP TABLE embeddings_8")
            conn.execute("ALTER TABLE vectors_copy RENAME TO embeddings_8")
    else:
        table = lance.dataset(str(path / "chunks.lance")).to_table()
        if defect == "chunk_link":
            table = table.drop(["file_id"])
        else:
            name = "provider" if defect == "identity" else "embedding"
            if defect == "identity":
                values = pa.array([""] * table.num_rows)
            elif defect == "dimensions":
                values = pa.array([[]] * table.num_rows, type=pa.list_(pa.float32()))
            else:
                values = pa.array(
                    [["not a number"] * 8] * table.num_rows, type=pa.list_(pa.string())
                )
            table = table.set_column(table.schema.get_field_index(name), name, values)
        lance.write_dataset(table, str(path / "chunks.lance"), mode="overwrite")
    before = snapshot(tmp_path)
    with pytest.raises((ValueError, IndexFormatError)):
        inspect_database(path, backend)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_inventory_does_not_create_absent_database(tmp_path, backend):
    path = tmp_path / "missing" / "database"
    with pytest.raises(ValueError, match="absent"):
        inspect_database(path, backend)
    assert not path.parent.exists()


def test_inventory_rejects_unsupported_backend(tmp_path):
    with pytest.raises(ValueError, match="Unsupported database backend"):
        inspect_database(tmp_path, "sqlite")


@pytest.mark.parametrize("fixed", [False, True])
def test_lance_inventory_counts_only_vectors_by_exact_identity(tmp_path, fixed):
    path = build_database(tmp_path / "published", "lancedb")
    table = lance.dataset(str(path / "chunks.lance")).to_table()
    embeddings = [[0.25] * 8] * 99 + [None]
    if not fixed:
        embeddings[0] = [0.25] * 4
    columns = {
        "embedding": pa.array(
            embeddings,
            type=pa.list_(pa.float32(), 8) if fixed else pa.list_(pa.float32()),
        ),
        "provider": pa.array(["azure_openai"] * 50 + ["openai"] * 49 + [None]),
        "model": pa.array(["other"] * 50 + ["hub-test"] * 49 + [None]),
        "created_time": pa.array([10.0] * 49 + [20.0] + [None] * 50, type=pa.float64()),
    }
    for name, values in columns.items():
        table = table.set_column(table.schema.get_field_index(name), name, values)
    lance.write_dataset(table, str(path / "chunks.lance"), mode="overwrite")
    before = snapshot(tmp_path)
    inventory = inspect_database(path, "lancedb")
    groups = {
        (g.provider, g.model, g.dims): (g.count, g.latest) for g in inventory.groups
    }
    expected = {
        ("azure_openai", "other", 8): (50 if fixed else 49, 20.0),
        ("openai", "hub-test", 8): (49, float("-inf")),
    }
    if not fixed:
        expected[("azure_openai", "other", 4)] = (1, 10.0)
    assert groups == expected
    assert snapshot(tmp_path) == before


@pytest.fixture
def local_vss():
    # Only probe installed support; never install an extension for this fixture.
    with duckdb.connect(
        ":memory:",
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
        },
    ) as conn:
        try:
            conn.execute("LOAD vss")
        except duckdb.Error:
            pytest.skip("Local VSS extension is unavailable")


def test_checkpointed_hnsw_inventory_needs_no_extension_or_registry(
    tmp_path, local_vss
):
    path = build_database(tmp_path / "published")
    with duckdb.connect(str(path), read_only=True) as conn:
        assert any(
            "HNSW" in row[0]
            for row in conn.execute(
                "SELECT sql FROM duckdb_indexes() WHERE table_name='embeddings_8'"
            ).fetchall()
        )
    before = snapshot(tmp_path)
    run_python(
        """
        import sys
        from pathlib import Path
        import duckdb
        from chunkhound.hub.inspect import inspect_database

        path, missing = map(Path, sys.argv[1:])
        connect = duckdb.connect
        def isolated_connect(*args, **kwargs):
            config = dict(kwargs.pop("config", {}))
            config["extension_directory"] = str(missing)
            return connect(*args, config=config, **kwargs)
        duckdb.connect = isolated_connect
        inventory = inspect_database(path, "duckdb")
        assert [(g.provider, g.model, g.dims, g.count) for g in inventory.groups] == [
            ("openai", "hub-test", 8, 100)]
        assert not missing.exists()
        assert "chunkhound.registry" not in sys.modules
        """,
        path,
        tmp_path / "absent_extensions",
    )
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("kind", ["ordinary", "hnsw_insert", "hnsw_create"])
@pytest.mark.parametrize("missing_vss", [False, True])
def test_inventory_preserves_wal_and_approved_refusals(
    tmp_path, local_vss, kind, missing_vss
):
    path = build_database(tmp_path / "published' quoted")
    run_python(
        """
        import os, sys
        import duckdb
        path, kind = sys.argv[1:]
        conn = duckdb.connect(":memory:", config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
        })
        conn.execute("LOAD vss")
        conn.execute("SET hnsw_enable_experimental_persistence = true")
        escaped = path.replace("'", "''")
        conn.execute(f"ATTACH '{escaped}' AS published")
        conn.execute("USE published")
        conn.execute("SET wal_autocheckpoint = '1GB'")
        if kind != "hnsw_insert":
            indexes = conn.execute("SELECT index_name FROM duckdb_indexes() "
                                   "WHERE sql LIKE '%HNSW%'").fetchall()
            for (name,) in indexes:
                conn.execute('DROP INDEX "' + name.replace('"', '""') + '"')
        conn.execute("CHECKPOINT")
        if kind == "hnsw_create":
            conn.execute("CREATE INDEX wal_hnsw ON embeddings_8 USING HNSW(embedding)")
        conn.execute(
            "INSERT INTO embeddings_8(chunk_id, provider, model, embedding, dims) "
            "SELECT chunk_id, 'azure_openai', model, embedding, dims FROM embeddings_8")
        conn.execute("INSERT INTO schema_version(version, description) "
                     "VALUES (2, 'future in WAL')")
        os._exit(0)
        """,
        path,
        kind,
    )
    assert Path(str(path) + ".wal").stat().st_size > 0
    before = snapshot(tmp_path)
    run_python(
        """
        import sys
        from pathlib import Path
        import duckdb
        import pytest
        from chunkhound.core.index_format import IndexFormatError
        from chunkhound.hub.inspect import inspect_database

        path, kind, missing, missing_vss = sys.argv[1:]
        if missing_vss == "True":
            connect = duckdb.connect
            def isolated_connect(*args, **kwargs):
                config = dict(kwargs.pop("config", {}))
                config["extension_directory"] = missing
                return connect(*args, config=config, **kwargs)
            duckdb.connect = isolated_connect
        if missing_vss == "True" or kind == "hnsw_create":
            with pytest.raises(IndexFormatError) as caught:
                inspect_database(Path(path), "duckdb")
            message = str(caught.value)
            assert "Preserve the database" in message
            assert "format compatibility is unknown" in message
            assert "corrupt" not in message.lower()
            if missing_vss == "True":
                assert "VSS" in message and "explicit setup" in message
        else:
            inventory = inspect_database(Path(path), "duckdb")
            assert inventory.format_version == 2
            groups = {(g.provider, g.model, g.dims, g.count) for g in inventory.groups}
            assert groups == {("openai", "hub-test", 8, 100),
                              ("azure_openai", "hub-test", 8, 100)}
        assert not Path(missing).exists()
        """,
        path,
        kind,
        tmp_path / "absent_extensions",
        missing_vss,
    )
    assert snapshot(tmp_path) == before
