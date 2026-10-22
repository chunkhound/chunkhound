"""Public safety and lifecycle contracts for downloaded hub candidates."""

import asyncio
import errno
import gzip
import hashlib
import io
import json
import shutil
import tarfile
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import duckdb
import lance
import pyarrow as pa
import pytest

from chunkhound.hub.artifact import extract, prepare, verify_candidate
from chunkhound.hub.errors import HubDatabaseMismatch, HubUnavailable
from chunkhound.hub.manifest import MatchKey, select_manifest
from chunkhound.hub.transport import CHUNK, RESERVE
from tests.hub.artifacts import (
    build_database,
    collate_identities,
    pack_database,
    replace_with_generated_column,
    replace_with_view,
)
from tests.hub.test_index_format import snapshot
from tests.hub.test_manifest import catalogue


def selected_archive(archive, hub_http, backend="duckdb"):
    body = archive.read_bytes()
    hub_http.resources["/artifact?token=secret"] = body
    raw = catalogue()
    variant = raw["embeddings"][0]
    variant["db_provider"] = backend
    variant["versions"][0]["artifact"] = {
        "url": hub_http.url + "/artifact?token=secret",
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    return select_manifest(
        json.dumps(raw).encode(),
        MatchKey(backend, "openai", "hub-test", 8, 1),
        "latest",
    )


def tar_archive(path, entries):
    """Build real tar headers, including deliberately invalid publisher metadata."""
    with tarfile.open(path, "w:gz") as tar:
        for name, kind, payload, mode in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.mode = mode
            info.linkname = "outside"
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("stored_model", ["hub-test", "wrong-model"])
async def test_downloaded_identity_checked_before_any_install(
    tmp_path, hub_http, backend, stored_model
):
    path = build_database(tmp_path / "published", backend, model=stored_model)
    archive = tmp_path / "index.tar.gz"
    pack_database(path, archive)
    selected = selected_archive(archive, hub_http, backend)
    before = snapshot(tmp_path)
    entries = set(tmp_path.iterdir())
    if stored_model == "wrong-model":
        with pytest.raises(HubDatabaseMismatch):
            async with prepare(selected, tmp_path):
                pytest.fail("Mismatched DB was accepted")
    else:
        async with prepare(selected, tmp_path) as candidate:
            assert candidate.exists()
            assert candidate.name == path.name
            staged = snapshot(candidate.parent)
            verify_candidate(candidate, selected)
            assert snapshot(candidate.parent) == staged
        assert not candidate.exists()
    assert set(tmp_path.iterdir()) == entries
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "name,kind,mode",
    [
        ("../escape", tarfile.REGTYPE, 0),
        ("/absolute", tarfile.REGTYPE, 0),
        ("C:/drive", tarfile.REGTYPE, 0),
        ("chunks.db", tarfile.SYMTYPE, 0),
        ("chunks.db", tarfile.LNKTYPE, 0),
        ("chunks.db", tarfile.FIFOTYPE, 0),
        ("chunks.db", tarfile.CHRTYPE, 0),
        ("chunks.db", tarfile.BLKTYPE, 0),
        ("chunks.db", tarfile.GNUTYPE_SPARSE, 0),
        ("chunks.db", tarfile.REGTYPE, 0o100),
        ("chunks.db", tarfile.REGTYPE, 0o4000),
        ("lancedb.lancedb/../escape", tarfile.REGTYPE, 0),
        ("lancedb.lancedb\\escape", tarfile.REGTYPE, 0),
        ("lancedb.lancedb//file", tarfile.REGTYPE, 0),
        ("lancedb.lancedb/./file", tarfile.REGTYPE, 0),
        ("lancedb.lancedb/file ", tarfile.REGTYPE, 0),
        ("lancedb.lancedb/file.", tarfile.REGTYPE, 0),
        ("lancedb.lancedb/file:stream", tarfile.REGTYPE, 0),
        ("lancedb.lancedb/CON.txt", tarfile.REGTYPE, 0),
        ("lancedb.lancedb/LpT9", tarfile.REGTYPE, 0),
        ("lancedb.lancedb/COM1.dat", tarfile.REGTYPE, 0),
        ("lancedb.lancedb/COM¹.dat", tarfile.REGTYPE, 0),
        ("lancedb.lancedb/NUL .txt", tarfile.REGTYPE, 0),
        ("lancedb.lancedb/a?b", tarfile.REGTYPE, 0),
        ("lancedb.lancedb/a\nb", tarfile.REGTYPE, 0),
    ],
)
def test_unsafe_tar_is_soft_and_does_not_escape(tmp_path, name, kind, mode):
    archive = tar_archive(tmp_path / "bad.tar.gz", [(name, kind, b"", mode)])
    stage = tmp_path / "stage"
    stage.mkdir()
    before = snapshot(tmp_path)
    backend = "lancedb" if name.startswith("lancedb") else "duckdb"
    with pytest.raises(HubUnavailable):
        extract(archive, stage, backend)
    assert snapshot(tmp_path) == before
    assert list(stage.iterdir()) == []


@pytest.mark.parametrize(
    "names",
    [
        ["lancedb.lancedb/a", "lancedb.lancedb/a"],
        ["lancedb.lancedb/a", "lancedb.lancedb/A"],
        ["lancedb.lancedb/A/x", "lancedb.lancedb/a/y"],
        ["lancedb.lancedb/a", "lancedb.lancedb/a/x"],
        ["lancedb.lancedb/a/x", "lancedb.lancedb/a"],
    ],
)
def test_portable_collisions_and_file_directory_conflicts_are_soft(tmp_path, names):
    archive = tar_archive(
        tmp_path / "bad.tar.gz", [(name, tarfile.REGTYPE, b"x", 0) for name in names]
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(HubUnavailable):
        extract(archive, stage, "lancedb")


@pytest.mark.parametrize(
    "backend,entries",
    [
        ("duckdb", []),
        ("duckdb", [("lancedb.lancedb", tarfile.DIRTYPE, b"", 0)]),
        ("lancedb", [("chunks.db", tarfile.REGTYPE, b"x", 0)]),
        ("duckdb", [("chunks.db", tarfile.DIRTYPE, b"", 0)]),
        ("duckdb", [("chunks.db/data", tarfile.REGTYPE, b"x", 0)]),
        ("lancedb", [("lancedb.lancedb", tarfile.REGTYPE, b"x", 0)]),
        *[
            (
                backend,
                [(root + suffix, tarfile.REGTYPE, b"x", 0)],
            )
            for backend, root in [
                ("duckdb", "chunks.db"),
                ("lancedb", "lancedb.lancedb"),
            ]
            for suffix in [".root.json", ".embedding.json", ".wal"]
        ],
    ],
)
def test_backend_shape_and_sidecar_disagreements_are_fatal(tmp_path, backend, entries):
    archive = tar_archive(tmp_path / "bad.tar.gz", entries)
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(HubDatabaseMismatch):
        extract(archive, stage, backend)


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize(
    "defect", ["provider", "dims", "format", "empty", "unreadable"]
)
def test_candidate_mismatch_is_fatal_without_mutation(tmp_path, backend, defect):
    options = {"provider": "azure_openai"} if defect == "provider" else {}
    if defect == "dims":
        options = {"dims": 4}
    path = build_database(tmp_path / "published", backend, **options)
    if defect in {"format", "empty", "unreadable"}:
        if backend == "duckdb":
            with duckdb.connect(str(path)) as conn:
                if defect == "format":
                    conn.execute("INSERT INTO schema_version(version) VALUES (2)")
                elif defect == "empty":
                    conn.execute("DROP TABLE embeddings_8")
                else:
                    conn.execute("DROP TABLE chunks")
        elif defect == "format":
            lance.write_dataset(
                pa.table({"version": [2]}),
                str(path / "schema_version.lance"),
                mode="overwrite",
            )
        elif defect == "empty":
            table = lance.dataset(str(path / "chunks.lance")).to_table().slice(0, 0)
            lance.write_dataset(table, str(path / "chunks.lance"), mode="overwrite")
        else:
            shutil.rmtree(path / "chunks.lance")
    selected = select_manifest(
        json.dumps(catalogue()).encode(),
        MatchKey("duckdb", "openai", "hub-test", 8, 1),
        "latest",
    )
    selected.variant.db_provider = backend
    before = snapshot(tmp_path)
    with pytest.raises(HubDatabaseMismatch) as caught:
        verify_candidate(path, selected)
    assert "token=secret" not in str(caught.value)
    assert str(path) not in str(caught.value)
    assert caught.value.__suppress_context__
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("version", [2, 0, "1"])
def test_candidate_case_variant_format_is_fatal(tmp_path, version):
    path = build_database(tmp_path / "published")
    with duckdb.connect(str(path)) as conn:
        conn.execute("DROP TABLE schema_version")
        conn.execute('CREATE TABLE "ScHeMa_VeRsIoN" AS SELECT ? AS version', [version])
    selected = select_manifest(
        json.dumps(catalogue()).encode(),
        MatchKey("duckdb", "openai", "hub-test", 8, 1),
        "latest",
    )
    before = snapshot(tmp_path)
    with pytest.raises(HubDatabaseMismatch):
        verify_candidate(path, selected)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("table", ["schema_version", "files", "chunks", "embeddings_8"])
@pytest.mark.parametrize("external", [False, True])
def test_candidate_requires_physical_storage(tmp_path, table, external):
    path = build_database(tmp_path / "published")
    replace_with_view(path, table, external)
    selected = select_manifest(
        json.dumps(catalogue()).encode(),
        MatchKey("duckdb", "openai", "hub-test", 8, 1),
        "latest",
    )
    before = snapshot(tmp_path)
    with pytest.raises(HubDatabaseMismatch):
        verify_candidate(path, selected)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "table,column",
    [
        ("files", "path"),
        ("chunks", "code"),
        ("embeddings_8", "chunk_id"),
        ("embeddings_8", "provider"),
        ("embeddings_8", "model"),
        ("embeddings_8", "dims"),
        ("embeddings_8", "embedding"),
        ("embeddings_8", "created_at"),
        ("schema_version", "version"),
    ],
)
def test_downloaded_generated_contract_columns_are_fatal(tmp_path, table, column):
    path = build_database(tmp_path / "published")
    replace_with_generated_column(path, table, column)
    selected = select_manifest(
        json.dumps(catalogue()).encode(),
        MatchKey("duckdb", "openai", "hub-test", 8, 1),
        "latest",
    )
    before = snapshot(tmp_path)
    with pytest.raises(HubDatabaseMismatch) as caught:
        verify_candidate(path, selected)
    assert caught.value.__suppress_context__
    assert str(path) not in str(caught.value)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("collation", ["nocase", "noaccent"])
def test_candidate_dominant_identity_is_exact_despite_collation(tmp_path, collation):
    path = build_database(tmp_path / "published")
    collate_identities(path, collation)
    selected = select_manifest(
        json.dumps(catalogue()).encode(),
        MatchKey("duckdb", "openai", "hub-test", 8, 1),
        "latest",
    )
    before = snapshot(tmp_path)
    with pytest.raises(HubDatabaseMismatch):
        verify_candidate(path, selected)
    assert snapshot(tmp_path) == before


def test_tar_member_live_space_check_defeats_compression_heuristic(
    tmp_path, monkeypatch
):
    archive = tar_archive(
        tmp_path / "large.tar.gz",
        [("chunks.db", tarfile.REGTYPE, b"x" * (20 * CHUNK), 0)],
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: SimpleNamespace(free=20 * CHUNK)
    )
    with pytest.raises(HubUnavailable, match="space"):
        extract(archive, stage, "duckdb")
    assert not (stage / "chunks.db").exists()


def test_live_space_is_rechecked_before_each_bounded_write(tmp_path, monkeypatch):
    archive = tar_archive(
        tmp_path / "large.tar.gz",
        [("chunks.db", tarfile.REGTYPE, b"x" * (3 * CHUNK), 0)],
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    destination = stage / "chunks.db"

    def disk_usage(path):
        size = destination.stat().st_size if destination.exists() else 0
        return SimpleNamespace(free=RESERVE if size else RESERVE + 4 * CHUNK)

    monkeypatch.setattr(shutil, "disk_usage", disk_usage)
    with pytest.raises(HubUnavailable, match="space"):
        extract(archive, stage, "duckdb")
    assert destination.stat().st_size == CHUNK


@pytest.mark.parametrize("kind", [tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME])
def test_oversized_extended_metadata_is_rejected_before_allocation(tmp_path, kind):
    archive = tmp_path / "metadata.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        header = tarfile.TarInfo("extended")
        header.type = kind
        if kind == tarfile.XHDTYPE:
            payload = tarfile.TarInfo._create_pax_generic_header(
                {"comment": "x" * (8 * CHUNK)}, tarfile.XHDTYPE, "utf-8"
            )[512:]
        else:
            payload = b"x" * (8 * CHUNK) + b"\0"
        header.size = len(payload)
        tar.addfile(header, io.BytesIO(payload))
        tar.addfile(tarfile.TarInfo("chunks.db"), io.BytesIO())
    stage = tmp_path / "stage"
    stage.mkdir()
    tracemalloc.start()
    try:
        with pytest.raises(HubUnavailable):
            extract(archive, stage, "duckdb")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * CHUNK


@pytest.mark.parametrize("distinct_keys", [True, False])
def test_global_pax_retained_metadata_is_bounded_without_a_lifetime_quota(
    tmp_path, distinct_keys
):
    archive = tmp_path / "global-metadata.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for index in range(16):
            key = f"comment-{index}" if distinct_keys else "comment"
            payload = tarfile.TarInfo.create_pax_global_header(
                {key: "x" * (CHUNK // 2)}
            )[512:]
            header = tarfile.TarInfo("global")
            header.type = tarfile.XGLTYPE
            header.size = len(payload)
            tar.addfile(header, io.BytesIO(payload))
            tar.addfile(tarfile.TarInfo(f"lancedb.lancedb/file-{index}"))
    stage = tmp_path / "stage"
    stage.mkdir()
    failure = None
    tracemalloc.start()
    try:
        try:
            extract(archive, stage, "lancedb")
        except HubUnavailable as exc:
            failure = exc
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * CHUNK
    if distinct_keys:
        assert failure is not None
        assert "metadata" in str(failure)
    else:
        assert failure is None
        assert len(list((stage / "lancedb.lancedb").iterdir())) == 16


def test_pax_sparse_maps_are_rejected_before_buffering(tmp_path):
    archive = tmp_path / "sparse.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("chunks.db")
        member.pax_headers = {"GNU.sparse.major": "1", "GNU.sparse.minor": "0"}
        # GNU sparse 1.0 stores its map in the file payload, not PAX metadata.
        payload = b"600000\n" + b"0\n0\n" * 600_000
        payload += b"\0" * (-len(payload) % 512)
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    stage = tmp_path / "stage"
    stage.mkdir()
    tracemalloc.start()
    try:
        with pytest.raises(HubUnavailable):
            extract(archive, stage, "duckdb")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * CHUNK


def test_portable_pax_paths_and_implicit_directories_are_supported(tmp_path):
    name = "lancedb.lancedb/" + "é" * 60 + "/data"
    archive = tar_archive(
        tmp_path / "portable.tar.gz", [(name, tarfile.REGTYPE, b"payload", 0)]
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    assert extract(archive, stage, "lancedb") == stage / "lancedb.lancedb"
    assert (stage / name).read_bytes() == b"payload"


def test_large_compressible_member_is_streamed_and_metadata_not_restored(tmp_path):
    archive = tar_archive(
        tmp_path / "large.tar.gz",
        [("chunks.db", tarfile.REGTYPE, b"x" * (20 * CHUNK), 0)],
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    tracemalloc.start()
    try:
        candidate = extract(archive, stage, "duckdb")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert candidate.stat().st_size == 20 * CHUNK
    assert peak < 8 * CHUNK
    assert candidate.stat().st_mode & 0o600 == 0o600
    assert candidate.stat().st_mtime > 0


@pytest.mark.parametrize(
    "suffix",
    [
        b"invalid header".ljust(512, b"\0") + b"\0" * 1024,
        tarfile.TarInfo("chunks.db").tobuf()[:127],
        b"\0" * 127,
        tarfile.TarInfo.create_pax_global_header({"comment": "ok"})[:512]
        + b"invalid pax record".ljust(512, b"\0"),
    ],
    ids=["bad-checksum", "truncated-header", "partial-zero-header", "bad-pax"],
)
def test_corrupt_header_after_valid_member_is_soft(tmp_path, suffix):
    archive = tmp_path / "bad-header.tar.gz"
    # Gzip itself is intact: this is tar corruption after a readable member.
    archive.write_bytes(gzip.compress(tarfile.TarInfo("chunks.db").tobuf() + suffix))
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(HubUnavailable):
        extract(archive, stage, "duckdb")


@pytest.mark.parametrize(
    "format", [tarfile.USTAR_FORMAT, tarfile.GNU_FORMAT, tarfile.PAX_FORMAT]
)
@pytest.mark.parametrize("padding", [0, 512, 1024, 10240])
def test_portable_tar_formats_and_clean_header_boundary_eof_are_supported(
    tmp_path, format, padding
):
    archive = tmp_path / "portable.tar.gz"
    name = "lancedb.lancedb/" + "a" * 90 + "/" + "b" * 50
    with tarfile.open(archive, "w:gz", format=format) as tar:
        member = tarfile.TarInfo(name)
        member.size = len(b"payload")
        tar.addfile(member, io.BytesIO(b"payload"))
    # Preserve data alignment; vary optional zero blocks after the final member.
    body = gzip.decompress(archive.read_bytes()).rstrip(b"\0")
    body += b"\0" * (-len(body) % 512 + padding)
    archive.write_bytes(gzip.compress(body))
    stage = tmp_path / "stage"
    stage.mkdir()
    assert extract(archive, stage, "lancedb") == stage / "lancedb.lancedb"
    assert (stage / name).read_bytes() == b"payload"


@pytest.mark.parametrize(
    "defect", ["not-gzip", "truncated-member", "gzip-trailer", "gzip-crc"]
)
def test_unpack_corruption_is_soft(tmp_path, defect):
    archive = tar_archive(
        tmp_path / "bad.tar.gz", [("chunks.db", tarfile.REGTYPE, b"x" * CHUNK, 0)]
    )
    if defect == "not-gzip":
        archive.write_bytes(b"not a gzip tar")
    elif defect == "truncated-member":
        archive.write_bytes(gzip.compress(gzip.decompress(archive.read_bytes())[:1024]))
    elif defect == "gzip-trailer":
        archive.write_bytes(archive.read_bytes()[:-8])
    else:
        body = bytearray(archive.read_bytes())
        body[-8] ^= 1
        archive.write_bytes(body)
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(HubUnavailable):
        extract(archive, stage, "duckdb")


@pytest.mark.parametrize(
    "boundary", ["stage-symlink", "nested-symlink", "existing-file"]
)
def test_existing_paths_cannot_redirect_or_be_overwritten(tmp_path, boundary):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious").write_bytes(b"keep")
    stage = tmp_path / "stage"
    if boundary == "stage-symlink":
        stage.symlink_to(outside, target_is_directory=True)
    else:
        stage.mkdir()
        if boundary == "nested-symlink":
            (stage / "lancedb.lancedb").symlink_to(outside, target_is_directory=True)
        else:
            (stage / "lancedb.lancedb").mkdir()
            (stage / "lancedb.lancedb" / "precious").write_bytes(b"keep")
    archive = tar_archive(
        tmp_path / "bad.tar.gz",
        [("lancedb.lancedb/precious", tarfile.REGTYPE, b"overwrite", 0)],
    )
    before = snapshot(tmp_path)
    with pytest.raises(HubUnavailable):
        extract(archive, stage, "lancedb")
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "failure", ["transport", "checksum", "unpack", "tar-header", "database", "enospc"]
)
async def test_failed_prepare_removes_workspace_and_preserves_installed_files(
    tmp_path, hub_http, monkeypatch, failure
):
    installed = tmp_path / "installed"
    installed.mkdir()
    for name in [
        "chunks.db",
        "chunks.db.root.json",
        "chunks.db.embedding.json",
        "chunks.db.wal",
    ]:
        (installed / name).write_bytes(b"preserve local state")
    archive = tar_archive(
        tmp_path / "index.tar.gz",
        [("chunks.db", tarfile.REGTYPE, b"not a database", 0)],
    )
    if failure == "unpack":
        archive.write_bytes(b"not gzip")
    elif failure == "tar-header":
        body = gzip.decompress(archive.read_bytes())[:1024]
        archive.write_bytes(gzip.compress(body + b"bad header".ljust(512, b"\0")))
    selected = selected_archive(archive, hub_http)
    if failure == "transport":
        hub_http.status = 503
    elif failure == "checksum":
        selected.version.artifact.sha256 = "a" * 64
    elif failure == "enospc":
        real_open = Path.open

        class FullDisk:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def write(self, block):
                raise OSError(errno.ENOSPC, "secret disk location")

        def open_path(path, *args, **kwargs):
            if path.name == "chunks.db" and path.parent.name == "stage":
                return FullDisk()
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_path)
    before = snapshot(tmp_path)
    entries = set(tmp_path.iterdir())
    error = HubDatabaseMismatch if failure == "database" else HubUnavailable
    with pytest.raises(error) as caught:
        async with prepare(selected, tmp_path):
            pytest.fail("Invalid candidate was yielded")
    assert "secret" not in str(caught.value)
    assert snapshot(tmp_path) == before
    assert set(tmp_path.iterdir()) == entries


@pytest.mark.parametrize(
    "failure", [None, RuntimeError, OSError, asyncio.CancelledError]
)
async def test_prepare_cleans_up_on_normal_and_exceptional_context_exit(
    tmp_path, hub_http, failure
):
    path = build_database(tmp_path / "published")
    archive = tmp_path / "index.tar.gz"
    pack_database(path, archive)
    selected = selected_archive(archive, hub_http)
    before = snapshot(tmp_path)
    entries = set(tmp_path.iterdir())
    caught = None
    try:
        async with prepare(selected, tmp_path) as candidate:
            assert candidate.exists()
            if failure:
                raise failure("caller failure")
    except BaseException as exc:
        caught = exc
    assert type(caught) is failure if failure else caught is None
    assert not candidate.exists()
    assert snapshot(tmp_path) == before
    assert set(tmp_path.iterdir()) == entries


async def test_cleanup_failure_cannot_downgrade_a_fatal_database_mismatch(
    tmp_path, hub_http, monkeypatch
):
    archive = tar_archive(
        tmp_path / "index.tar.gz", [("chunks.db", tarfile.REGTYPE, b"invalid DB", 0)]
    )
    selected = selected_archive(archive, hub_http)
    real_rmtree = shutil.rmtree

    def unavailable_cleanup(path, *args, **kwargs):
        raise OSError(errno.EACCES, "secret workspace location")

    with monkeypatch.context() as boundary:
        boundary.setattr(shutil, "rmtree", unavailable_cleanup)
        with pytest.raises(HubDatabaseMismatch):
            async with prepare(selected, tmp_path):
                pytest.fail("Invalid candidate was yielded")
    for workspace in tmp_path.glob(".chunkhound-pull-*"):
        real_rmtree(workspace)


async def test_prepare_uses_independent_temporary_workspaces(tmp_path, hub_http):
    path = build_database(tmp_path / "published")
    archive = tmp_path / "index.tar.gz"
    pack_database(path, archive)
    selected = selected_archive(archive, hub_http)
    entries = set(tmp_path.iterdir())
    async with prepare(selected, tmp_path) as first:
        async with prepare(selected, tmp_path) as second:
            assert first != second
            assert first.exists() and second.exists()
        assert first.exists() and not second.exists()
    assert set(tmp_path.iterdir()) == entries
