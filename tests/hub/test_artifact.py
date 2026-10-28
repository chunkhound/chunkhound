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

from chunkhound.core.config.hub_config import ArchiveLimits
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


@pytest.mark.parametrize(
    "archive_format", [tarfile.USTAR_FORMAT, tarfile.GNU_FORMAT, tarfile.PAX_FORMAT]
)
def test_ordinary_publisher_directory_and_file_modes_are_accepted(
    tmp_path, archive_format
):
    archive = tmp_path / "ordinary.tar.gz"
    with tarfile.open(archive, "w:gz", format=archive_format) as tar:
        for name, kind, mode, data in [
            ("lancedb.lancedb", tarfile.DIRTYPE, 0o755, b""),
            ("lancedb.lancedb/data", tarfile.REGTYPE, 0o644, b"payload"),
        ]:
            info = tarfile.TarInfo(name)
            info.type, info.mode, info.size = kind, mode, len(data)
            info.uid = info.gid = 12345
            info.mtime = 1
            tar.addfile(info, io.BytesIO(data))
    stage = tmp_path / "stage"
    stage.mkdir()
    candidate = extract(archive, stage, "lancedb")
    assert (candidate / "data").read_bytes() == b"payload"
    assert candidate.stat().st_mtime > 1
    assert (candidate / "data").stat().st_mtime > 1


@pytest.mark.parametrize("kind", [tarfile.DIRTYPE, tarfile.REGTYPE])
@pytest.mark.parametrize("special", [0o4000, 0o2000, 0o1000])
def test_special_permission_bits_are_refused_for_files_and_directories(
    tmp_path, kind, special
):
    name = "lancedb.lancedb" if kind == tarfile.DIRTYPE else "lancedb.lancedb/data"
    archive = tar_archive(
        tmp_path / "special.tar.gz", [(name, kind, b"", 0o644 | special)]
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(HubUnavailable):
        extract(archive, stage, "lancedb")
    assert list(stage.iterdir()) == []


@pytest.mark.parametrize("execute", [0o100, 0o010, 0o001])
def test_executable_regular_files_are_refused(tmp_path, execute):
    archive = tar_archive(
        tmp_path / "executable.tar.gz",
        [("chunks.db", tarfile.REGTYPE, b"", 0o644 | execute)],
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(HubUnavailable):
        extract(archive, stage, "duckdb")
    assert list(stage.iterdir()) == []


def raw_archive(tmp_path, raw, name="quota.tar.gz"):
    path = tmp_path / name
    path.write_bytes(gzip.compress(raw))
    return path


def raw_regular(name, payload=b"", *, pax=None, header_size=None):
    info = tarfile.TarInfo(name)
    info.mode = 0o644
    info.size = len(payload) if header_size is None else header_size
    info.pax_headers = pax or {}
    return (
        info.tobuf(format=tarfile.PAX_FORMAT) + payload + b"\0" * (-len(payload) % 512)
    )


def quota_stage(tmp_path, name):
    stage = tmp_path / name
    stage.mkdir()
    return stage


@pytest.mark.parametrize("extensions", [0, 1, 2])
def test_physical_extension_headers_count_toward_member_limit(tmp_path, extensions):
    extension = tarfile.TarInfo.create_pax_global_header({"comment": "ok"})
    archive = raw_archive(tmp_path, extension * extensions + raw_regular("chunks.db"))
    exact = extensions + 1
    assert extract(
        archive,
        quota_stage(tmp_path, "exact"),
        "duckdb",
        limits=ArchiveLimits(max_members=exact),
    ).exists()
    if exact > 1:
        stage = quota_stage(tmp_path, "over")
        with pytest.raises(HubUnavailable, match="member"):
            extract(
                archive, stage, "duckdb", limits=ArchiveLimits(max_members=exact - 1)
            )
        assert not (stage / "chunks.db").exists()


def test_next_regular_header_is_refused_at_physical_member_limit(tmp_path):
    # Explicit headers exceed the cap while retained paths remain within it:
    # repeat global metadata before the two ordinary members.
    extension = tarfile.TarInfo.create_pax_global_header({"comment": "ok"})
    archive = raw_archive(
        tmp_path,
        extension * 2
        + raw_regular("lancedb.lancedb/a")
        + raw_regular("lancedb.lancedb/b"),
    )
    stage = quota_stage(tmp_path, "stage")
    with pytest.raises(HubUnavailable, match="member"):
        extract(archive, stage, "lancedb", limits=ArchiveLimits(max_members=3))
    assert (stage / "lancedb.lancedb/a").exists()
    assert not (stage / "lancedb.lancedb/b").exists()
    assert extract(
        archive,
        quota_stage(tmp_path, "exact"),
        "lancedb",
        limits=ArchiveLimits(max_members=4),
    ).exists()


@pytest.mark.parametrize(
    "kind", [tarfile.XGLTYPE, tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME]
)
@pytest.mark.parametrize("count", [32, 33])
def test_fixed_extension_chain_boundary(tmp_path, kind, count):
    if kind == tarfile.GNUTYPE_LONGNAME:
        info = tarfile.TarInfo("longname")
        info.type = kind
        payload = b"chunks.db\0"
        info.size = len(payload)
        extension = info.tobuf() + payload.ljust(512, b"\0")
    else:
        extension = tarfile.TarInfo._create_pax_generic_header(
            {"comment": "ok"}, kind, "utf-8"
        )
    archive = raw_archive(tmp_path, extension * count + raw_regular("chunks.db"))
    stage = quota_stage(tmp_path, "stage")
    if count == 33:
        with pytest.raises(HubUnavailable, match="extension"):
            extract(archive, stage, "duckdb")
        assert list(stage.iterdir()) == []
    else:
        assert extract(archive, stage, "duckdb").exists()


def test_extension_chain_guard_resets_between_members(tmp_path):
    extension = tarfile.TarInfo.create_pax_global_header({"comment": "ok"})
    archive = raw_archive(
        tmp_path,
        extension * 32
        + raw_regular("lancedb.lancedb/a")
        + extension * 32
        + raw_regular("lancedb.lancedb/b"),
    )
    candidate = extract(archive, quota_stage(tmp_path, "stage"), "lancedb")
    assert {p.name for p in candidate.iterdir()} == {"a", "b"}


def test_extension_special_mode_is_refused_before_metadata_processing(tmp_path):
    info = tarfile.TarInfo("extended")
    info.type = tarfile.XHDTYPE
    info.mode = 0o4000
    info.size = 0
    archive = raw_archive(tmp_path, info.tobuf() + raw_regular("chunks.db"))
    stage = quota_stage(tmp_path, "stage")
    with pytest.raises(HubUnavailable):
        extract(archive, stage, "duckdb")
    assert list(stage.iterdir()) == []


@pytest.mark.parametrize(
    "field", ["max_members", "max_path_depth", "max_path_bytes", "max_total_path_bytes"]
)
def test_effective_paths_and_implicit_prefixes_have_exact_budgets(tmp_path, field):
    name = "lancedb.lancedb/İ/data"
    prefixes = ["lancedb.lancedb", "lancedb.lancedb/İ", name]
    exact = {
        "max_members": 3,
        "max_path_depth": 3,
        "max_path_bytes": len(name.encode("utf-8")),
        "max_total_path_bytes": sum(
            len(p.encode("utf-8")) + len(p.casefold().encode("utf-8")) for p in prefixes
        ),
    }[field]
    # PAX makes the effective path differ from its short physical-header name.
    archive = raw_archive(tmp_path, raw_regular("short", b"x", pax={"path": name}))
    stage = quota_stage(tmp_path, "over")
    with pytest.raises(HubUnavailable):
        extract(archive, stage, "lancedb", limits=ArchiveLimits(**{field: exact - 1}))
    assert list(stage.iterdir()) == []
    candidate = extract(
        archive,
        quota_stage(tmp_path, "exact"),
        "lancedb",
        limits=ArchiveLimits(**{field: exact}),
    )
    assert (candidate / "İ/data").read_bytes() == b"x"


def test_direct_extraction_uses_default_limits_and_accepts_explicit_increase(tmp_path):
    name = "lancedb.lancedb/" + "/".join(["a"] * 31 + ["data"])
    archive = raw_archive(tmp_path, raw_regular(name))
    stage = quota_stage(tmp_path, "default")
    with pytest.raises(HubUnavailable, match="depth"):
        extract(archive, stage, "lancedb")
    assert list(stage.iterdir()) == []
    candidate = extract(
        archive,
        quota_stage(tmp_path, "raised"),
        "lancedb",
        limits=ArchiveLimits(max_path_depth=33),
    )
    assert (candidate / "/".join(["a"] * 31 + ["data"])).exists()


def test_later_explicit_directories_do_not_recharge_implicit_paths(tmp_path):
    name = "lancedb.lancedb/a/data"
    prefixes = ["lancedb.lancedb", "lancedb.lancedb/a", name]
    archive = tar_archive(
        tmp_path / "explicit.tar.gz",
        [
            (name, tarfile.REGTYPE, b"x", 0o644),
            (prefixes[1], tarfile.DIRTYPE, b"", 0o755),
            (prefixes[0], tarfile.DIRTYPE, b"", 0o755),
        ],
    )
    budget = sum(len(p.encode()) + len(p.casefold().encode()) for p in prefixes)
    candidate = extract(
        archive,
        quota_stage(tmp_path, "stage"),
        "lancedb",
        limits=ArchiveLimits(max_members=3, max_total_path_bytes=budget),
    )
    assert (candidate / "a/data").read_bytes() == b"x"


def test_retained_path_budget_is_aggregate_across_members(tmp_path):
    names = ["lancedb.lancedb/a", "lancedb.lancedb/b"]
    prefixes = ["lancedb.lancedb", *names]
    budget = sum(len(p.encode()) + len(p.casefold().encode()) for p in prefixes)
    archive = raw_archive(tmp_path, b"".join(raw_regular(name) for name in names))
    stage = quota_stage(tmp_path, "over")
    with pytest.raises(HubUnavailable, match="path"):
        extract(
            archive,
            stage,
            "lancedb",
            limits=ArchiveLimits(max_total_path_bytes=budget - 1),
        )
    assert (stage / names[0]).exists()
    assert not (stage / names[1]).exists()
    assert extract(
        archive,
        quota_stage(tmp_path, "exact"),
        "lancedb",
        limits=ArchiveLimits(max_total_path_bytes=budget),
    ).exists()


def test_effective_pax_payload_is_checked_before_offending_file_creation(tmp_path):
    raw = raw_regular("lancedb.lancedb/first", b"ab")
    raw += raw_regular(
        "lancedb.lancedb/second", b"cde", pax={"size": "3"}, header_size=0
    )
    archive = raw_archive(tmp_path, raw)
    stage = quota_stage(tmp_path, "over")
    with pytest.raises(HubUnavailable, match="extracted"):
        extract(archive, stage, "lancedb", limits=ArchiveLimits(max_extracted_bytes=4))
    assert (stage / "lancedb.lancedb/first").read_bytes() == b"ab"
    assert not (stage / "lancedb.lancedb/second").exists()
    candidate = extract(
        archive,
        quota_stage(tmp_path, "exact"),
        "lancedb",
        limits=ArchiveLimits(max_extracted_bytes=5),
    )
    assert (candidate / "second").read_bytes() == b"cde"


@pytest.mark.parametrize("tail_kind", ["none", "padding", "trailing", "concatenated"])
def test_every_decompressed_byte_counts_including_tar_read_ahead(tmp_path, tail_kind):
    metadata = tarfile.TarInfo.create_pax_global_header({"comment": "metadata"})
    raw = metadata + raw_regular("chunks.db", b"x")
    tail = b""
    if tail_kind != "none":
        raw += b"\0" * 1024
        tail = b"\0" * (2 * CHUNK) if tail_kind == "padding" else b"trailing" * CHUNK
    body = gzip.compress(raw + tail)
    if tail_kind == "concatenated":
        body = gzip.compress(raw) + gzip.compress(tail)
    archive = tmp_path / "stream.tar.gz"
    archive.write_bytes(body)
    exact = len(raw) + len(tail)
    with pytest.raises(HubUnavailable, match="decompressed"):
        extract(
            archive,
            quota_stage(tmp_path, "over"),
            "duckdb",
            limits=ArchiveLimits(max_decompressed_bytes=exact - 1),
        )
    candidate = extract(
        archive,
        quota_stage(tmp_path, "exact"),
        "duckdb",
        limits=ArchiveLimits(max_decompressed_bytes=exact),
    )
    assert candidate.read_bytes() == b"x"


def test_decompression_stops_after_one_overflow_probe(tmp_path, monkeypatch):
    archive = raw_archive(tmp_path, raw_regular("chunks.db", b"x" * (4 * CHUNK)))
    emitted = []
    requested = []
    real_read = gzip.GzipFile.read

    def measured_read(stream, size=-1):
        requested.append(size)
        data = real_read(stream, size)
        emitted.append(len(data))
        return data

    # Third-party I/O boundary: prove output admitted by the extraction reader,
    # not a private helper call count. Gzip's internal buffers are not this API.
    monkeypatch.setattr(gzip.GzipFile, "read", measured_read)
    with pytest.raises(HubUnavailable, match="decompressed"):
        extract(
            archive,
            quota_stage(tmp_path, "stage"),
            "duckdb",
            limits=ArchiveLimits(max_decompressed_bytes=512),
        )
    assert sum(emitted) == 513
    assert all(0 <= n <= 513 for n in requested)


def test_crc_is_checked_after_tar_eof_at_exact_decompressed_budget(tmp_path):
    raw = raw_regular("chunks.db") + b"\0" * 1024
    raw_tail = b"accepted trailing output"
    damaged = bytearray(gzip.compress(raw_tail))
    damaged[-8] ^= 1
    archive = tmp_path / "crc.tar.gz"
    archive.write_bytes(gzip.compress(raw) + damaged)
    with pytest.raises(HubUnavailable):
        extract(
            archive,
            quota_stage(tmp_path, "stage"),
            "duckdb",
            limits=ArchiveLimits(max_decompressed_bytes=len(raw) + len(raw_tail)),
        )


async def test_prepare_budget_refusal_cleans_stage_and_preserves_installed_state(
    tmp_path, hub_http
):
    installed = tmp_path / "installed"
    installed.mkdir()
    for suffix in ["", ".wal", ".root.json", ".embedding.json"]:
        (installed / ("chunks.db" + suffix)).write_bytes(b"keep")
    archive = tar_archive(
        tmp_path / "candidate.tar.gz", [("chunks.db", tarfile.REGTYPE, b"xx", 0o644)]
    )
    selected = selected_archive(archive, hub_http)
    before = snapshot(tmp_path)
    entries = set(tmp_path.iterdir())
    with pytest.raises(HubUnavailable, match="extracted"):
        async with prepare(
            selected, tmp_path, limits=ArchiveLimits(max_extracted_bytes=1)
        ):
            pytest.fail("Over-budget candidate was yielded")
    assert snapshot(tmp_path) == before
    assert set(tmp_path.iterdir()) == entries
