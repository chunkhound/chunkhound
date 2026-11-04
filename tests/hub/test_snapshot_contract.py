"""Protect the nonmutation assertion oracle at the filesystem/platform boundary."""

import errno
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.hub.test_index_format import snapshot


@pytest.fixture
def snapshot_files(tmp_path):
    # Disposable bytes, not a live engine's WAL: corruption is intentional below.
    for name in (
        "live.db",
        "live.db.wal",
        "closed.db",
        "closed.db.wal",
        "unrelated.wal",
        "live.db.old.wal",
        "live.db.compact_backup.wal",
        "notes.txt",
    ):
        (tmp_path / name).write_bytes(b"before")
    return tmp_path / "live.db", tmp_path / "live.db.wal"


@pytest.fixture
def denied_reads(monkeypatch):
    denied = set()
    read_bytes = Path.read_bytes

    def restricted_read(path):
        if path.resolve() in denied:
            raise PermissionError(errno.EACCES, "simulated read denial", str(path))
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", restricted_read)
    return denied


def test_windows_live_wal_metadata_requires_explicit_opt_in(
    tmp_path, monkeypatch, snapshot_files, denied_reads
):
    primary, wal = snapshot_files
    monkeypatch.setitem(snapshot.__globals__, "os", SimpleNamespace(name="nt"))
    denied_reads.update((primary.resolve(), wal.resolve()))

    # A live-primary declaration alone must not relax WAL byte evidence.
    with pytest.raises(PermissionError):
        snapshot(tmp_path, live_database_paths=(primary,))

    before = snapshot(
        tmp_path, live_database_paths=(primary,), live_wal_paths=(wal,)
    )
    assert before[str(primary)][0] == "live-windows-primary-metadata"
    assert before[str(wal)][0] == "live-windows-wal-metadata"
    wal.write_bytes(b"a larger WAL")
    assert snapshot(
        tmp_path, live_database_paths=(primary,), live_wal_paths=(wal,)
    ) != before


@pytest.mark.parametrize("platform", ["nt", "posix"])
@pytest.mark.parametrize(
    "wal_name, declare_primary",
    [
        ("live.db.wal", False),
        ("unrelated.wal", True),
        ("live.db.old.wal", True),
        ("live.db.compact_backup.wal", True),
    ],
)
def test_live_wal_opt_in_requires_declared_adjacent_primary(
    tmp_path, monkeypatch, snapshot_files, platform, wal_name, declare_primary
):
    primary, _ = snapshot_files
    monkeypatch.setitem(snapshot.__globals__, "os", SimpleNamespace(name=platform))
    with pytest.raises(ValueError, match="adjacent WAL of a declared live database"):
        snapshot(
            tmp_path,
            live_database_paths=(primary,) if declare_primary else (),
            live_wal_paths=(tmp_path / wal_name,),
        )


@pytest.mark.parametrize(
    "platform, strict_name",
    [
        ("nt", "live.db.wal"),
        ("nt", "closed.db"),
        ("nt", "closed.db.wal"),
        ("nt", "unrelated.wal"),
        ("nt", "live.db.old.wal"),
        ("nt", "live.db.compact_backup.wal"),
        ("nt", "notes.txt"),
        ("posix", "live.db"),
        ("posix", "live.db.wal"),
    ],
)
def test_strict_paths_never_substitute_metadata_for_bytes(
    tmp_path, monkeypatch, snapshot_files, denied_reads, platform, strict_name
):
    primary, wal = snapshot_files
    strict = tmp_path / strict_name
    # Even an adjacent live WAL stays strict on Windows without its own opt-in.
    live_wals = () if platform == "nt" and strict == wal else (wal,)
    monkeypatch.setitem(snapshot.__globals__, "os", SimpleNamespace(name=platform))
    before = snapshot(
        tmp_path, live_database_paths=(primary,), live_wal_paths=live_wals
    )
    stat = strict.stat()
    strict.write_bytes(b"after!")
    os.utime(strict, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert strict.stat().st_size == stat.st_size
    assert strict.stat().st_mtime_ns == stat.st_mtime_ns
    assert snapshot(
        tmp_path, live_database_paths=(primary,), live_wal_paths=live_wals
    ) != before

    # Neither the default nor another file's live opt-in may hide read failures.
    denied_reads.add(strict.resolve())
    with pytest.raises(PermissionError):
        snapshot(tmp_path)
    with pytest.raises(PermissionError):
        snapshot(tmp_path, live_database_paths=(primary,), live_wal_paths=live_wals)
