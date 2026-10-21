"""Native installation contracts with real closed databases and filesystem paths."""

import os
import tempfile
from pathlib import Path

import pytest

import chunkhound_native as native
from chunkhound.hub.inspect import inspect_database
from tests.hub.artifacts import build_database

DUCKDB_SUFFIXES = (
    ".wal", ".wal.corrupt", ".compact_backup", ".compact_backup.wal",
    ".compact_new", ".compact_new.wal", ".old", ".old.wal",
    ".compact", ".compact.wal", ".swap_intent", ".export_tmp",
    ".root.json", ".embedding.json",
)


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_install_isolates_exact_old_inputs_and_preserves_neighbors(tmp_path, backend):
    directory = tmp_path / "index"
    target = build_database(directory, backend, model="old-model")
    candidate = build_database(directory / "private-stage", backend)
    expected = inspect_database(candidate, backend)
    suffixes = DUCKDB_SUFFIXES if backend == "duckdb" else (".embedding.json",)
    for suffix in suffixes:
        path = Path(str(target) + suffix)
        if suffix == ".export_tmp":
            path.mkdir(exist_ok=True)
            (path / "old-export").write_text("old")
        else:
            path.write_text("old")
    neighbors = [directory / "notes.txt", Path(str(target) + ".root.json.1.2.tmp")]
    for neighbor in neighbors:
        neighbor.write_text("keep")
    lease = native.acquire_install_lease(str(target), backend)
    try:
        assert lease.install(str(candidate)) == []
        assert not lease.interrupted
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_database_lease(str(target))
    finally:
        lease.close()
    assert inspect_database(target, backend) == expected
    assert not candidate.exists()
    assert all(not Path(str(target) + suffix).exists() for suffix in suffixes)
    assert all(neighbor.read_text() == "keep" for neighbor in neighbors)
    assert Path(str(target) + ".chunkhound.lock").exists()


def test_missing_candidate_does_not_touch_old_target_or_create_marker(tmp_path):
    target = tmp_path / "chunks.db"
    target.write_bytes(b"old")
    lease = native.acquire_install_lease(str(target), "duckdb")
    try:
        with pytest.raises(native.DatabaseLayoutError):
            lease.install(str(tmp_path / "missing" / "chunks.db"))
        assert target.read_bytes() == b"old"
        assert not lease.interrupted
    finally:
        lease.close()


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize(
    "kind", ["same", "wrong-name", "wrong-shape", "symlink", "symlink-absent-parent"]
)
def test_invalid_candidate_preserves_old_database(tmp_path, backend, kind):
    name = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    target = tmp_path / name
    stage = tmp_path / "stage"
    stage.mkdir()
    candidate = stage / ("wrong" if kind == "wrong-name" else name)
    if backend == "duckdb":
        target.write_bytes(b"old")
    else:
        target.mkdir()
        (target / "payload").write_bytes(b"old")
    if kind == "same":
        candidate = target
    elif kind in ("symlink", "symlink-absent-parent"):
        referent = target
        if kind == "symlink-absent-parent":
            # A separate valid candidate prevents same-target rejection from
            # masking accidental leaf-symlink following during canonicalization.
            referent = stage / "private" / name
            referent.parent.mkdir()
            if backend == "duckdb":
                referent.write_bytes(b"new")
            else:
                referent.mkdir()
                (referent / "payload").write_bytes(b"new")
        try:
            candidate.symlink_to(referent, target_is_directory=backend == "lancedb")
        except OSError:
            pytest.skip("symlink creation is unavailable")
    elif (backend == "lancedb") != (kind == "wrong-shape"):
        candidate.mkdir()
    else:
        candidate.write_bytes(b"new")
    lease = native.acquire_install_lease(str(target), backend)
    try:
        spelling = (
            stage / "missing" / ".." / candidate.name
            if kind == "symlink-absent-parent" else candidate
        )
        with pytest.raises(native.DatabaseLayoutError):
            lease.install(str(spelling))
        assert not (stage / "missing").exists()
        payload = target if backend == "duckdb" else target / "payload"
        assert payload.read_bytes() == b"old"
        assert candidate.exists()
        if kind == "symlink-absent-parent":
            assert candidate.is_symlink()
            payload = referent if backend == "duckdb" else referent / "payload"
            assert payload.read_bytes() == b"new"
        assert not lease.interrupted
    finally:
        lease.close()


def test_candidate_inside_old_database_is_rejected_before_any_changes(tmp_path):
    target = tmp_path / "lancedb.lancedb"
    candidate = target / "stage" / target.name
    candidate.mkdir(parents=True)
    (target / "payload").write_bytes(b"old")
    lease = native.acquire_install_lease(str(target), "lancedb")
    try:
        with pytest.raises(native.DatabaseLayoutError):
            lease.install(str(candidate))
        assert (target / "payload").read_bytes() == b"old"
        assert candidate.is_dir()
        assert not lease.interrupted
    finally:
        lease.close()


@pytest.mark.parametrize("marker_kind", ["directory", "symlink"])
def test_ambiguous_marker_is_refusal_without_consuming_candidate(tmp_path, marker_kind):
    target = tmp_path / "chunks.db"
    target.write_bytes(b"old")
    candidate = tmp_path / "stage" / target.name
    candidate.parent.mkdir()
    candidate.write_bytes(b"new")
    marker = Path(str(target) + ".chunkhound-install.json")
    neighbor = tmp_path / "notes.txt"
    neighbor.write_bytes(b"keep")
    if marker_kind == "directory":
        marker.mkdir()
    else:
        try:
            marker.symlink_to(neighbor)
        except OSError:
            pytest.skip("symlink creation is unavailable")
    lease = native.acquire_install_lease(str(target), "duckdb")
    try:
        with pytest.raises(native.DatabaseLayoutError):
            lease.install(str(candidate))
        assert lease.interrupted
        assert target.read_bytes() == b"old"
        assert candidate.read_bytes() == b"new"
        assert neighbor.read_bytes() == b"keep"
    finally:
        lease.close()
    with pytest.raises(native.DatabaseInterruptedError):
        native.acquire_database_lease(str(target))


def test_recovery_ignores_marker_contents_and_leaves_abandoned_quarantines(tmp_path):
    target = tmp_path / "chunks.db"
    target.write_bytes(b"partial")
    candidate = tmp_path / "stage" / target.name
    candidate.parent.mkdir()
    candidate.write_bytes(b"fresh")
    marker = Path(str(target) + ".chunkhound-install.json")
    marker.write_bytes(b'{"cleanup_path":"notes.txt",')
    abandoned = Path(str(target) + ".chunkhound-quarantine-abandoned")
    abandoned.mkdir()
    (abandoned / "payload").write_bytes(b"keep")
    neighbor = tmp_path / "notes.txt"
    neighbor.write_bytes(b"keep")
    lease = native.acquire_install_lease(str(target), "duckdb")
    try:
        assert lease.interrupted
        assert lease.install(str(candidate)) == []
        assert not lease.interrupted
    finally:
        lease.close()
    assert target.read_bytes() == b"fresh"
    assert (abandoned / "payload").read_bytes() == b"keep"
    assert neighbor.read_bytes() == b"keep"
    native.acquire_database_lease(str(target)).close()


def test_sidecar_symlinks_are_removed_without_following_their_targets(tmp_path):
    target = tmp_path / "chunks.db"
    target.write_bytes(b"old")
    candidate = tmp_path / "stage" / target.name
    candidate.parent.mkdir()
    candidate.write_bytes(b"new")
    neighbor = tmp_path / "neighbor"
    neighbor.mkdir()
    (neighbor / "payload").write_bytes(b"keep")
    sidecar = Path(str(target) + ".export_tmp")
    try:
        sidecar.symlink_to(neighbor, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    lease = native.acquire_install_lease(str(target), "duckdb")
    try:
        assert lease.install(str(candidate)) == []
    finally:
        lease.close()
    assert target.read_bytes() == b"new"
    assert not sidecar.is_symlink()
    assert (neighbor / "payload").read_bytes() == b"keep"


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_symlink_parent_candidates_and_targets_preserve_canonical_identity(
    tmp_path, backend
):
    real = tmp_path / "real"
    (real / "nested").mkdir(parents=True)
    (real / "stage").mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real / "nested", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    name = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    candidate = real / "stage" / name
    if backend == "duckdb":
        candidate.write_bytes(b"new")
    else:
        candidate.mkdir()
        (candidate / "payload").write_bytes(b"new")
    target = real / name
    lease = native.acquire_install_lease(str(alias / ".." / name), backend)
    try:
        assert lease.install(str(alias / ".." / "stage" / name)) == []
        lease.assert_covers(str(target))
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_database_lease(str(target))
    finally:
        lease.close()
    payload = target if backend == "duckdb" else target / "payload"
    assert payload.read_bytes() == b"new"
    assert not (tmp_path / name).exists()


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_candidate_after_absent_parent_traversal_uses_canonical_identity(
    tmp_path, backend
):
    name = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    target = tmp_path / name
    candidate = tmp_path / "stage" / name
    candidate.parent.mkdir()
    if backend == "duckdb":
        target.write_bytes(b"old")
        candidate.write_bytes(b"new")
    else:
        target.mkdir()
        (target / "payload").write_bytes(b"old")
        candidate.mkdir()
        (candidate / "payload").write_bytes(b"new")
    spelling = tmp_path / "missing" / ".." / "stage" / name
    lease = native.acquire_install_lease(str(target), backend)
    try:
        assert lease.install(str(spelling)) == []
        lease.assert_covers(str(target))
        assert not lease.interrupted
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_database_lease(str(target))
    finally:
        lease.close()
    payload = target if backend == "duckdb" else target / "payload"
    assert payload.read_bytes() == b"new"
    assert not candidate.exists()
    assert not (tmp_path / "missing").exists()
    native.acquire_database_lease(str(target)).close()


def test_closed_install_lease_rejects_candidate_without_changes(tmp_path):
    target = tmp_path / "chunks.db"
    candidate = tmp_path / "stage" / target.name
    candidate.parent.mkdir()
    candidate.write_bytes(b"new")
    lease = native.acquire_install_lease(str(target), "duckdb")
    lease.close()
    with pytest.raises(native.DatabaseLayoutError):
        lease.install(str(candidate))
    assert candidate.read_bytes() == b"new"
    assert not target.exists()


@pytest.mark.skipif(
    os.name != "posix", reason="requires discoverable second filesystem"
)
def test_cross_filesystem_candidate_is_rejected_before_old_database_changes(tmp_path):
    # Probe the device rather than assuming /tmp and shared memory are distinct.
    other = next(
        (
            path for path in (Path("/dev/shm"), Path("/tmp"))
            if path.is_dir() and os.access(path, os.W_OK)
            and path.stat().st_dev != tmp_path.stat().st_dev
        ),
        None,
    )
    if other is None:
        pytest.skip("no writable second filesystem available")
    with tempfile.TemporaryDirectory(dir=other, prefix="chunkhound-install-") as stage:
        target = tmp_path / "chunks.db"
        target.write_bytes(b"old")
        wal = Path(str(target) + ".wal")
        wal.write_bytes(b"old wal")
        candidate = Path(stage) / target.name
        candidate.write_bytes(b"new")
        lease = native.acquire_install_lease(str(target), "duckdb")
        try:
            with pytest.raises(native.DatabaseFilesystemError):
                lease.install(str(candidate))
            assert not lease.interrupted
            assert target.read_bytes() == b"old"
            assert wal.read_bytes() == b"old wal"
            assert candidate.read_bytes() == b"new"
        finally:
            lease.close()
