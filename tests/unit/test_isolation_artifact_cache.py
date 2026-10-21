"""Contracts for the shared native artifact cache outside pytest temp cleanup."""

import os
import sys
from pathlib import Path

import pytest
from filelock import FileLock

from tests.helpers.isolation import (
    ARTIFACTS_KEY,
    SESSION_KEY,
    _link_shared_tree,
    _prune_stale_artifact_runs,
)


def test_shared_artifacts_live_outside_session_temp(
    pytestconfig: pytest.Config,
) -> None:
    """A loaded DLL must outlive pytest's in-process temp cleanup."""
    artifacts = pytestconfig.stash[ARTIFACTS_KEY]
    session_root = pytestconfig.stash[SESSION_KEY][0]
    assert not artifacts.is_relative_to(session_root)


def test_stale_artifact_runs_are_pruned_with_read_only_files(tmp_path: Path) -> None:
    """Stale caches are deleted while the current run cache stays intact."""
    cache = tmp_path / "cache"
    stale = cache / "run-stale"
    fresh = cache / "run-fresh"
    for run in (stale, fresh):
        vss = run / "vss"
        vss.mkdir(parents=True)
        (vss / "vss.duckdb_extension").write_bytes(b"artifact")
        (vss / "vss.duckdb_extension").chmod(0o444)
        vss.chmod(0o555)

    active_lock = FileLock(fresh / ".lock")
    active_lock.acquire(timeout=0)
    try:
        _prune_stale_artifact_runs(cache)
        assert not stale.exists()
        assert fresh.exists()
    finally:
        active_lock.release()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction fallback")
def test_windows_shared_tree_uses_junction_when_symlink_is_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows without Developer Mode still gets one shared native image."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "vss.duckdb_extension").write_bytes(b"artifact")
    target = tmp_path / "home" / ".duckdb" / "extensions"

    def deny_symlink(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("symlink privilege unavailable")

    monkeypatch.setattr(Path, "symlink_to", deny_symlink)
    _link_shared_tree(source, target)

    assert os.path.samefile(target, source)
    assert (target / "vss.duckdb_extension").read_bytes() == b"artifact"
