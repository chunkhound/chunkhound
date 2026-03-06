from __future__ import annotations

import os
from pathlib import Path

import pytest

from chunkhound.daemon.discovery import (
    DaemonDiscovery,
    _normalized_project_dir,
    _roots_overlap,
)


def test_roots_overlap_classifies_same_parent_child_and_siblings(
    tmp_path: Path,
) -> None:
    """Overlap checks should be path-segment aware."""
    parent = tmp_path / "repo"
    child = parent / "subdir"
    sibling = tmp_path / "repo-b"

    child.mkdir(parents=True)
    sibling.mkdir()

    assert _roots_overlap(parent, parent)
    assert _roots_overlap(parent, child)
    assert _roots_overlap(child, parent)
    assert not _roots_overlap(parent, sibling)


def test_normalized_project_dir_resolves_symlink_to_same_root(tmp_path: Path) -> None:
    """Symlinked paths should normalize to the same canonical root."""
    real_root = tmp_path / "real"
    real_root.mkdir()
    link_root = tmp_path / "link"
    try:
        link_root.symlink_to(real_root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic links not supported on this platform")

    assert _normalized_project_dir(real_root) == _normalized_project_dir(link_root)
    assert _roots_overlap(real_root, link_root)


def test_registry_validation_removes_dead_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dead registry entries should be removed instead of blocking startup."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    discovery = DaemonDiscovery(project_dir)
    discovery.write_lock(999_999_999, "tcp:127.0.0.1:54321", auth_token="token")
    discovery.write_registry_entry(999_999_999, "tcp:127.0.0.1:54321")

    other = DaemonDiscovery(tmp_path / "other")
    assert other.find_conflicting_daemon() is None
    assert not discovery.get_registry_entry_path().exists()


def test_registry_validation_removes_entry_without_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Live-PID entries without a material lock should be cleaned up."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    discovery = DaemonDiscovery(project_dir)
    discovery.write_registry_entry(os.getpid(), "tcp:127.0.0.1:54321")

    other = DaemonDiscovery(tmp_path / "other")
    assert other.find_conflicting_daemon() is None
    assert not discovery.get_registry_entry_path().exists()


def test_registry_validation_reports_live_overlapping_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Validated registry entries should block overlapping parent/child roots."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    parent = tmp_path / "repo"
    child = parent / "subdir"
    child.mkdir(parents=True)

    discovery = DaemonDiscovery(parent)
    discovery.write_lock(os.getpid(), "tcp:127.0.0.1:54321", auth_token="token")
    discovery.write_registry_entry(os.getpid(), "tcp:127.0.0.1:54321")

    conflict = DaemonDiscovery(child).find_conflicting_daemon()
    assert conflict is not None
    assert conflict["project_dir"] == str(parent.resolve())
    assert conflict["pid"] == os.getpid()
    assert Path(conflict["lock_path"]) == parent / ".chunkhound" / "daemon.lock"
