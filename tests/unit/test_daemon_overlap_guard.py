from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

import pytest

from chunkhound.daemon.discovery import (
    DaemonDiscovery,
    _normalized_project_dir,
    _roots_overlap,
    _write_json_atomically,
)

_RUNTIME_DIR_ENV = "CHUNKHOUND_DAEMON_RUNTIME_DIR"


def _set_runtime_dir_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Point daemon runtime metadata at a test-local directory."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_dir))
    return runtime_dir


def _write_registry_payload(entry_path: Path, data: dict[str, object]) -> None:
    """Write a registry entry payload for validation tests."""
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(json.dumps(data))


def _atomic_write_race_worker(
    path_str: str,
    start_event: multiprocessing.synchronize.Event,
    result_queue: multiprocessing.queues.Queue[tuple[str, str] | None],
) -> None:
    """Hammer the same JSON target from multiple processes."""
    path = Path(path_str)
    start_event.wait()
    for index in range(200):
        try:
            _write_json_atomically(path, {"worker": os.getpid(), "index": index})
        except Exception as exc:
            result_queue.put((type(exc).__name__, str(exc)))
            return
    result_queue.put(None)


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
    _set_runtime_dir_env(monkeypatch, tmp_path)

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
    _set_runtime_dir_env(monkeypatch, tmp_path)

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
    _set_runtime_dir_env(monkeypatch, tmp_path)

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


def test_registry_validation_removes_entry_with_unexpected_lock_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Entries pointing at the wrong lock file should be removed."""
    _set_runtime_dir_env(monkeypatch, tmp_path)

    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    discovery = DaemonDiscovery(project_dir)
    discovery.write_lock(os.getpid(), "tcp:127.0.0.1:54321", auth_token="token")
    _write_registry_payload(
        discovery.get_registry_entry_path(),
        {
            "project_dir": str(project_dir.resolve()),
            "pid": os.getpid(),
            "socket_path": "tcp:127.0.0.1:54321",
            "lock_path": str(project_dir / ".chunkhound" / "wrong.lock"),
            "started_at": 0.0,
        },
    )

    assert DaemonDiscovery(tmp_path / "other").find_conflicting_daemon() is None
    assert not discovery.get_registry_entry_path().exists()


def test_registry_validation_removes_entry_with_mismatched_lock_pid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Entries should be removed if the authoritative lock names a different PID."""
    _set_runtime_dir_env(monkeypatch, tmp_path)

    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    discovery = DaemonDiscovery(project_dir)
    discovery.write_lock(os.getpid() + 1, "tcp:127.0.0.1:54321", auth_token="token")
    discovery.write_registry_entry(os.getpid(), "tcp:127.0.0.1:54321")

    assert DaemonDiscovery(tmp_path / "other").find_conflicting_daemon() is None
    assert not discovery.get_registry_entry_path().exists()


def test_registry_validation_removes_entry_with_mismatched_lock_project_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Entries should be removed if the authoritative lock disagrees on the root."""
    _set_runtime_dir_env(monkeypatch, tmp_path)

    project_dir = tmp_path / "repo"
    wrong_root = tmp_path / "other-root"
    project_dir.mkdir()
    wrong_root.mkdir()

    discovery = DaemonDiscovery(project_dir)
    lock_path = project_dir / ".chunkhound" / "daemon.lock"
    discovery.write_registry_entry(os.getpid(), "tcp:127.0.0.1:54321")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "socket_path": "tcp:127.0.0.1:54321",
                "started_at": 0.0,
                "project_dir": str(wrong_root.resolve()),
                "auth_token": "token",
            }
        )
    )

    assert DaemonDiscovery(tmp_path / "other").find_conflicting_daemon() is None
    assert not discovery.get_registry_entry_path().exists()


def test_write_json_atomically_survives_same_target_concurrency(tmp_path: Path) -> None:
    """Concurrent writers should not collide on the same temp file name."""
    target_path = tmp_path / "state.json"
    start_event = multiprocessing.Event()
    result_queue: multiprocessing.Queue[tuple[str, str] | None] = (
        multiprocessing.Queue()
    )
    processes = [
        multiprocessing.Process(
            target=_atomic_write_race_worker,
            args=(str(target_path), start_event, result_queue),
        )
        for _ in range(4)
    ]

    try:
        for process in processes:
            process.start()
        start_event.set()

        results = []
        for _ in processes:
            results.append(result_queue.get(timeout=20.0))

        for process in processes:
            process.join(timeout=10.0)
            assert process.exitcode == 0

        assert results == [None, None, None, None]
        payload = json.loads(target_path.read_text())
        assert set(payload) == {"worker", "index"}
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=1.0)
