from __future__ import annotations

import json
import multiprocessing
import os
import socket
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import chunkhound.daemon.discovery as discovery_module
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


def test_unix_ipc_address_uses_runtime_scoped_socket_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unix IPC should live under the active runtime-scoped socket directory."""
    runtime_dir = _set_runtime_dir_env(monkeypatch, tmp_path)
    monkeypatch.setattr(discovery_module.sys, "platform", "linux")

    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    discovery = DaemonDiscovery(project_dir)

    socket_path = Path(discovery.get_ipc_address())
    assert socket_path.parent == discovery.get_socket_dir()
    assert socket_path.parent.parent == Path("/tmp") / "chunkhound-daemon-sockets"
    assert socket_path.name.startswith("chunkhound-")
    assert socket_path.suffix == ".sock"


def test_unix_ipc_address_changes_with_runtime_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The same project root should get a different Unix socket per runtime."""
    monkeypatch.setattr(discovery_module.sys, "platform", "linux")

    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    runtime_a = tmp_path / "runtime-a"
    runtime_b = tmp_path / "runtime-b"
    runtime_a.mkdir()
    runtime_b.mkdir()

    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_a))
    address_a = DaemonDiscovery(project_dir).get_ipc_address()

    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_b))
    address_b = DaemonDiscovery(project_dir).get_ipc_address()

    assert address_a != address_b


def test_unix_ipc_address_ignores_long_platform_tempdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unix IPC should stay under a short fixed root even on long tempdirs."""
    _set_runtime_dir_env(monkeypatch, tmp_path)
    monkeypatch.setattr(discovery_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        discovery_module.tempfile,
        "gettempdir",
        lambda: "/var/folders/zz/zyxvpxvq6csfxvn_n00000sm00006d/T",
    )

    project_dir = tmp_path / "repo"
    project_dir.mkdir()

    socket_path = Path(DaemonDiscovery(project_dir).get_ipc_address())

    assert socket_path.parent.parent == Path("/tmp") / "chunkhound-daemon-sockets"
    assert len(str(socket_path)) < 104


def test_windows_ipc_address_is_deterministic_within_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Windows IPC should be stable for the same root within one runtime."""
    _set_runtime_dir_env(monkeypatch, tmp_path)
    monkeypatch.setattr(discovery_module.sys, "platform", "win32")

    project_dir = tmp_path / "repo"
    project_dir.mkdir()

    first = DaemonDiscovery(project_dir).get_ipc_address()
    second = DaemonDiscovery(project_dir).get_ipc_address()

    assert first == second
    assert first.startswith("tcp:127.0.0.1:")
    assert not first.endswith(":0")


def test_windows_ipc_address_changes_with_runtime_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Different runtimes should not share the same Windows transport address."""
    monkeypatch.setattr(discovery_module.sys, "platform", "win32")

    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    runtime_a = tmp_path / "runtime-a"
    runtime_b = tmp_path / "runtime-b"
    runtime_a.mkdir()
    runtime_b.mkdir()

    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_a))
    address_a = DaemonDiscovery(project_dir).get_ipc_address()

    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_b))
    address_b = DaemonDiscovery(project_dir).get_ipc_address()

    assert address_a != address_b


def test_windows_startup_ipc_address_avoids_live_sibling_port_collision_without_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sibling roots should avoid a live lock port even if registry publish failed."""
    _set_runtime_dir_env(monkeypatch, tmp_path)
    monkeypatch.setattr(discovery_module.sys, "platform", "win32")

    root_a = tmp_path / "repo-a"
    root_b = tmp_path / "repo-b"
    root_a.mkdir()
    root_b.mkdir()

    discovery_a = DaemonDiscovery(root_a)
    discovery_b = DaemonDiscovery(root_b)
    collided_address = "tcp:127.0.0.1:55000"

    discovery_a.write_lock(os.getpid(), collided_address, auth_token="token")
    monkeypatch.setattr(
        discovery_b,
        "_preferred_windows_ipc_address",
        lambda: collided_address,
    )

    startup_address = discovery_b._select_startup_ipc_address()

    assert startup_address != collided_address
    assert startup_address.startswith("tcp:127.0.0.1:")


def test_windows_startup_ipc_address_skips_kernel_occupied_port_without_lock_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Startup selection should avoid ports occupied outside ChunkHound metadata."""
    _set_runtime_dir_env(monkeypatch, tmp_path)
    monkeypatch.setattr(discovery_module.sys, "platform", "win32")

    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    discovery = DaemonDiscovery(project_dir)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    occupied_port = listener.getsockname()[1]
    occupied_address = f"tcp:127.0.0.1:{occupied_port}"

    try:
        monkeypatch.setattr(
            discovery,
            "_preferred_windows_ipc_address",
            lambda: occupied_address,
        )

        startup_address = discovery._select_startup_ipc_address()

        assert startup_address != occupied_address
        assert startup_address.startswith("tcp:127.0.0.1:")
    finally:
        listener.close()


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
    assert Path(conflict["lock_path"]) == discovery.get_lock_path()


def test_lock_validation_reports_live_overlapping_root_without_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Live runtime locks must block overlap even before registry publication."""
    _set_runtime_dir_env(monkeypatch, tmp_path)

    parent = tmp_path / "repo"
    child = parent / "subdir"
    child.mkdir(parents=True)

    discovery = DaemonDiscovery(parent)
    discovery.write_lock(os.getpid(), "tcp:127.0.0.1:54321", auth_token="token")

    conflict = DaemonDiscovery(child).find_conflicting_daemon()
    assert conflict is not None
    assert conflict["project_dir"] == str(parent.resolve())
    assert conflict["pid"] == os.getpid()
    assert Path(conflict["lock_path"]) == discovery.get_lock_path()
    assert not discovery.get_registry_entry_path().exists()


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
    lock_path = discovery.get_lock_path()
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


def test_write_json_atomically_retries_transient_windows_replace_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Transient Windows replace failures should be retried before surfacing."""
    target_path = tmp_path / "state.json"
    original_replace = Path.replace
    attempts = {"count": 0}

    def flaky_replace(self: Path, target: Path) -> Path:
        if self.name.startswith(".state.json.") and attempts["count"] == 0:
            attempts["count"] += 1
            raise PermissionError("transient windows replace contention")
        return original_replace(self, target)

    monkeypatch.setattr(discovery_module.sys, "platform", "win32")
    monkeypatch.setattr(Path, "replace", flaky_replace)

    _write_json_atomically(target_path, {"value": 1})

    assert attempts["count"] == 1
    assert json.loads(target_path.read_text()) == {"value": 1}


def test_format_startup_failure_includes_phase_elapsed_and_error(
    tmp_path: Path,
) -> None:
    """Startup timeout formatting should expose the latest breadcrumb context."""
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    discovery = DaemonDiscovery(project_dir)
    log_path = project_dir / ".chunkhound" / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = (datetime.now() - timedelta(seconds=9)).isoformat()
    log_path.write_text(
        "\n".join(
            [
                f"[{started_at}] [startup] startup tracking began mode=daemon",
                f"[{started_at}] [startup] phase started: watchman_scope_discovery",
                (
                    f"[{started_at}] [startup] startup failed duration=9.5s "
                    "error=watchman bootstrap exploded"
                ),
            ]
        ),
        encoding="utf-8",
    )

    message = discovery._format_startup_failure(
        prefix="ChunkHound daemon did not become reachable within 30.0s",
        log_path=log_path,
    )

    assert "Last known startup phase: watchman_scope_discovery" in message
    assert "Elapsed startup duration so far: 9.500s" in message
    assert "Last startup error: watchman bootstrap exploded" in message
    assert "Recent daemon log output" in message


def test_format_startup_failure_parses_prefixed_breadcrumbs_and_keeps_legacy_support(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    discovery = DaemonDiscovery(project_dir)
    log_path = project_dir / ".chunkhound" / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = (datetime.now() - timedelta(seconds=12)).isoformat()
    log_path.write_text(
        "\n".join(
            [
                f"[{started_at}] [startup] startup: startup tracking began mode=daemon",
                f"[{started_at}] [startup] phase completed: db_connect duration=0.125s",
                f"[{started_at}] [startup] startup: phase started: watchman_watch_project",
                (
                    f"[{started_at}] [startup] startup: startup failed duration=12.0s "
                    "error=watchman session bootstrap exploded"
                ),
            ]
        ),
        encoding="utf-8",
    )

    message = discovery._format_startup_failure(
        prefix="ChunkHound daemon did not become reachable within 30.0s",
        log_path=log_path,
    )

    assert "Last known startup phase: watchman_watch_project" in message
    assert "Elapsed startup duration so far: 12.000s" in message
    assert "Last startup error: watchman session bootstrap exploded" in message
