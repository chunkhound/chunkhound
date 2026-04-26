from __future__ import annotations

import json
import multiprocessing
import os
import socket
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from unittest.mock import AsyncMock

import chunkhound.daemon.discovery as discovery_module
from chunkhound.daemon.discovery import (
    DaemonDiscovery,
    DaemonStartupHandle,
    _normalized_project_dir,
    _roots_overlap,
    _write_json_atomically,
)

_RUNTIME_DIR_ENV = "CHUNKHOUND_DAEMON_RUNTIME_DIR"
_REGISTRY_DIR_ENV = "CHUNKHOUND_DAEMON_REGISTRY_DIR"


def _set_runtime_dir_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Point daemon runtime metadata at a test-local directory.

    Also isolates the user-scoped registry directory so tests cannot reach
    a real developer's registry state on the host.
    """
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_dir))
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(_REGISTRY_DIR_ENV, str(registry_dir))
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


def test_registry_overlap_same_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Within one runtime dir, same-root protection runs through the lock path.

    Two daemons under the same ``CHUNKHOUND_DAEMON_RUNTIME_DIR`` collide on
    the runtime-scoped lock file before overlap detection is consulted, so
    the registry pass intentionally skips entries whose authoritative
    ``lock_path`` is this runtime's own ``get_lock_path()``. Cross-runtime
    same-root protection is exercised by
    ``test_registry_overlap_blocks_second_daemon_across_runtime_dirs``.
    """
    _set_runtime_dir_env(monkeypatch, tmp_path)

    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    seeded = DaemonDiscovery(project_dir)
    seeded.write_lock(os.getpid(), "tcp:127.0.0.1:54330", auth_token="token")
    seeded.write_registry_entry(os.getpid(), "tcp:127.0.0.1:54330")

    conflict = DaemonDiscovery(project_dir).find_conflicting_daemon()
    assert conflict is None
    assert seeded.get_registry_entry_path().exists()


def test_registry_overlap_nested_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live registry entry for a parent root must block a nested-child daemon."""
    _set_runtime_dir_env(monkeypatch, tmp_path)

    parent = tmp_path / "proj"
    child = parent / "sub"
    child.mkdir(parents=True)

    seeded = DaemonDiscovery(parent)
    seeded.write_lock(os.getpid(), "tcp:127.0.0.1:54331", auth_token="token")
    seeded.write_registry_entry(os.getpid(), "tcp:127.0.0.1:54331")

    conflict = DaemonDiscovery(child).find_conflicting_daemon()
    assert conflict is not None
    assert conflict["project_dir"] == str(parent.resolve())
    assert conflict["pid"] == os.getpid()


def test_registry_overlap_nested_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live registry entry for a child root must block a parent-root daemon."""
    _set_runtime_dir_env(monkeypatch, tmp_path)

    parent = tmp_path / "proj"
    child = parent / "sub"
    child.mkdir(parents=True)

    seeded = DaemonDiscovery(child)
    seeded.write_lock(os.getpid(), "tcp:127.0.0.1:54332", auth_token="token")
    seeded.write_registry_entry(os.getpid(), "tcp:127.0.0.1:54332")

    conflict = DaemonDiscovery(parent).find_conflicting_daemon()
    assert conflict is not None
    assert conflict["project_dir"] == str(child.resolve())
    assert conflict["pid"] == os.getpid()


def test_registry_overlap_blocks_second_daemon_across_runtime_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two daemons under different runtime dirs must still collide.

    Locks a same-root daemon launched under ``runtime_a`` and then constructs
    a fresh ``DaemonDiscovery`` under ``runtime_b`` (different runtime dir,
    same user-scoped registry dir). The second daemon's startup must be
    refused via the cross-runtime registry pass because the first daemon
    already owns an overlapping project root.
    """
    # Shared user-scoped registry dir so both runtimes look at the same file.
    registry_dir = tmp_path / "shared-registry"
    registry_dir.mkdir()
    monkeypatch.setenv(_REGISTRY_DIR_ENV, str(registry_dir))

    runtime_a = tmp_path / "runtime-a"
    runtime_b = tmp_path / "runtime-b"
    runtime_a.mkdir()
    runtime_b.mkdir()

    parent = tmp_path / "proj"
    child = parent / "sub"
    child.mkdir(parents=True)

    # Seed a live daemon under runtime A for the parent root.
    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_a))
    seeded = DaemonDiscovery(parent)
    seeded.write_lock(os.getpid(), "tcp:127.0.0.1:54340", auth_token="token")
    seeded.write_registry_entry(os.getpid(), "tcp:127.0.0.1:54340")
    runtime_a_lock_path = seeded.get_lock_path()
    assert runtime_a_lock_path.is_file()

    # Construct a fresh discovery under runtime B (different runtime dir, same
    # user-scoped registry dir). Both the same-root and the parent/child cases
    # must be refused: the registry pass owns cross-runtime protection.
    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_b))
    fresh_same_root = DaemonDiscovery(parent)
    child_conflict = DaemonDiscovery(child).find_conflicting_daemon()
    assert child_conflict is not None
    assert child_conflict["project_dir"] == str(parent.resolve())
    assert child_conflict["pid"] == os.getpid()
    # The lock path in the returned entry points at runtime A's lock file,
    # not runtime B's expected lock path. That is the invariant that proves
    # the cross-runtime registry pass trusts stored lock paths rather than
    # deriving them from the current runtime dir.
    assert Path(child_conflict["lock_path"]) == runtime_a_lock_path
    assert not Path(child_conflict["lock_path"]).is_relative_to(runtime_b)

    # Same-root across runtime dirs: runtime B must see runtime A's parent-root
    # daemon and refuse to start. This is the cross-runtime contract the lock
    # pass cannot enforce — lock files are runtime-scoped and would not collide.
    parent_conflict = fresh_same_root.find_conflicting_daemon()
    assert parent_conflict is not None
    assert parent_conflict["project_dir"] == str(parent.resolve())
    assert parent_conflict["pid"] == os.getpid()
    assert Path(parent_conflict["lock_path"]) == runtime_a_lock_path
    assert not Path(parent_conflict["lock_path"]).is_relative_to(runtime_b)
    assert seeded.get_registry_entry_path().exists()


def test_cross_runtime_startup_lock_path_is_not_runtime_scoped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cross-runtime startup lock must live outside ``get_runtime_dir``.

    If this lock ever leaks back under the runtime dir, two proxies under
    different ``CHUNKHOUND_DAEMON_RUNTIME_DIR`` values can both acquire their
    own copy and the pre-registry cross-runtime startup race reopens. Pin the
    invariant so no future refactor can silently put the file back.
    """
    _set_runtime_dir_env(monkeypatch, tmp_path)

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    discovery = DaemonDiscovery(project_dir)

    lock_path = discovery.get_cross_runtime_startup_lock_path()
    runtime_dir = discovery.get_runtime_dir().resolve()
    assert not lock_path.resolve().is_relative_to(runtime_dir)


def test_cross_runtime_startup_lock_path_is_user_wide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cross-runtime startup lock must be a single user-wide file.

    Hash-keying the lock per canonical project root would only serialize
    exact same-root startups and would still let parent/child overlapping
    roots (e.g. ``/proj`` and ``/proj/sub``) race past each other before
    either publishes a registry entry. Pin the invariant that
    ``get_cross_runtime_startup_lock_path`` returns the same path for every
    project root so the barrier is user-wide.
    """
    _set_runtime_dir_env(monkeypatch, tmp_path)

    proj_a = tmp_path / "proj-a"
    proj_b = tmp_path / "proj-b"
    nested = proj_a / "sub"
    proj_a.mkdir()
    proj_b.mkdir()
    nested.mkdir()

    path_a = DaemonDiscovery(proj_a).get_cross_runtime_startup_lock_path()
    path_b = DaemonDiscovery(proj_b).get_cross_runtime_startup_lock_path()
    path_nested = DaemonDiscovery(nested).get_cross_runtime_startup_lock_path()
    assert path_a == path_b == path_nested


def test_cross_runtime_startup_lock_serializes_same_root_across_runtime_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two same-root proxies under different runtime dirs must not both acquire.

    Concrete pre-registry race the user-scoped barrier closes: proxy A under
    ``runtime_a`` and proxy B under ``runtime_b`` both target ``/proj``; both
    pass ``_reuse_live_daemon`` (no daemon yet), both would call
    ``find_conflicting_daemon`` and find nothing, and both would spawn a
    daemon. The new user-scoped per-root startup lock must block proxy B
    from acquiring while proxy A holds.
    """
    # Shared user-scoped registry dir so both runtimes resolve the same
    # cross-runtime startup lock path.
    registry_dir = tmp_path / "shared-registry"
    registry_dir.mkdir()
    monkeypatch.setenv(_REGISTRY_DIR_ENV, str(registry_dir))

    runtime_a = tmp_path / "runtime-a"
    runtime_b = tmp_path / "runtime-b"
    runtime_a.mkdir()
    runtime_b.mkdir()

    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    # Proxy A acquires under runtime A.
    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_a))
    discovery_a = DaemonDiscovery(project_dir)
    assert discovery_a._acquire_cross_runtime_startup_lock() is True
    lock_path_a = discovery_a.get_cross_runtime_startup_lock_path()
    assert lock_path_a.is_file()

    # Proxy B attempts to acquire under runtime B — must be refused even
    # though runtime dirs differ, because the lock is user-scoped per root.
    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_b))
    discovery_b = DaemonDiscovery(project_dir)
    lock_path_b = discovery_b.get_cross_runtime_startup_lock_path()
    assert lock_path_b == lock_path_a
    assert discovery_b._acquire_cross_runtime_startup_lock() is False

    # Proxy A releases; proxy B can now acquire.
    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_a))
    discovery_a._release_cross_runtime_startup_lock()
    assert not lock_path_a.exists()

    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_b))
    assert discovery_b._acquire_cross_runtime_startup_lock() is True
    discovery_b._release_cross_runtime_startup_lock()


def test_cross_runtime_startup_lock_serializes_parent_child_across_runtime_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Parent/child overlapping roots must contend on the cross-runtime lock.

    The original pre-registry race is not limited to exact same-root: a
    proxy on ``/proj`` under runtime A and a proxy on ``/proj/sub`` under
    runtime B can both pass ``find_conflicting_daemon`` and spawn duplicate
    overlapping daemons. A hash-keyed per-root lock would let them through
    because the two canonical roots produce different project hashes. The
    user-wide barrier must block them regardless.
    """
    registry_dir = tmp_path / "shared-registry"
    registry_dir.mkdir()
    monkeypatch.setenv(_REGISTRY_DIR_ENV, str(registry_dir))

    runtime_a = tmp_path / "runtime-a"
    runtime_b = tmp_path / "runtime-b"
    runtime_a.mkdir()
    runtime_b.mkdir()

    parent = tmp_path / "proj"
    child = parent / "sub"
    child.mkdir(parents=True)

    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_a))
    disc_parent = DaemonDiscovery(parent)
    assert disc_parent._acquire_cross_runtime_startup_lock() is True
    try:
        monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_b))
        disc_child = DaemonDiscovery(child)
        # Parent-vs-child must contend even though canonical project hashes
        # differ, because the underlying lock file is user-wide.
        assert (
            disc_parent.get_cross_runtime_startup_lock_path()
            == disc_child.get_cross_runtime_startup_lock_path()
        )
        assert disc_child._acquire_cross_runtime_startup_lock() is False
    finally:
        monkeypatch.setenv(_RUNTIME_DIR_ENV, str(runtime_a))
        disc_parent._release_cross_runtime_startup_lock()


def test_cross_runtime_startup_lock_serializes_unrelated_roots_user_wide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unrelated roots must also contend; the barrier is deliberately user-wide.

    Serializing the startup critical section across every root is the only
    sound way to close the parent/child race without building a full overlap
    lattice inside the lock layer. Pin that unrelated roots contend too so a
    future refactor cannot silently hash-key this lock and reopen the gap.
    """
    registry_dir = tmp_path / "shared-registry"
    registry_dir.mkdir()
    monkeypatch.setenv(_REGISTRY_DIR_ENV, str(registry_dir))
    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(tmp_path / "runtime"))

    proj_a = tmp_path / "proj-a"
    proj_b = tmp_path / "proj-b"
    proj_a.mkdir()
    proj_b.mkdir()

    disc_a = DaemonDiscovery(proj_a)
    disc_b = DaemonDiscovery(proj_b)
    assert disc_a._acquire_cross_runtime_startup_lock() is True
    try:
        assert disc_b._acquire_cross_runtime_startup_lock() is False
    finally:
        disc_a._release_cross_runtime_startup_lock()
    # After the holder releases, the second proxy can finally acquire.
    assert disc_b._acquire_cross_runtime_startup_lock() is True
    disc_b._release_cross_runtime_startup_lock()


async def test_ensure_daemon_running_publishes_registry_entry_authoritatively(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The proxy must write the registry entry before releasing the cross-runtime lock.

    Cross-runtime discovery reads only the user-scoped registry dir, so if
    the entry is missing when the user-wide startup lock is released, a
    later proxy under a different ``CHUNKHOUND_DAEMON_RUNTIME_DIR`` cannot
    see this daemon — IPC addresses and authoritative lock files are
    runtime-scoped and invisible to it. Server-side ``write_registry_entry``
    in ``daemon/server.py`` is wrapped in a non-fatal ``try/except``, so the
    only way to guarantee publication for this startup path is for the
    proxy itself to write the entry while the lock is still held.

    This test simulates a daemon that writes its lock file but never calls
    ``write_registry_entry`` (mirroring the non-fatal publish-failure mode)
    and asserts that after ``find_or_start_daemon`` returns, the registry
    entry is nonetheless present and visible to a proxy under a different
    runtime dir.
    """
    _set_runtime_dir_env(monkeypatch, tmp_path)

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    discovery = DaemonDiscovery(project_dir)

    fake_address = "tcp:127.0.0.1:54997"

    class _FakeProcess:
        def poll(self) -> int | None:
            return None

    log_path = discovery.get_daemon_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch()

    def fake_start(args: object, *, socket_path: str | None = None) -> DaemonStartupHandle:
        # Publish an authoritative runtime-scoped lock file the way a real
        # daemon would — but deliberately never publish a registry entry, so
        # the proxy-side write is the only thing that can close the gap.
        discovery.write_lock(os.getpid(), fake_address, auth_token="token")
        return DaemonStartupHandle(process=_FakeProcess(), log_path=log_path)  # type: ignore[arg-type]

    async def fake_connectable(address: str) -> bool:
        return address == fake_address

    monkeypatch.setattr(discovery, "_start_daemon_subprocess", fake_start)
    monkeypatch.setattr(discovery, "_socket_connectable", fake_connectable)

    entry_path = discovery.get_registry_entry_path()
    assert not entry_path.exists()

    address = await discovery.find_or_start_daemon(object())
    assert address == fake_address

    # The proxy-side authoritative write ran before the cross-runtime lock
    # was released, so the registry entry is present even though the
    # simulated daemon never published its own.
    assert entry_path.exists()
    payload = json.loads(entry_path.read_text())
    assert payload["pid"] == os.getpid()
    assert payload["socket_path"] == fake_address
    assert payload["project_dir"] == str(project_dir.resolve())

    # And a proxy under a different runtime dir can see the daemon via
    # the user-scoped registry pass — the contract the whole cross-runtime
    # barrier exists to enforce.
    alt_runtime = tmp_path / "runtime-alt"
    alt_runtime.mkdir()
    monkeypatch.setenv(_RUNTIME_DIR_ENV, str(alt_runtime))
    alt_discovery = DaemonDiscovery(project_dir)
    conflict = alt_discovery.find_conflicting_daemon()
    assert conflict is not None
    assert conflict["pid"] == os.getpid()
    assert conflict["socket_path"] == fake_address


@pytest.mark.asyncio
async def test_find_or_start_daemon_terminates_child_when_registry_publish_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_runtime_dir_env(monkeypatch, tmp_path)

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    discovery = DaemonDiscovery(project_dir)

    fake_address = "tcp:127.0.0.1:54998"

    class _FakeProcess:
        def __init__(self) -> None:
            self.terminate_calls = 0
            self.wait_calls = 0
            self.returncode = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminate_calls += 1
            self.returncode = -15

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            return -15

        def kill(self) -> None:
            self.returncode = -9

    fake_process = _FakeProcess()
    log_path = discovery.get_daemon_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch()

    def fake_start(args: object, *, socket_path: str | None = None) -> DaemonStartupHandle:
        del args, socket_path
        discovery.write_lock(os.getpid(), fake_address, auth_token="token")
        return DaemonStartupHandle(process=fake_process, log_path=log_path)  # type: ignore[arg-type]

    async def fake_connectable(address: str) -> bool:
        return address == fake_address

    def fake_write_registry_entry(pid: int, socket_path: str) -> None:
        del pid, socket_path
        raise OSError("registry write failed")

    monkeypatch.setattr(discovery, "_start_daemon_subprocess", fake_start)
    monkeypatch.setattr(discovery, "_socket_connectable", fake_connectable)
    monkeypatch.setattr(discovery, "write_registry_entry", fake_write_registry_entry)

    with pytest.raises(OSError, match="registry write failed"):
        await discovery.find_or_start_daemon(object())

    assert fake_process.terminate_calls == 1
    assert fake_process.wait_calls >= 1
    assert not discovery.get_registry_entry_path().exists()


@pytest.mark.asyncio
async def test_find_or_start_daemon_terminates_detached_child_on_startup_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_runtime_dir_env(monkeypatch, tmp_path)

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    discovery = DaemonDiscovery(project_dir)

    class _FakeProcess:
        def __init__(self) -> None:
            self.terminate_calls = 0
            self.kill_calls = 0
            self.wait_calls = 0
            self.returncode = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminate_calls += 1

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("chunkhound", timeout)
            self.returncode = -9
            return self.returncode

        def kill(self) -> None:
            self.kill_calls += 1

    fake_process = _FakeProcess()
    log_path = discovery.get_daemon_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(discovery_module, "_STARTUP_TIMEOUT", 0.05)
    monkeypatch.setattr(discovery, "_reuse_live_daemon", AsyncMock(return_value=None))
    monkeypatch.setattr(discovery, "_acquire_cross_runtime_startup_lock", lambda: True)
    monkeypatch.setattr(discovery, "_release_cross_runtime_startup_lock", lambda: None)
    monkeypatch.setattr(discovery, "_acquire_global_startup_lock", lambda: True)
    monkeypatch.setattr(discovery, "_release_global_startup_lock", lambda: None)
    monkeypatch.setattr(discovery, "_acquire_starter_lock", lambda: True)
    monkeypatch.setattr(discovery, "_release_starter_lock", lambda: None)
    monkeypatch.setattr(discovery, "find_conflicting_daemon", lambda: None)
    monkeypatch.setattr(discovery, "read_lock", lambda: None)
    monkeypatch.setattr(
        discovery,
        "_socket_connectable",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        discovery,
        "_start_daemon_subprocess",
        lambda args, socket_path=None: DaemonStartupHandle(
            process=fake_process,  # type: ignore[arg-type]
            log_path=log_path,
        ),
    )

    with pytest.raises(RuntimeError, match="did not start within"):
        await discovery.find_or_start_daemon(object())

    assert fake_process.terminate_calls == 1
    assert fake_process.kill_calls == 1


@pytest.mark.asyncio
async def test_terminate_startup_handle_swallows_process_errors() -> None:
    class _ExplodingProcess:
        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            raise OSError("terminate exploded")

    startup = DaemonStartupHandle(
        process=_ExplodingProcess(),  # type: ignore[arg-type]
        log_path=Path("/tmp/daemon.log"),
    )

    await discovery_module._terminate_startup_handle(startup)


def test_validated_registry_entry_normalizes_corrupt_started_at(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-numeric ``started_at`` must default to 0.0 without raising."""
    _set_runtime_dir_env(monkeypatch, tmp_path)

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    discovery = DaemonDiscovery(project_dir)
    discovery.write_lock(os.getpid(), "tcp:127.0.0.1:54333", auth_token="token")
    # Overwrite the lock file with a corrupt started_at so both lookup paths
    # exercise the defensive normalization.
    lock_path = discovery.get_lock_path()
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "socket_path": "tcp:127.0.0.1:54333",
                "started_at": "not-a-float",
                "project_dir": str(project_dir.resolve()),
                "auth_token": "token",
            }
        )
    )
    discovery.write_registry_entry(os.getpid(), "tcp:127.0.0.1:54333")
    entry_path = discovery.get_registry_entry_path()
    payload = json.loads(entry_path.read_text())
    payload["started_at"] = "also-not-a-float"
    entry_path.write_text(json.dumps(payload))

    entry = discovery._validated_registry_entry(entry_path)
    assert entry is not None
    assert entry["started_at"] == 0.0
    assert entry["pid"] == os.getpid()
    # Defensive normalization must NOT delete the entry.
    assert entry_path.exists()


def test_find_conflicting_daemon_skips_corrupt_started_at_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A corrupt registry entry must not break iteration over real overlaps."""
    _set_runtime_dir_env(monkeypatch, tmp_path)

    parent = tmp_path / "proj"
    other_root = tmp_path / "unrelated"
    parent.mkdir()
    other_root.mkdir()

    # Seed the *real* overlapping entry first so file iteration order does
    # not let the test pass by accident.
    real = DaemonDiscovery(parent)
    real.write_lock(os.getpid(), "tcp:127.0.0.1:54334", auth_token="token")
    real.write_registry_entry(os.getpid(), "tcp:127.0.0.1:54334")

    # Seed a corrupt registry entry for an unrelated root with non-numeric
    # started_at metadata in BOTH the lock file and the registry entry.
    # _validated_registry_entry walks through the float() conversion and
    # must not raise on either field.
    other = DaemonDiscovery(other_root)
    other.write_lock(os.getpid(), "tcp:127.0.0.1:54335", auth_token="token")
    other_lock_path = other.get_lock_path()
    other_lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "socket_path": "tcp:127.0.0.1:54335",
                "started_at": "garbage",
                "project_dir": str(other_root.resolve()),
                "auth_token": "token",
            }
        )
    )
    other.write_registry_entry(os.getpid(), "tcp:127.0.0.1:54335")
    other_entry_path = other.get_registry_entry_path()
    other_payload = json.loads(other_entry_path.read_text())
    other_payload["started_at"] = "also-garbage"
    other_entry_path.write_text(json.dumps(other_payload))

    child = parent / "sub"
    child.mkdir()
    conflict = DaemonDiscovery(child).find_conflicting_daemon()
    assert conflict is not None
    assert conflict["project_dir"] == str(parent.resolve())


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
