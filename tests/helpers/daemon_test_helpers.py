"""Helper utilities for daemon integration tests.

All helpers are parallel-safe unless explicitly noted otherwise (e.g.
``count_daemon_processes`` scans the global process table and should NOT
be used when multiple tests run concurrently).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from pathlib import Path

import psutil

from chunkhound.daemon import ipc
from chunkhound.daemon.discovery import DaemonDiscovery

_RUNTIME_DIR_ENV = "CHUNKHOUND_DAEMON_RUNTIME_DIR"


@contextlib.contextmanager
def runtime_dir_env(runtime_dir: Path | None):
    """Temporarily point daemon runtime metadata at a test runtime dir."""
    previous_runtime_dir = os.environ.get(_RUNTIME_DIR_ENV)
    if runtime_dir is not None:
        os.environ[_RUNTIME_DIR_ENV] = str(runtime_dir)
    try:
        yield
    finally:
        if previous_runtime_dir is None:
            os.environ.pop(_RUNTIME_DIR_ENV, None)
        else:
            os.environ[_RUNTIME_DIR_ENV] = previous_runtime_dir


def is_daemon_running(project_dir: Path) -> bool:
    """Check if a ChunkHound daemon is running for ``project_dir``.

    Scoped to the lock file of the given project — safe for parallel test
    execution when each test uses a distinct temp directory.

    Args:
        project_dir: Project root that the daemon was started for.

    Returns:
        True if the lock file exists and the recorded PID is alive.
    """
    discovery = DaemonDiscovery(project_dir)
    return discovery.is_daemon_alive()


async def wait_for_daemon_start(
    project_dir: Path,
    timeout: float = 10.0,
    *,
    runtime_dir: Path | None = None,
) -> bool:
    """Poll until the daemon for ``project_dir`` is alive and accepting connections.

    Args:
        project_dir: Project root to check.
        timeout: Maximum seconds to wait.

    Returns:
        True if the daemon became available within ``timeout``, False otherwise.
    """
    with runtime_dir_env(runtime_dir):
        discovery = DaemonDiscovery(project_dir)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            lock = discovery.read_lock()
            if lock is not None:
                pid = lock.get("pid")
                address = str(lock.get("socket_path", discovery.get_ipc_address()))
                if isinstance(pid, int) and psutil.pid_exists(pid):
                    try:
                        if await ipc.is_connectable(address):
                            return True
                    except Exception:
                        pass
            await asyncio.sleep(0.1)
    return False


async def wait_for_daemon_shutdown(
    project_dir: Path,
    timeout: float = 5.0,
    *,
    runtime_dir: Path | None = None,
) -> bool:
    """Poll until the daemon lock file for ``project_dir`` is gone.

    This is sufficient to show that the daemon is no longer discoverable for
    the project, but it does not guarantee that every cleanup side effect
    (socket removal, registry cleanup) has completed yet.

    Args:
        project_dir: Project root to check.
        timeout: Maximum seconds to wait.

    Returns:
        True if the lock file was removed within ``timeout``, False otherwise.
    """
    with runtime_dir_env(runtime_dir):
        lock_path = DaemonDiscovery(project_dir).get_lock_path()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not lock_path.exists():
                return True
            await asyncio.sleep(0.1)
    return False


async def wait_for_daemon_full_cleanup(
    project_dir: Path,
    *,
    runtime_dir: Path | None = None,
    timeout: float = 5.0,
) -> bool:
    """Poll until lock, socket, and registry cleanup are all complete."""
    with runtime_dir_env(runtime_dir):
        discovery = DaemonDiscovery(project_dir)
        lock_path = discovery.get_lock_path()
        socket_path = discovery.get_socket_path()
        registry_entry_path = discovery.get_registry_entry_path()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lock_gone = not lock_path.exists()
        socket_gone = socket_path.startswith("tcp:") or not os.path.exists(socket_path)
        registry_gone = not registry_entry_path.exists()
        if lock_gone and socket_gone and registry_gone:
            return True
        await asyncio.sleep(0.1)
    return False


def count_daemon_processes() -> int:
    """Count how many ChunkHound daemon processes are currently running.

    Uses psutil for cross-platform process inspection.

    WARNING: This function is GLOBAL and NOT parallel-safe.  Only use it
    in tests that run serially (i.e. not with ``-n auto``).

    Returns:
        Number of matching daemon processes found.
    """
    count = 0
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline)
            if "chunkhound" in cmdline_str and "_daemon" in cmdline_str:
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return count
