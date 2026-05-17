"""Shared process-liveness helper for daemon components."""

from __future__ import annotations

import time


def pid_alive(pid: int) -> bool:
    """Return True if the process with *pid* is still running (not a zombie)."""
    if pid <= 0:
        return False
    import psutil
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


def process_create_time(pid: int) -> float | None:
    """Return the OS process creation time, or None if it cannot be proven."""
    if pid <= 0:
        return None
    import psutil
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return None


def stop_pid(pid: int, timeout: float = 10.0) -> bool:
    """Stop pid and wait up to timeout seconds for it to die.

    Uses psutil for cross-platform terminate→kill escalation:
      graceful: Process.terminate() (SIGTERM on POSIX, TerminateProcess on Windows)
      forceful: Process.kill()     (SIGKILL on POSIX, TerminateProcess on Windows)
    """
    import psutil

    if not pid_alive(pid):
        return True
    deadline = time.monotonic() + timeout

    # Phase 1: graceful terminate
    try:
        psutil.Process(pid).terminate()
    except psutil.NoSuchProcess:
        return True
    except psutil.AccessDenied:
        return not pid_alive(pid)

    while True:
        if not pid_alive(pid):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.1, remaining))

    # Phase 2: force kill
    try:
        psutil.Process(pid).kill()
    except psutil.NoSuchProcess:
        return True
    except psutil.AccessDenied:
        return not pid_alive(pid)

    while True:
        if not pid_alive(pid):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.1, remaining))
