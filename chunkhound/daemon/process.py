"""Shared process-liveness helper for daemon components."""

from __future__ import annotations

import os
import signal
import sys
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
    """Stop pid and wait up to timeout seconds for it to die."""
    if not pid_alive(pid):
        return True
    deadline = time.monotonic() + timeout
    try:
        if sys.platform == "win32":
            import psutil
            try:
                psutil.Process(pid).terminate()
            except psutil.NoSuchProcess:
                return True
            except psutil.AccessDenied:
                return not pid_alive(pid)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        # Process vanished between pid_alive check and kill — already gone.
        return True
    except (PermissionError, OSError):
        return not pid_alive(pid)

    while True:
        if not pid_alive(pid):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.1, remaining))

    try:
        if sys.platform == "win32":
            import psutil
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                return True
            except psutil.AccessDenied:
                return not pid_alive(pid)
        else:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return not pid_alive(pid)

    while True:
        if not pid_alive(pid):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.1, remaining))
