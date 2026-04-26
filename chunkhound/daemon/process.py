"""Shared process-liveness helper for daemon components."""

from __future__ import annotations

import os
import signal
import sys
import time


def pid_alive(pid: int) -> bool:
    """Return True if the process with *pid* is still running."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import psutil

        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        # EPERM: process exists but is owned by another user — it IS alive
        return True
    except ProcessLookupError:
        return False


def stop_pid(pid: int, timeout: float = 10.0) -> bool:
    """Send SIGTERM to pid and wait up to timeout seconds for it to die."""
    if not pid_alive(pid):
        return True
    try:
        if sys.platform == "win32":
            import psutil
            try:
                psutil.Process(pid).terminate()
            except psutil.NoSuchProcess:
                return True
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return not pid_alive(pid)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.1)
    return False
