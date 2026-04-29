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


def stop_pid(pid: int, timeout: float = 10.0) -> bool:
    """Stop pid and wait up to timeout seconds for it to die (SIGTERM on Unix, TerminateProcess on Windows)."""
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
    except ProcessLookupError:
        # Process vanished between pid_alive check and kill — already gone.
        return True
    except (PermissionError, OSError):
        return not pid_alive(pid)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.1)
    return False
