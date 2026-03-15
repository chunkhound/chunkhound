"""Shared process-liveness helper for daemon components."""

from __future__ import annotations

import os
import sys


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
