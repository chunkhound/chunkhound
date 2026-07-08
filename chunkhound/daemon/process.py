"""Shared process-liveness helper for daemon components."""

from __future__ import annotations

import time

# Seconds budget for the SIGKILL polling phase, separate from the
# user-requested graceful timeout, preventing starvation when Phase 1
# consumes the entire user timeout before the OS updates process state.
_FORCE_KILL_DEADLINE: float = 2.0


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
    """Stop pid and wait for graceful termination plus force-kill fallback.

    ``timeout`` is the graceful-stop budget before escalation. If SIGTERM does
    not stop the process in time, a fixed additional SIGKILL polling window is
    used so OS process-state updates are not starved by the graceful phase.

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

    # Phase 2: force kill — separate deadline so the OS process-state
    # update (running → zombie) is not starved by Phase 1 having
    # consumed the entire user-requested timeout.
    try:
        psutil.Process(pid).kill()
    except psutil.NoSuchProcess:
        return True
    except psutil.AccessDenied:
        return not pid_alive(pid)

    force_deadline = time.monotonic() + _FORCE_KILL_DEADLINE
    while True:
        if not pid_alive(pid):
            return True
        remaining = force_deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.1, remaining))
