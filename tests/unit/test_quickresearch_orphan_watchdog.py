"""Watchdog tests for `_quickresearch` — die when the parent process is gone.

See `test_orphan_watchdog_triggers_shutdown_when_orphaned` in
tests/unit/test_client_proxy.py for the daemon precedent.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time

import pytest

import chunkhound.api.cli.commands.quickresearch as qr_mod


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no ppid reparenting")
def test_orphan_watchdog_calls_exit_when_ppid_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The watchdog invokes _orphan_exit once getppid() diverges from initial."""
    monkeypatch.setattr(qr_mod, "_ORPHAN_POLL_INTERVAL", 0.0)
    monkeypatch.setattr(qr_mod.os, "getppid", lambda: 9999)

    called = threading.Event()
    monkeypatch.setattr(qr_mod, "_orphan_exit", called.set)

    stop = threading.Event()
    qr_mod._orphan_watchdog_thread(initial_ppid=1234, stop=stop)

    assert called.is_set()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no ppid reparenting")
def test_orphan_watchdog_does_not_exit_when_ppid_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stable ppid keeps the watchdog looping without firing _orphan_exit."""
    monkeypatch.setattr(qr_mod, "_ORPHAN_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(qr_mod.os, "getppid", lambda: 1234)

    called = False

    def fake_exit() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(qr_mod, "_orphan_exit", fake_exit)

    stop = threading.Event()
    thread = threading.Thread(
        target=qr_mod._orphan_watchdog_thread,
        args=(1234, stop),
        daemon=True,
    )
    thread.start()
    time.sleep(0.05)  # let it poll several times
    stop.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert called is False


def test_orphan_exit_calls_os_exit_even_when_stderr_raises_broken_pipe() -> None:
    """Regression for the MCP-path BrokenPipeError case (stderr=PIPE).

    The MCP server spawns _quickresearch with stderr=PIPE (tools.py:903). When
    the parent dies, the pipe's read end closes and the next write raises
    BrokenPipeError. _orphan_exit must swallow that and still reach os._exit.
    """
    script = textwrap.dedent(
        """
        import sys
        from chunkhound.api.cli.commands.quickresearch import _orphan_exit

        class BrokenPipeStderr:
            def write(self, _s):
                raise BrokenPipeError(32, "Broken pipe")
            def flush(self):
                raise BrokenPipeError(32, "Broken pipe")

        sys.stderr = BrokenPipeStderr()
        _orphan_exit()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
    )
    assert result.returncode == 1
