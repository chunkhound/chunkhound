"""Tests for Unicode-safe stdout/stderr configuration in CLI entry points."""
import asyncio
import io
import sys

import pytest


class TrackingWriter(io.StringIO):
    """StringIO that records reconfigure() calls and starts with strict errors."""

    errors = "strict"
    encoding = "cp1252"

    def __init__(self):
        super().__init__()
        self.reconfigure_calls: list[dict] = []

    def reconfigure(self, **kwargs):
        self.reconfigure_calls.append(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


class NoReconfigureWriter(io.StringIO):
    """StringIO without a reconfigure() method."""


def _stub_asyncio_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_exit(coro):
        coro.close()
        raise SystemExit(0)

    monkeypatch.setattr(asyncio, "run", _raise_exit)


# ---------------------------------------------------------------------------
# CLI main() entry point
# ---------------------------------------------------------------------------


def test_main_sets_backslashreplace_on_windows(monkeypatch: pytest.MonkeyPatch):
    """main() reconfigures stdout/stderr with backslashreplace on Windows."""
    stdout, stderr = TrackingWriter(), TrackingWriter()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr("chunkhound.api.cli.main.IS_WINDOWS", True)
    _stub_asyncio_run(monkeypatch)

    from chunkhound.api.cli.main import main

    with pytest.raises(SystemExit):
        main()

    assert stdout.errors == "backslashreplace"
    assert stderr.errors == "backslashreplace"


def test_main_skips_reconfigure_on_non_windows(monkeypatch: pytest.MonkeyPatch):
    """main() must not reconfigure stdout/stderr on Linux/macOS."""
    stdout, stderr = TrackingWriter(), TrackingWriter()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr("chunkhound.api.cli.main.IS_WINDOWS", False)
    _stub_asyncio_run(monkeypatch)

    from chunkhound.api.cli.main import main

    with pytest.raises(SystemExit):
        main()

    assert stdout.reconfigure_calls == []
    assert stderr.reconfigure_calls == []


def test_main_no_crash_when_reconfigure_absent(monkeypatch: pytest.MonkeyPatch):
    """main() must not raise when streams lack reconfigure()."""
    monkeypatch.setattr(sys, "stdout", NoReconfigureWriter())
    monkeypatch.setattr(sys, "stderr", NoReconfigureWriter())
    monkeypatch.setattr("chunkhound.api.cli.main.IS_WINDOWS", True)
    _stub_asyncio_run(monkeypatch)

    from chunkhound.api.cli.main import main

    with pytest.raises(SystemExit):
        main()


# ---------------------------------------------------------------------------
# MCP main_sync() entry point
# ---------------------------------------------------------------------------


def test_main_sync_sets_backslashreplace_on_windows(monkeypatch: pytest.MonkeyPatch):
    """main_sync() reconfigures stdout/stderr with backslashreplace on Windows."""
    stdout, stderr = TrackingWriter(), TrackingWriter()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr("chunkhound.mcp_server.stdio.IS_WINDOWS", True)
    _stub_asyncio_run(monkeypatch)

    from chunkhound.mcp_server.stdio import main_sync

    with pytest.raises(SystemExit):
        main_sync()

    assert stdout.errors == "backslashreplace"
    assert stderr.errors == "backslashreplace"


def test_main_sync_skips_reconfigure_on_non_windows(monkeypatch: pytest.MonkeyPatch):
    """main_sync() must not reconfigure stdout/stderr on Linux/macOS."""
    stdout, stderr = TrackingWriter(), TrackingWriter()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr("chunkhound.mcp_server.stdio.IS_WINDOWS", False)
    _stub_asyncio_run(monkeypatch)

    from chunkhound.mcp_server.stdio import main_sync

    with pytest.raises(SystemExit):
        main_sync()

    assert stdout.reconfigure_calls == []
    assert stderr.reconfigure_calls == []
