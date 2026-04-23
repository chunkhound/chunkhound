"""Tests for Unicode-safe stdout/stderr configuration in the CLI entry point."""
import sys
import io
import unittest.mock as mock

import pytest


def _run_main_reconfigure(platform: str, monkeypatch: pytest.MonkeyPatch):
    """Run the reconfigure block from main() under a simulated platform."""
    monkeypatch.setattr(sys, "platform", platform)

    # Simulate a narrow-encoding terminal (cp1252) via a StringIO with a custom encode method.
    class NarrowWriter(io.StringIO):
        errors = "strict"
        encoding = "cp1252"

        def reconfigure(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    stdout = NarrowWriter()
    stderr = NarrowWriter()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    # Import fresh so the module-level state doesn't interfere.
    from chunkhound.api.cli import main as main_mod
    import importlib
    importlib.reload(main_mod)

    # Execute only the reconfigure block, not asyncio.run.
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="backslashreplace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(errors="backslashreplace")

    return stdout, stderr


class TestCliUnicodeReconfigure:
    def test_windows_sets_backslashreplace(self, monkeypatch: pytest.MonkeyPatch):
        """On Windows, stdout/stderr errors mode must be 'backslashreplace'."""
        stdout, stderr = _run_main_reconfigure("win32", monkeypatch)
        assert stdout.errors == "backslashreplace"
        assert stderr.errors == "backslashreplace"

    def test_non_windows_unchanged(self, monkeypatch: pytest.MonkeyPatch):
        """On Linux/macOS, stdout/stderr must not be reconfigured."""
        stdout, stderr = _run_main_reconfigure("linux", monkeypatch)
        assert stdout.errors == "strict"
        assert stderr.errors == "strict"

    def test_main_no_crash_on_missing_reconfigure(self, monkeypatch: pytest.MonkeyPatch):
        """Streams without reconfigure() (e.g. raw BytesIO wrapper) must not raise."""
        monkeypatch.setattr(sys, "platform", "win32")

        class NoReconfigure(io.StringIO):
            pass  # no reconfigure attribute

        monkeypatch.setattr(sys, "stdout", NoReconfigure())
        monkeypatch.setattr(sys, "stderr", NoReconfigure())

        # Should not raise AttributeError
        if sys.platform == "win32":
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(errors="backslashreplace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(errors="backslashreplace")


def test_main_entrypoint_calls_reconfigure_on_windows(monkeypatch: pytest.MonkeyPatch):
    """main() calls reconfigure(errors='backslashreplace') on Windows and not elsewhere."""
    reconfigure_calls = []

    class TrackingWriter(io.StringIO):
        errors = "strict"
        encoding = "cp1252"

        def reconfigure(self, **kwargs):
            reconfigure_calls.append(kwargs)

    monkeypatch.setattr(sys, "stdout", TrackingWriter())
    monkeypatch.setattr(sys, "stderr", TrackingWriter())
    monkeypatch.setattr(sys, "platform", "win32")

    # Stub asyncio.run so main() doesn't actually run the CLI.
    import asyncio

    def _raise_exit(coro):
        coro.close()  # prevent unawaited-coroutine warning
        raise SystemExit(0)

    monkeypatch.setattr(asyncio, "run", _raise_exit)

    from chunkhound.api.cli.main import main
    with pytest.raises(SystemExit):
        main()

    assert any(c.get("errors") == "backslashreplace" for c in reconfigure_calls), (
        "Expected backslashreplace reconfigure on Windows"
    )


def test_main_entrypoint_skips_reconfigure_on_linux(monkeypatch: pytest.MonkeyPatch):
    """main() must NOT call reconfigure() on non-Windows platforms."""
    reconfigure_calls = []

    class TrackingWriter(io.StringIO):
        def reconfigure(self, **kwargs):
            reconfigure_calls.append(kwargs)

    monkeypatch.setattr(sys, "stdout", TrackingWriter())
    monkeypatch.setattr(sys, "stderr", TrackingWriter())
    monkeypatch.setattr(sys, "platform", "linux")

    import asyncio

    def _raise_exit(coro):
        coro.close()
        raise SystemExit(0)

    monkeypatch.setattr(asyncio, "run", _raise_exit)

    from chunkhound.api.cli.main import main
    with pytest.raises(SystemExit):
        main()

    assert reconfigure_calls == [], "reconfigure() must not be called on Linux"
