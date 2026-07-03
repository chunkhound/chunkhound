import builtins
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.services.realtime.events import SimpleEventHandler


def _clear_watchdog_sensitive_modules() -> None:
    prefixes = (
        "watchdog",
        "chunkhound.daemon.server",
        "chunkhound.mcp_server.base",
        "chunkhound.services.realtime",
    )
    for name in list(sys.modules):
        if name.startswith(prefixes):
            sys.modules.pop(name, None)


def test_daemon_server_import_does_not_touch_watchdog(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.startswith("watchdog"):
            raise AssertionError(f"unexpected watchdog import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    _clear_watchdog_sensitive_modules()
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("chunkhound.daemon.server")

    assert module.ChunkHoundDaemon is not None


def test_simple_event_handler_dispatches_without_watchdog_base(tmp_path: Path) -> None:
    queued: list[tuple[str, Path, bool, str | None]] = []
    handler = SimpleEventHandler(
        event_queue=None,
        loop=None,
        root_path=tmp_path,
        queue_result_callback=lambda *args: queued.append(args),
    )
    event = SimpleNamespace(
        event_type="created",
        is_directory=False,
        src_path=str(tmp_path / "file.py"),
    )

    handler.dispatch(event)

    assert queued == [
        (
            "created",
            (tmp_path / "file.py").resolve(),
            False,
            "loop_unavailable",
        )
    ]


def test_legacy_realtime_indexing_service_module_no_longer_exists() -> None:
    """The realtime_indexing_service compatibility shim has been removed."""
    with pytest.raises(ImportError):
        importlib.import_module("chunkhound.services.realtime_indexing_service")
