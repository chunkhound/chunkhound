from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chunkhound.services.realtime.adapters.watchman import WatchmanRealtimeAdapter
from chunkhound.watchman import PrivateWatchmanSidecar, WatchmanSubscriptionScope
from chunkhound.watchman.session import WatchmanCliSession
from chunkhound.watchman_runtime import bridge as bridge_module


class _ExplodingStream:
    async def readline(self) -> bytes:
        raise RuntimeError("stderr exploded")


class _DummyProcess:
    def __init__(self) -> None:
        self.stderr = _ExplodingStream()
        self.pid = None
        self.returncode = None


@pytest.mark.asyncio
async def test_watchman_session_stderr_loop_records_reader_failures() -> None:
    session = WatchmanCliSession(
        binary_path=Path("/tmp/watchman"),
        socket_path="/tmp/watchman.sock",
        statefile_path=Path("/tmp/watchman.state"),
        logfile_path=Path("/tmp/watchman.log"),
        pidfile_path=Path("/tmp/watchman.pid"),
        project_root=Path("/tmp"),
    )
    session._process = _DummyProcess()
    pending_reply = asyncio.get_running_loop().create_future()
    session._pending_reply = pending_reply

    await session._stderr_loop()

    assert session._last_error == (
        "Watchman stderr reader failed: stderr exploded"
    )
    assert pending_reply.done()
    with pytest.raises(RuntimeError, match="stderr exploded"):
        pending_reply.result()


def test_runtime_subscription_stop_warns_when_observer_thread_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning_messages: list[str] = []

    class _FakeObserver:
        def stop(self) -> None:
            return None

        def join(self, timeout: float) -> None:
            assert timeout == 5.0

        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr(bridge_module, "_stderr", warning_messages.append)

    subscription = bridge_module.RuntimeSubscription(
        subscription_name="chunkhound-live-indexing",
        watch_root=Path("/tmp"),
        relative_root=None,
        event_queue=bridge_module.queue.Queue(),
    )
    subscription._observer = _FakeObserver()

    subscription.stop()

    assert warning_messages == [
        "chunkhound watchman runtime: observer thread did not stop within 5.0s"
    ]


def test_watchman_subscribe_command_requests_directory_entries(tmp_path: Path) -> None:
    session = WatchmanCliSession(
        binary_path=tmp_path / "watchman",
        socket_path=tmp_path / "watchman.sock",
        statefile_path=tmp_path / "watchman.state",
        logfile_path=tmp_path / "watchman.log",
        pidfile_path=tmp_path / "watchman.pid",
        project_root=tmp_path,
    )
    scope = WatchmanSubscriptionScope(
        requested_path=tmp_path,
        watch_root=tmp_path,
        relative_root=None,
        scope_kind="primary",
    )

    command = session._build_subscribe_command(
        scope=scope,
        subscription_name="chunkhound-live-indexing",
    )

    assert command[:3] == ["subscribe", str(tmp_path), "chunkhound-live-indexing"]
    payload = command[3]
    assert isinstance(payload, dict)
    assert "expression" not in payload
    assert payload["fields"] == ["name", "exists", "new", "type"]
    assert payload["empty_on_fresh_instance"] is True


def test_watchman_adapter_translates_deleted_directory_entries(tmp_path: Path) -> None:
    adapter = object.__new__(WatchmanRealtimeAdapter)
    scope = WatchmanSubscriptionScope(
        requested_path=tmp_path,
        watch_root=tmp_path,
        relative_root=None,
        scope_kind="primary",
    )

    translated = adapter._translate_watchman_file_entry(
        {
            "name": "src/deleted_dir",
            "exists": False,
            "new": False,
            "type": "d",
        },
        scope,
    )

    assert translated == ("dir_deleted", tmp_path / "src" / "deleted_dir")


def test_watchman_adapter_skips_existing_directory_change_entries(
    tmp_path: Path,
) -> None:
    adapter = object.__new__(WatchmanRealtimeAdapter)
    scope = WatchmanSubscriptionScope(
        requested_path=tmp_path,
        watch_root=tmp_path,
        relative_root=None,
        scope_kind="primary",
    )

    translated = adapter._translate_watchman_file_entry(
        {
            "name": "src/existing_dir",
            "exists": True,
            "new": False,
            "type": "d",
        },
        scope,
    )

    assert translated is None


@pytest.mark.asyncio
async def test_cleanup_stale_state_ignores_failed_log_without_metadata(
    tmp_path: Path,
) -> None:
    sidecar = PrivateWatchmanSidecar(tmp_path / "repo")
    sidecar.paths.root.mkdir(parents=True, exist_ok=True)
    failed_log_path = sidecar.paths.logfile_path.with_name("watchman.failed.log")
    failed_log_path.write_text("watchman startup failed\n", encoding="utf-8")

    result = await sidecar.cleanup_stale_state()

    assert result is None
    assert failed_log_path.exists()
