from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

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
