"""Shared test helpers for mocking subprocess interactions."""

import asyncio
import sys
from collections.abc import Awaitable, Callable
from types import ModuleType

import pytest


async def cli_runs_after_cancel(
    monkeypatch: pytest.MonkeyPatch,
    provider_module: ModuleType,
    call: Callable[[], Awaitable[object]],
) -> bool:
    """Cancel ``call`` while its CLI (a real sleeping child) runs.

    Returns whether the child was still running after the cancellation;
    the child is reaped either way.
    """
    monkeypatch.setattr(
        provider_module,
        "build_cli_argv",
        lambda binary, *args: [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    real_exec = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []
    started = asyncio.Event()

    async def spawn(*args, **kwargs):
        process = await real_exec(*args, **kwargs)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(call())
    await asyncio.wait_for(started.wait(), timeout=30)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancellation was not re-raised")
    process = processes[0]
    still_running = process.returncode is None
    if still_running:
        process.kill()
        await process.wait()
    return still_running


class DummyPipe:
    """Mock stdin pipe for subprocess."""

    def __init__(self) -> None:
        self._buf = b""

    def write(self, data: bytes) -> None:
        self._buf += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


class DummyProc:
    """Mock asyncio subprocess process."""

    def __init__(
        self,
        rc: int = 0,
        out: bytes = b"OK",
        err: bytes = b"",
        stdin: DummyPipe | None = None,
    ) -> None:
        self.returncode = rc
        self._out = out
        self._err = err
        self.stdin = stdin

    async def communicate(self):
        return self._out, self._err

    def kill(self) -> None:
        return None

    async def wait(self) -> None:
        return None
