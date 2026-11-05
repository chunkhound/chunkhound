"""Tests for chunkhound.utils.stdio_lines.

The Windows-polling tests below exercise this contract:
`iter_stdin_lines()`'s Windows branch yields complete lines and cancels
promptly without waiting for stdin EOF.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

import chunkhound.utils.stdio_lines as stdio_lines_module
from chunkhound.utils.stdio_lines import iter_stdin_lines, write_stdout_line


@pytest.mark.asyncio
async def test_iter_stdin_lines_windows_forwards_json_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [b"not-json\n", b'{"method":"initialize"}\n', b""]

    stdin = SimpleNamespace(buffer=SimpleNamespace(fileno=lambda: 123))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(
        stdio_lines_module, "_windows_stdin_handle", lambda _fd: 456
    )
    monkeypatch.setattr(
        stdio_lines_module,
        "_windows_pipe_bytes_available",
        lambda _handle: len(chunks[0]) if chunks[0] else None,
    )
    monkeypatch.setattr(
        stdio_lines_module.os,
        "read",
        lambda _fd, _size: chunks.pop(0),
    )

    async def _collect() -> list[bytes]:
        lines = []
        async for line in iter_stdin_lines():
            lines.append(line)
        return lines

    lines = await asyncio.wait_for(_collect(), timeout=1.0)

    assert lines == [b"not-json\n", b'{"method":"initialize"}\n']


@pytest.mark.asyncio
async def test_iter_stdin_lines_windows_cancellation_does_not_wait_for_stdin_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_pipe = asyncio.Event()

    stdin = SimpleNamespace(buffer=SimpleNamespace(fileno=lambda: 123))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(
        stdio_lines_module, "_windows_stdin_handle", lambda _fd: 456
    )

    def no_data(_handle: int) -> int:
        checked_pipe.set()
        return 0

    monkeypatch.setattr(stdio_lines_module, "_windows_pipe_bytes_available", no_data)

    async def _consume() -> None:
        async for _line in iter_stdin_lines():
            pass

    task = asyncio.create_task(_consume())
    await asyncio.wait_for(checked_pipe.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)


def test_write_stdout_line_writes_and_flushes(monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[bytes] = []
    flushed = False

    class _FakeBuffer:
        def write(self, data: bytes) -> None:
            written.append(data)

        def flush(self) -> None:
            nonlocal flushed
            flushed = True

    # sys.stdout.buffer is a read-only property on the real TextIOWrapper
    # (confirmed: `sys.stdout.buffer = x` raises AttributeError even outside
    # pytest) -- replace the whole sys.stdout object instead of its attribute.
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=_FakeBuffer()))

    write_stdout_line(b'{"ok":true}\n')

    assert written == [b'{"ok":true}\n']
    assert flushed is True
