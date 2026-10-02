"""Unit tests for chunkhound.core.git_diff.runner."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from chunkhound.core.git_diff.runner import (
    _MAX_STDERR_BYTES,
    stream_git_diff_file_blocks,
)


class _LineStdout:
    def __init__(
        self, lines: list[bytes], hang: bool = False, error: Exception | None = None
    ) -> None:
        self._lines = list(lines)
        self._hang = hang
        self._error = error

    async def readline(self) -> bytes:
        if self._error is not None:
            raise self._error
        if self._lines:
            return self._lines.pop(0)
        if self._hang:
            await asyncio.Event().wait()
        return b""

    async def readuntil(self, separator: bytes = b"\n") -> bytes:
        # This test double always hands out complete, pre-terminated lines
        # (or a clean EOF), so it never needs to exercise the
        # LimitOverrunError recovery path -- that's covered separately by
        # test_stream_reassembles_line_longer_than_reader_limit against a
        # real asyncio.StreamReader.
        line = await self.readline()
        if not line:
            raise asyncio.IncompleteReadError(b"", None)
        return line


class _Stderr:
    def __init__(self, payload: bytes = b"", hang: bool = False) -> None:
        self._payload = payload
        self._hang = hang
        self.read_calls = 0

    async def read(self, n: int = -1) -> bytes:
        self.read_calls += 1
        if self._hang:
            await asyncio.Event().wait()
        if n is None or n < 0:
            chunk, self._payload = self._payload, b""
            return chunk
        chunk, self._payload = self._payload[:n], self._payload[n:]
        return chunk


class StreamProcess:
    def __init__(
        self,
        lines: list[bytes],
        hang_stdout: bool = False,
        hang_stderr: bool = False,
        error: Exception | None = None,
        exit_code: int = 0,
        stderr: bytes = b"",
    ) -> None:
        self.stdout = _LineStdout(lines, hang=hang_stdout, error=error)
        self.stderr = _Stderr(payload=stderr, hang=hang_stderr)
        self.returncode: int | None = None
        self.killed = False
        self._exit_code = exit_code

    def kill(self) -> None:
        self.killed = True
        self.returncode = -1

    async def wait(self) -> None:
        if self.returncode is None:
            self.returncode = self._exit_code


@pytest.mark.asyncio
async def test_streams_complete_file_blocks_from_real_git(
    tmp_path: Path,
) -> None:
    proc = await asyncio.create_subprocess_exec(
        "git", "init", cwd=str(tmp_path), stdout=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    for name in ("one.py", "two.py"):
        (tmp_path / name).write_text(f"{name} = 1\n", encoding="utf-8")
    proc = await asyncio.create_subprocess_exec("git", "add", ".", cwd=str(tmp_path))
    await proc.communicate()
    proc = await asyncio.create_subprocess_exec(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
        cwd=str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    for name in ("one.py", "two.py"):
        (tmp_path / name).write_text(f"{name} = 2\n", encoding="utf-8")

    blocks = [
        block async for block in stream_git_diff_file_blocks("HEAD", cwd=tmp_path)
    ]

    assert len(blocks) == 2
    assert all(block.startswith("diff --git ") for block in blocks)
    assert "one.py" in blocks[0]
    assert "two.py" in blocks[1]


@pytest.mark.asyncio
async def test_stream_reassembles_line_longer_than_reader_limit(
    tmp_path: Path,
) -> None:
    """A single diff line longer than asyncio's default 64 KiB reader limit
    must not truncate the stream or raise -- diffs can legitimately contain
    very long single lines (minified bundles, generated lockfiles)."""
    proc = await asyncio.create_subprocess_exec(
        "git", "init", cwd=str(tmp_path), stdout=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    (tmp_path / "bundle.js").write_text("var x = 1;\n", encoding="utf-8")
    proc = await asyncio.create_subprocess_exec("git", "add", ".", cwd=str(tmp_path))
    await proc.communicate()
    proc = await asyncio.create_subprocess_exec(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
        cwd=str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
    )
    await proc.communicate()

    long_line = "x" * 200_000  # comfortably over the 64 KiB default reader limit
    (tmp_path / "bundle.js").write_text(f"var x = 1;\n{long_line}\n", encoding="utf-8")

    blocks = [
        block async for block in stream_git_diff_file_blocks("HEAD", cwd=tmp_path)
    ]

    assert len(blocks) == 1
    assert f"+{long_line}" in blocks[0]


@pytest.mark.asyncio
async def test_stream_root_commit_uses_empty_tree(tmp_path: Path) -> None:
    proc = await asyncio.create_subprocess_exec(
        "git", "init", cwd=str(tmp_path), stdout=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    (tmp_path / "root.py").write_text("root = True\n", encoding="utf-8")
    proc = await asyncio.create_subprocess_exec("git", "add", ".", cwd=str(tmp_path))
    await proc.communicate()
    proc = await asyncio.create_subprocess_exec(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "root",
        cwd=str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    proc = await asyncio.create_subprocess_exec(
        "git",
        "rev-parse",
        "HEAD",
        cwd=str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    root_hash = stdout.decode().strip()

    blocks = [
        block
        async for block in stream_git_diff_file_blocks(
            f"{root_hash}^..{root_hash}", cwd=tmp_path
        )
    ]

    assert len(blocks) == 1
    assert blocks[0].startswith("diff --git ")
    assert "root.py" in blocks[0]
    assert "+root = True" in blocks[0]


@pytest.mark.asyncio
async def test_unsafe_ref_rejected() -> None:
    with pytest.raises(ValueError, match="Unsafe git ref rejected"):
        async for _ in stream_git_diff_file_blocks("--output=/tmp/x", cwd=Path("/tmp")):
            pass


@pytest.mark.asyncio
async def test_option_injection_rejected() -> None:
    """Git options must be rejected even though they pass the char regex."""
    for bad_ref in ("--cached", "--staged", "-p", "--no-index"):
        with pytest.raises(ValueError, match="Unsafe git ref rejected"):
            async for _ in stream_git_diff_file_blocks(bad_ref, cwd=Path("/tmp")):
                pass


@pytest.mark.asyncio
async def test_root_commit_retry_also_fails(tmp_path: Path) -> None:
    """If the empty-tree retry also fails, the error from the retry is raised."""
    commit_hash = "b" * 40
    procs = [
        StreamProcess(lines=[], exit_code=128, stderr=b"fatal: unknown revision bbbb^"),
        StreamProcess(lines=[], exit_code=128, stderr=b"fatal: not a git repository"),
    ]
    call_count = 0

    async def fake_exec(*args: object, **kwargs: object) -> StreamProcess:
        nonlocal call_count
        call_count += 1
        return procs.pop(0)

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        with pytest.raises(ValueError, match="not a git repository"):
            async for _ in stream_git_diff_file_blocks(
                f"{commit_hash}^..{commit_hash}", cwd=tmp_path
            ):
                pass

    assert call_count == 2


@pytest.mark.asyncio
async def test_non_root_failure_not_retried(tmp_path: Path) -> None:
    """Unrelated git failures (no 'unknown revision') are not retried."""
    commit_hash = "c" * 40
    call_count = 0

    async def fake_exec(*args: object, **kwargs: object) -> StreamProcess:
        nonlocal call_count
        call_count += 1
        return StreamProcess(
            lines=[], exit_code=128, stderr=b"fatal: bad object HEAD~999"
        )

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        with pytest.raises(ValueError, match="bad object"):
            async for _ in stream_git_diff_file_blocks(
                f"{commit_hash}^..{commit_hash}", cwd=tmp_path
            ):
                pass

    assert call_count == 1


@pytest.mark.asyncio
async def test_stream_caps_stderr_retained_in_memory(tmp_path: Path) -> None:
    """A pathologically large stderr payload must not be fully buffered."""
    huge_stderr = b"fatal: bad object HEAD~999 " + b"x" * (10 * _MAX_STDERR_BYTES)
    proc = StreamProcess(lines=[], exit_code=128, stderr=huge_stderr)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(ValueError) as exc_info:
            async for _ in stream_git_diff_file_blocks("HEAD", cwd=tmp_path):
                pass

    message = str(exc_info.value)
    assert "fatal: bad object" in message
    assert len(message) <= _MAX_STDERR_BYTES + len("git diff failed: ")


@pytest.mark.asyncio
async def test_stream_stderr_drain_reaches_eof_past_the_cap(
    tmp_path: Path,
) -> None:
    """Draining must continue past the cap so git never blocks on a full pipe."""
    huge_stderr = b"x" * (10 * _MAX_STDERR_BYTES)
    proc = StreamProcess(lines=[], exit_code=128, stderr=huge_stderr)

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(ValueError):
            async for _ in stream_git_diff_file_blocks("HEAD", cwd=tmp_path):
                pass

    assert proc.stderr._payload == b""
    assert proc.stderr.read_calls > 1


@pytest.mark.asyncio
async def test_stream_timeout_kills_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chunkhound.core.git_diff.runner as runner_module

    monkeypatch.setattr(runner_module, "_GIT_DIFF_TIMEOUT_SECONDS", 0.05)
    proc = StreamProcess(lines=[], hang_stdout=True, hang_stderr=True)
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(TimeoutError, match="timed out"):
            async for _ in stream_git_diff_file_blocks("HEAD", cwd=tmp_path):
                pass
    assert proc.killed is True


@pytest.mark.asyncio
async def test_stream_cancel_kills_process(tmp_path: Path) -> None:
    proc = StreamProcess(lines=[], hang_stdout=True, hang_stderr=True)

    async def consume() -> None:
        async for _ in stream_git_diff_file_blocks("HEAD", cwd=tmp_path):
            pass

    with patch("asyncio.create_subprocess_exec", return_value=proc):
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert proc.killed is True


@pytest.mark.asyncio
async def test_stream_unexpected_io_kills_process(tmp_path: Path) -> None:
    proc = StreamProcess(lines=[], error=OSError("pipe closed"), hang_stderr=True)
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(OSError, match="pipe closed"):
            async for _ in stream_git_diff_file_blocks("HEAD", cwd=tmp_path):
                pass
    assert proc.killed is True


@pytest.mark.asyncio
async def test_stream_budget_excludes_consumer_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow consumer (embedding a batch) must not trip git's liveness guard.

    The generator pauses at each yield while the caller embeds that file block.
    That time is not spent waiting on git, so it must not count against the
    timeout — otherwise a large commit range kills its own git process.
    """
    import chunkhound.core.git_diff.runner as runner_module

    monkeypatch.setattr(runner_module, "_GIT_DIFF_TIMEOUT_SECONDS", 0.2)
    proc = StreamProcess(
        lines=[
            b"diff --git a/a.py b/a.py\n",
            b"+first\n",
            b"diff --git a/b.py b/b.py\n",
            b"+second\n",
        ]
    )
    blocks: list[str] = []
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        async for block in stream_git_diff_file_blocks("HEAD~1..HEAD", cwd=tmp_path):
            blocks.append(block)
            await asyncio.sleep(0.15)  # 2 x 0.15s = 0.3s > the 0.2s git budget

    assert len(blocks) == 2
    assert "a.py" in blocks[0]
    assert "b.py" in blocks[1]
