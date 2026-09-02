"""Unit tests for chunkhound.core.git_diff.runner."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from chunkhound.core.git_diff.runner import run_git_diff, stream_git_diff_file_blocks


class FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def kill(self) -> None:
        pass


def make_fake_process(
    stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0
) -> FakeProcess:
    return FakeProcess(stdout, stderr, returncode)


@pytest.mark.asyncio
async def test_happy_path(tmp_path: Path) -> None:
    fake = make_fake_process(stdout=b"diff content", stderr=b"", returncode=0)
    with patch("asyncio.create_subprocess_exec", return_value=fake):
        result = await run_git_diff("HEAD~1..HEAD", tmp_path)
    assert result == "diff content"


@pytest.mark.asyncio
async def test_nonzero_returncode(tmp_path: Path) -> None:
    fake = make_fake_process(stdout=b"", stderr=b"fatal: bad object", returncode=128)
    with patch("asyncio.create_subprocess_exec", return_value=fake):
        with pytest.raises(ValueError, match="git diff failed"):
            await run_git_diff("HEAD~1..HEAD", tmp_path)


@pytest.mark.asyncio
async def test_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import chunkhound.core.git_diff.runner as runner_module

    monkeypatch.setattr(runner_module, "_GIT_DIFF_TIMEOUT_SECONDS", 0.5)

    class SlowProcess:
        returncode = None

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(9999)
            return b"", b""

        def kill(self) -> None:
            pass

        async def wait(self) -> None:
            pass

    with patch("asyncio.create_subprocess_exec", return_value=SlowProcess()):
        with pytest.raises(TimeoutError, match="timed out"):
            await run_git_diff("HEAD~1..HEAD", tmp_path)


@pytest.mark.asyncio
async def test_run_git_diff_cancellation_kills_process(tmp_path: Path) -> None:
    class BlockingProcess:
        returncode = None

        def __init__(self) -> None:
            self.killed = False

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.Event().wait()
            return b"", b""

        def kill(self) -> None:
            self.killed = True
            self.returncode = -1

        async def wait(self) -> None:
            return None

    proc = BlockingProcess()
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        task = asyncio.create_task(run_git_diff("HEAD", tmp_path))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert proc.killed is True


@pytest.mark.asyncio
async def test_empty_diff(tmp_path: Path) -> None:
    fake = make_fake_process(stdout=b"", stderr=b"", returncode=0)
    with patch("asyncio.create_subprocess_exec", return_value=fake):
        result = await run_git_diff("HEAD~1..HEAD", tmp_path)
    assert result == ""


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
        await run_git_diff("--output=/tmp/x", Path("/tmp"))


@pytest.mark.asyncio
async def test_option_injection_rejected() -> None:
    """--cached and other git options must be rejected even though they pass the char regex."""
    for bad_ref in ("--cached", "--staged", "-p", "--no-index"):
        with pytest.raises(ValueError, match="Unsafe git ref rejected"):
            await run_git_diff(bad_ref, Path("/tmp"))


@pytest.mark.asyncio
async def test_root_commit_uses_empty_tree(tmp_path: Path) -> None:
    """<hash>^..<hash> failing with 'unknown revision' triggers empty-tree retry."""
    HASH = "a" * 40
    EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    root_fail = make_fake_process(
        stdout=b"",
        stderr=b"fatal: ambiguous argument 'aaaa^': unknown revision or path",
        returncode=128,
    )
    root_success = make_fake_process(
        stdout=b"diff --git a/f b/f\n+hello", stderr=b"", returncode=0
    )

    call_count = 0

    async def fake_exec(*args: object, **kwargs: object) -> FakeProcess:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return root_fail
        return root_success

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await run_git_diff(f"{HASH}^..{HASH}", tmp_path)

    assert "hello" in result
    assert call_count == 2
    # Verify second call used empty tree SHA
    second_call_args = None
    call_count = 0

    async def fake_exec_capture(*args: object, **kwargs: object) -> FakeProcess:
        nonlocal call_count, second_call_args
        call_count += 1
        if call_count == 1:
            return root_fail
        second_call_args = args
        return root_success

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec_capture):
        await run_git_diff(f"{HASH}^..{HASH}", tmp_path)

    assert second_call_args is not None
    # Range is a single string arg like "4b825d...aaaa..." — check substring
    range_arg = second_call_args[2]  # ("git", "diff", "<range>", ...)
    assert EMPTY_TREE in range_arg
    assert HASH in range_arg


@pytest.mark.asyncio
async def test_root_commit_retry_also_fails(tmp_path: Path) -> None:
    """If empty-tree retry also fails, the error from the retry is raised."""
    HASH = "b" * 40
    root_fail = make_fake_process(
        stdout=b"",
        stderr=b"fatal: unknown revision bbbb^",
        returncode=128,
    )
    retry_fail = make_fake_process(
        stdout=b"",
        stderr=b"fatal: not a git repository",
        returncode=128,
    )

    call_count = 0

    async def fake_exec(*args: object, **kwargs: object) -> FakeProcess:
        nonlocal call_count
        call_count += 1
        return root_fail if call_count == 1 else retry_fail

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        with pytest.raises(ValueError, match="git diff failed"):
            await run_git_diff(f"{HASH}^..{HASH}", tmp_path)

    assert call_count == 2


@pytest.mark.asyncio
async def test_non_root_failure_not_retried(tmp_path: Path) -> None:
    """Unrelated git failures (no 'unknown revision') are not retried."""
    HASH = "c" * 40
    fail = make_fake_process(
        stdout=b"",
        stderr=b"fatal: bad object HEAD~999",
        returncode=128,
    )

    call_count = 0

    async def fake_exec(*args: object, **kwargs: object) -> FakeProcess:
        nonlocal call_count
        call_count += 1
        return fail

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        with pytest.raises(ValueError, match="git diff failed"):
            await run_git_diff(f"{HASH}^..{HASH}", tmp_path)

    assert call_count == 1  # no retry


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


class _Stderr:
    def __init__(self, hang: bool = False) -> None:
        self._hang = hang

    async def read(self) -> bytes:
        if self._hang:
            await asyncio.Event().wait()
        return b""


class StreamProcess:
    def __init__(
        self,
        lines: list[bytes],
        hang_stdout: bool = False,
        hang_stderr: bool = False,
        error: Exception | None = None,
    ) -> None:
        self.stdout = _LineStdout(lines, hang=hang_stdout, error=error)
        self.stderr = _Stderr(hang=hang_stderr)
        self.returncode: int | None = None
        self.killed = False

    def kill(self) -> None:
        self.killed = True
        self.returncode = -1

    async def wait(self) -> None:
        if self.returncode is None:
            self.returncode = 0


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
