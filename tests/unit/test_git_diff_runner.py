"""Unit tests for chunkhound.core.git_diff.runner."""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.git_diff.runner import run_git_diff


class FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def kill(self) -> None:
        pass


def make_fake_process(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> FakeProcess:
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

    monkeypatch.setattr(runner_module, "_GIT_DIFF_TIMEOUT_SECONDS", 0.01)

    class SlowProcess:
        returncode = None

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(9999)
            return b"", b""

        def kill(self) -> None:
            pass

    with patch("asyncio.create_subprocess_exec", return_value=SlowProcess()):
        with pytest.raises(TimeoutError, match="timed out"):
            await run_git_diff("HEAD~1..HEAD", tmp_path)


@pytest.mark.asyncio
async def test_empty_diff(tmp_path: Path) -> None:
    fake = make_fake_process(stdout=b"", stderr=b"", returncode=0)
    with patch("asyncio.create_subprocess_exec", return_value=fake):
        result = await run_git_diff("HEAD~1..HEAD", tmp_path)
    assert result == ""


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
    root_success = make_fake_process(stdout=b"diff --git a/f b/f\n+hello", stderr=b"", returncode=0)

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
