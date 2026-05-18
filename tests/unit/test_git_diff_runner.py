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
async def test_timeout(tmp_path: Path) -> None:
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


def test_unsafe_ref_rejected() -> None:
    with pytest.raises(ValueError, match="Unsafe git ref rejected"):
        asyncio.get_event_loop().run_until_complete(
            run_git_diff("--output=/tmp/x", Path("/tmp"))
        )
