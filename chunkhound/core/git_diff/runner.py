import asyncio
import re
from collections.abc import AsyncIterator, Awaitable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

_SAFE_REF = re.compile(r"^[a-zA-Z0-9_.^~/:@{}\-]+\Z")

_GIT_DIFF_TIMEOUT_SECONDS = 30

# Grace period for draining stderr once git has already exited.
_GIT_STDERR_READ_TIMEOUT_SECONDS = 5

# SHA1 of git's empty tree — used as the "no parent" base for root commits.
_EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

# Pattern matching <hash>^..<hash> produced by _resolve_commit_range for a
# single commit_hash.  Both capture groups must be identical.
# Accepts uppercase hex (git emits lowercase but accepts both) and up to 64
# chars to cover SHA256 object hashes as well as the standard SHA1 40-char form.
_SINGLE_COMMIT_RANGE_RE = re.compile(r"^([0-9a-fA-F]{4,64})\^\.\.([0-9a-fA-F]{4,64})\Z")


def _is_missing_parent_error(error: str) -> bool:
    return "unknown revision" in error or "bad revision" in error


def _validate_commit_range(commit_range: str) -> None:
    if (
        not _SAFE_REF.match(commit_range)
        or "../" in commit_range
        or commit_range.startswith("..")
        or commit_range.startswith("-")
    ):
        raise ValueError(f"Unsafe git ref rejected: {commit_range!r}")


async def _cleanup_git_diff_process(
    proc: asyncio.subprocess.Process, stderr_task: asyncio.Task[bytes]
) -> None:
    if proc.returncode is None:
        proc.kill()
        await proc.wait()
    if not stderr_task.done():
        stderr_task.cancel()
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass


async def stream_git_diff_file_blocks(
    commit_range: str, cwd: Path | str
) -> AsyncIterator[str]:
    """Yield one ``git diff`` file block at a time from subprocess stdout."""
    _validate_commit_range(commit_range)
    ranges = [commit_range]
    single_commit = _SINGLE_COMMIT_RANGE_RE.match(commit_range)

    while ranges:
        current_range = ranges.pop(0)
        proc = await asyncio.create_subprocess_exec(
            "git",
            "diff",
            current_range,
            "--",
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if proc.stdout is None or proc.stderr is None:
            proc.kill()
            await proc.wait()
            raise RuntimeError("git diff subprocess pipes were not created")

        stderr_task = asyncio.create_task(proc.stderr.read())
        block: list[str] = []
        loop = asyncio.get_running_loop()
        # Budget of time spent *blocked on git*, not wall clock. This generator
        # pauses at every yield while the caller embeds that file block, which
        # takes minutes on a large range; charging that to git's liveness guard
        # would make a big commit range kill its own git process.
        budget = float(_GIT_DIFF_TIMEOUT_SECONDS)

        async def await_git(awaitable: Awaitable[T]) -> T:
            nonlocal budget
            if budget <= 0:
                raise asyncio.TimeoutError
            started = loop.time()
            try:
                return await asyncio.wait_for(awaitable, timeout=budget)
            finally:
                budget -= loop.time() - started

        stderr = b""
        try:
            while True:
                line = await await_git(proc.stdout.readline())
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace")
                if decoded.startswith("diff --git ") and block:
                    yield "".join(block)
                    block = []
                block.append(decoded)
            if block:
                yield "".join(block)
            await await_git(proc.wait())
            # git has exited, so stderr is at EOF and this returns immediately.
            # Floor the timeout so draining it still works on a spent budget --
            # otherwise a timeout here would mask git's own error message.
            stderr = await asyncio.wait_for(
                stderr_task,
                timeout=max(budget, _GIT_STDERR_READ_TIMEOUT_SECONDS),
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"git diff timed out after {_GIT_DIFF_TIMEOUT_SECONDS}s"
                f" for range {current_range!r}"
            ) from None
        finally:
            await _cleanup_git_diff_process(proc, stderr_task)

        if proc.returncode == 0:
            return

        error = stderr.decode("utf-8", errors="replace").strip()
        if (
            current_range == commit_range
            and single_commit
            and single_commit.group(1) == single_commit.group(2)
            and _is_missing_parent_error(error)
        ):
            ranges.append(f"{_EMPTY_TREE_SHA}..{single_commit.group(2)}")
            continue
        raise ValueError(f"git diff failed: {error}")
