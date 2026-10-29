import asyncio
import codecs
import re
from collections.abc import AsyncIterator, Awaitable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

_SAFE_REF = re.compile(r"^[a-zA-Z0-9_.^~/:@{}\-]+\Z")

_GIT_DIFF_TIMEOUT_SECONDS = 30
_GIT_STDOUT_READ_BYTES = 64 * 1024

# Cleanup must not hang even if a descendant keeps a pipe open.
_GIT_CLEANUP_TIMEOUT_SECONDS = 5

# Grace period for draining stderr once git has already exited.
_GIT_STDERR_READ_TIMEOUT_SECONDS = 5

# Cap how much stderr we retain in memory for the eventual error message. A
# git process that writes a large volume of stderr should not be able to
# balloon memory just because stdout is streamed without a size cap.
_MAX_STDERR_BYTES = 64 * 1024

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


async def _drain_stderr(stream: asyncio.StreamReader) -> bytes:
    """Drain *stream* to EOF, retaining at most `_MAX_STDERR_BYTES`.

    Reads to EOF rather than stopping once the cap is hit, so git never
    blocks writing to a full stderr pipe just because we stopped reading it.
    """
    kept: list[bytes] = []
    kept_len = 0
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            return b"".join(kept)
        if kept_len < _MAX_STDERR_BYTES:
            remaining = _MAX_STDERR_BYTES - kept_len
            kept.append(chunk[:remaining])
            kept_len += min(len(chunk), remaining)


async def _read_stdout_chunk(stream: asyncio.StreamReader) -> bytes:
    """Read a bounded stdout block without imposing a line-length limit."""
    return await stream.read(_GIT_STDOUT_READ_BYTES)


async def _cleanup_git_diff_process(
    proc: asyncio.subprocess.Process, stderr_task: asyncio.Task[bytes]
) -> None:
    async def drain_stdout() -> None:
        if proc.stdout is not None:
            while await proc.stdout.read(8192):
                pass

    try:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass  # The child exited between checking and signalling.
        # Process.wait() can remain blocked after exit while the stdout
        # transport is paused on a full buffer. Drain concurrently, in bounded
        # blocks, including when returncode was already set before cleanup.
        cleanup = asyncio.gather(drain_stdout(), proc.wait())
        try:
            await asyncio.wait_for(cleanup, _GIT_CLEANUP_TIMEOUT_SECONDS)
        except (asyncio.TimeoutError, OSError):
            cleanup.cancel()
        finally:
            await asyncio.gather(cleanup, return_exceptions=True)
    finally:
        if not stderr_task.done():
            stderr_task.cancel()
        await asyncio.gather(stderr_task, return_exceptions=True)


async def stream_git_diff_fragments(
    commit_range: str, cwd: Path | str
) -> AsyncIterator[str]:
    """Yield bounded decoded fragments from ``git diff`` stdout."""
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

        stderr_task = asyncio.create_task(_drain_stderr(proc.stderr))
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
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
                data = await await_git(_read_stdout_chunk(proc.stdout))
                if not data:
                    break
                decoded = decoder.decode(data)
                if decoded:
                    yield decoded
            decoded = decoder.decode(b"", final=True)
            if decoded:
                yield decoded
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


async def stream_git_diff_file_blocks(
    commit_range: str, cwd: Path | str
) -> AsyncIterator[str]:
    """Compatibility wrapper yielding complete file blocks.

    Production search consumes :func:`stream_git_diff_fragments` directly so
    its memory does not scale with the largest changed file.
    """
    block: list[str] = []
    pending = ""
    fragments = stream_git_diff_fragments(commit_range, cwd)
    try:
        async for fragment in fragments:
            pending += fragment
            lines = pending.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                pending = lines.pop()
            else:
                pending = ""
            for line in lines:
                if line.startswith("diff --git ") and block:
                    yield "".join(block)
                    block = []
                block.append(line)
        if pending:
            if pending.startswith("diff --git ") and block:
                yield "".join(block)
                block = []
            block.append(pending)
        if block:
            yield "".join(block)
    finally:
        close = getattr(fragments, "aclose", None)
        if callable(close):
            await close()
