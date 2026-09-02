import asyncio
import re
from collections.abc import AsyncIterator
from pathlib import Path

_SAFE_REF = re.compile(r'^[a-zA-Z0-9_.^~/:@{}\-]+\Z')

_GIT_DIFF_TIMEOUT_SECONDS = 30

# SHA1 of git's empty tree — used as the "no parent" base for root commits.
_EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

# Pattern matching <hash>^..<hash> produced by _resolve_commit_range for a
# single commit_hash.  Both capture groups must be identical.
# Accepts uppercase hex (git emits lowercase but accepts both) and up to 64
# chars to cover SHA256 object hashes as well as the standard SHA1 40-char form.
_SINGLE_COMMIT_RANGE_RE = re.compile(r'^([0-9a-fA-F]{4,64})\^\.\.([0-9a-fA-F]{4,64})\Z')


def _validate_commit_range(commit_range: str) -> None:
    if (
        not _SAFE_REF.match(commit_range)
        or "../" in commit_range
        or commit_range.startswith("..")
        or commit_range.startswith("-")
    ):
        raise ValueError(f"Unsafe git ref rejected: {commit_range!r}")


async def run_git_diff(commit_range: str, cwd: Path | str) -> str:
    _validate_commit_range(commit_range)
    proc = await asyncio.create_subprocess_exec(
        "git", "diff", commit_range, "--",
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_GIT_DIFF_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(
            f"git diff timed out after {_GIT_DIFF_TIMEOUT_SECONDS}s"
            f" for range {commit_range!r}"
        )
    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        # Root commit has no parent: <hash>^..<hash> fails with "unknown revision".
        # Retry using the empty tree so `git diff EMPTY_TREE..<hash>` succeeds.
        m = _SINGLE_COMMIT_RANGE_RE.match(commit_range)
        if m and m.group(1) == m.group(2) and "unknown revision" in err:
            root_range = f"{_EMPTY_TREE_SHA}..{m.group(2)}"
            proc2 = await asyncio.create_subprocess_exec(
                "git", "diff", root_range, "--",
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout2, stderr2 = await asyncio.wait_for(
                    proc2.communicate(), timeout=_GIT_DIFF_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                proc2.kill()
                await proc2.wait()
                raise TimeoutError(
                    f"git diff timed out after {_GIT_DIFF_TIMEOUT_SECONDS}s"
                    f" for range {root_range!r}"
                )
            if proc2.returncode == 0:
                return stdout2.decode("utf-8", errors="replace")
            err = stderr2.decode("utf-8", errors="replace").strip()
        raise ValueError(f"git diff failed: {err}")
    return stdout.decode("utf-8", errors="replace")


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
        deadline = loop.time() + _GIT_DIFF_TIMEOUT_SECONDS
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace")
                if decoded.startswith("diff --git ") and block:
                    yield "".join(block)
                    block = []
                block.append(decoded)
            if block:
                yield "".join(block)
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(proc.wait(), timeout=remaining)
            stderr = await stderr_task
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stderr_task.cancel()
            raise TimeoutError(
                f"git diff timed out after {_GIT_DIFF_TIMEOUT_SECONDS}s"
                f" for range {current_range!r}"
            )
        except (asyncio.CancelledError, GeneratorExit):
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            stderr_task.cancel()
            raise

        if proc.returncode == 0:
            return

        error = stderr.decode("utf-8", errors="replace").strip()
        if (
            current_range == commit_range
            and single_commit
            and single_commit.group(1) == single_commit.group(2)
            and "unknown revision" in error
        ):
            ranges.append(f"{_EMPTY_TREE_SHA}..{single_commit.group(2)}")
            continue
        raise ValueError(f"git diff failed: {error}")
