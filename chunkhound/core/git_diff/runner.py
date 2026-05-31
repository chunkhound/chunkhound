import asyncio
import re
from pathlib import Path

_SAFE_REF = re.compile(r'^[a-zA-Z0-9_.^~/:@{}\-]+\Z')

_GIT_DIFF_TIMEOUT_SECONDS = 30

# SHA1 of git's empty tree — used as the "no parent" base for root commits.
_EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

# Pattern matching <hash>^..<hash> produced by _resolve_commit_range for a
# single commit_hash.  Both capture groups must be identical.
_SINGLE_COMMIT_RANGE_RE = re.compile(r'^([0-9a-f]{4,40})\^\.\.([0-9a-f]{4,40})\Z')


async def run_git_diff(commit_range: str, cwd: Path | str) -> str:
    if (
        not _SAFE_REF.match(commit_range)
        or "../" in commit_range
        or commit_range.startswith("..")
        or commit_range.startswith("-")
    ):
        raise ValueError(f"Unsafe git ref rejected: {commit_range!r}")
    proc = await asyncio.create_subprocess_exec(
        "git", "diff", commit_range,
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
                "git", "diff", root_range,
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
                raise TimeoutError(
                    f"git diff timed out after {_GIT_DIFF_TIMEOUT_SECONDS}s"
                    f" for range {root_range!r}"
                )
            if proc2.returncode == 0:
                return stdout2.decode("utf-8", errors="replace")
            err = stderr2.decode("utf-8", errors="replace").strip()
        raise ValueError(f"git diff failed: {err}")
    return stdout.decode("utf-8", errors="replace")
