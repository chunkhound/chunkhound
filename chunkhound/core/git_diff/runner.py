import asyncio
import re
from pathlib import Path

_SAFE_REF = re.compile(r'^[a-zA-Z0-9_.^~/:@{}\-]+\Z')

_GIT_DIFF_TIMEOUT_SECONDS = 30


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
        raise ValueError(f"git diff failed: {stderr.decode('utf-8', errors='replace').strip()}")
    return stdout.decode("utf-8", errors="replace")
