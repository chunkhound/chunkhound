import asyncio
import re
from pathlib import Path

_SAFE_REF = re.compile(r'^[a-zA-Z0-9_.^~/:@{}\-]+$')


async def run_git_diff(commit_range: str, cwd: Path | str) -> str:
    if not _SAFE_REF.match(commit_range):
        raise ValueError(f"Unsafe git ref rejected: {commit_range!r}")
    proc = await asyncio.create_subprocess_exec(
        "git", "diff", commit_range,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        proc.kill()
        raise TimeoutError(f"git diff timed out after 30s for range {commit_range!r}")
    if proc.returncode != 0:
        raise ValueError(f"git diff failed: {stderr.decode('utf-8', errors='replace').strip()}")
    return stdout.decode("utf-8", errors="replace")
