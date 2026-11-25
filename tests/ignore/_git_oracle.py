"""Shared git-as-oracle helpers for tests/ignore/ parity tests.

Git decides what is ignored (``git check-ignore --no-index``); the ChunkHound
ignore engine must agree. Engines are built through the repo-aware entry point
so each ``backend`` (see conftest.py) is exercised the same way production
builds it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path


def run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.safecrlf=false",
            *args,
        ],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def git_ignored(paths: Iterable[str], repo: Path) -> dict[str, bool]:
    """Return {path: is_ignored} from ``git check-ignore -q --no-index``."""
    out: dict[str, bool] = {}
    for p in paths:
        proc = run_git(["check-ignore", "-q", "--no-index", p], repo)
        out[p] = proc.returncode == 0
    return out


def engine_ignored(
    paths: Iterable[str],
    repo: Path,
    backend: str = "python",
    config_exclude: list[str] | None = None,
) -> dict[str, bool]:
    """Return {path: is_ignored} from the repo-aware engine for ``backend``."""
    from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine

    eng = build_repo_aware_ignore_engine(
        repo,
        sources=["gitignore"],
        chignore_file=".chignore",
        config_exclude=config_exclude or [],
        backend=backend,
    )
    return {p: eng.matches(repo / p, is_dir=False) is not None for p in paths}
