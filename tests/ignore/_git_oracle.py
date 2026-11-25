"""Shared git-as-oracle helpers for tests/ignore/ parity tests.

Helper-test mapping (Phase 1). Helper tests stay until Phase 6; the behavior
they protect is covered by these behavior-level tests:

- test_fnmatch_to_gitignore.py      -> ignore/test_ignore_matrix_new_cases.py
                                       (test_user_exclude_subtree_pattern_*)
- unit/test_prune_helpers.py        -> ignore/test_ignore_matrix_new_cases.py
                                       (test_user_exclude_subtree_pattern_*)
- unit/test_global_gitignore.py     -> ignore/test_global_excludes_end_to_end.py
- unit/test_discovery_backend_gitignore_prune.py
                                    -> parity tests (cache key is internal)
- ignore/test_detect_repo_roots_gitignore_prune.py
                                    -> ignore/test_git_parity*.py
- test_rust_scanner.py (fast-path decline)
                                    -> contracts/test_ignore_batch_realtime_
                                       consistency.py
- realtime/test_realtime_ignore_parity.py (log-once)
                                    -> KEPT: wrapper must still log errors once

Git is the oracle: decisions come from `git check-ignore --no-index`.
Engines are built through build_repo_aware_ignore_engine, so each ``backend``
(see conftest.py) is exercised the same way production builds it.
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
