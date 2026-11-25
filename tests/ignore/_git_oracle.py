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
                                    -> test_rust_scanner.py itself. Owner:
                                       test_heavy_dir_anchor_conflict_declines_
                                       rust_fast_path. NOT covered by
                                       contracts/test_ignore_batch_realtime_
                                       consistency.py, which uses only
                                       include=['**/*'] and no anchored pattern.
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chunkhound.utils.ignore_engine import (
        RepoAwareIgnoreEvaluator,
        RepoAwareLibgit2Evaluator,
    )


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


def write_file(repo: Path, rel: str, text: str = "x") -> None:
    """Create ``repo/rel`` (and parents) with ``text``, UTF-8."""
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def git_ignored(paths: Iterable[str], repo: Path) -> dict[str, bool]:
    """Return {path: is_ignored} from ``git check-ignore -q --no-index``."""
    out: dict[str, bool] = {}
    for p in paths:
        proc = run_git(["check-ignore", "-q", "--no-index", p], repo)
        out[p] = proc.returncode == 0
    return out


def assert_engine_backend(
    eng: RepoAwareIgnoreEvaluator | RepoAwareLibgit2Evaluator, backend: str
) -> None:
    """Fail loudly if the built engine is not the requested backend.

    libgit2 can silently degrade: when pygit2 imports but a repo fails to
    open, RepoAwareLibgit2Evaluator is still returned with fewer opened repos
    (ignore_engine.py, RepoAwareLibgit2Evaluator.__init__). Nothing falls back
    to python there, so the open-repo check is what catches it.
    """
    from chunkhound.utils.ignore_engine import (
        RepoAwareIgnoreEvaluator,
        RepoAwareLibgit2Evaluator,
    )

    expected = {
        "python": RepoAwareIgnoreEvaluator,
        "libgit2": RepoAwareLibgit2Evaluator,
    }[backend]
    assert type(eng) is expected, (
        f"requested backend {backend!r} but engine is {type(eng).__name__}"
    )
    if isinstance(eng, RepoAwareLibgit2Evaluator):
        opened = set(eng._repos)  # private: which pygit2 repos actually opened
        assert opened == set(eng.repo_roots), (
            f"libgit2 opened {len(opened)} of {len(eng.repo_roots)} repo(s); "
            "the engine would silently ignore nothing for the rest"
        )


def build_backend_engine(
    repo: Path,
    backend: str = "python",
    *,
    sources: Iterable[str] = ("gitignore",),
    config_exclude: Iterable[str] = (),
) -> RepoAwareIgnoreEvaluator | RepoAwareLibgit2Evaluator:
    """Build the repo-aware engine for ``backend`` and verify what was built."""
    from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine

    eng = build_repo_aware_ignore_engine(
        repo,
        sources=list(sources),
        chignore_file=".chignore",
        config_exclude=list(config_exclude),
        backend=backend,
    )
    assert_engine_backend(eng, backend)
    return eng


def engine_ignored(
    paths: Iterable[str],
    repo: Path,
    backend: str = "python",
    config_exclude: list[str] | None = None,
) -> dict[str, bool]:
    """Return {path: is_ignored} from the repo-aware engine for ``backend``."""
    eng = build_backend_engine(repo, backend, config_exclude=config_exclude or [])
    return {p: eng.matches(repo / p, is_dir=False) is not None for p in paths}
