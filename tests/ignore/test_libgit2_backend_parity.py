from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check,
    )


def _git_ignored(repo: Path, rel_path: str) -> bool:
    p = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "-q", "--no-index", rel_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode == 0


def test_libgit2_backend_matches_git_when_available(tmp_path: Path):
    try:
        import pygit2  # noqa: F401
    except Exception:
        pytest.skip("pygit2 not available")

    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init"],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Patterns: ignore foo/ but not bar/
    (repo / ".gitignore").write_text("foo/\n", encoding="utf-8")
    (repo / "foo").mkdir(parents=True, exist_ok=True)
    (repo / "foo" / "x.txt").write_text("x\n", encoding="utf-8")
    (repo / "bar").mkdir(parents=True, exist_ok=True)
    (repo / "bar" / "y.py").write_text("print('y')\n", encoding="utf-8")

    from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine

    eng = build_repo_aware_ignore_engine(
        root=repo,
        sources=["gitignore"],
        chignore_file=".chignore",
        config_exclude=[],
        backend="libgit2",
    )

    a = eng.matches(repo / "foo" / "x.txt", is_dir=False) is not None
    b = _git_ignored(repo, "foo/x.txt")
    assert a == b

    a2 = eng.matches(repo / "bar" / "y.py", is_dir=False) is not None
    b2 = _git_ignored(repo, "bar/y.py")
    assert a2 == b2
