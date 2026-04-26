import os
import shutil
import subprocess
from pathlib import Path

import pytest

from chunkhound.utils.git_discovery import list_repo_files_via_git


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git required",
)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check,
    )


def _git_init(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init"], cwd=str(repo), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "CI")


def test_git_pathspec_cap_falls_back_to_subtree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)
    # create a couple of trivial files to keep git happy
    (repo / "a.py").write_text("print(1)\n")
    (repo / "b.txt").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")

    # Build many include patterns to exceed CAP
    includes = [f"**/*.{i:03d}x" for i in range(20)] + [f"**/name{i:03d}.cfg" for i in range(20)]

    # Force a small CAP to trigger the fallback
    monkeypatch.setenv("CHUNKHOUND_INDEXING__GIT_PATHSPEC_CAP", "2")

    files, stats = list_repo_files_via_git(repo, repo, include_patterns=includes, config_excludes=[])

    # We only care about counters here; coverage may be empty due to synthetic patterns
    assert isinstance(stats.get("git_pathspecs"), int)
    assert stats["git_pathspecs"] <= 2  # capped
    # Optional flag should be present when capped
    assert stats.get("git_pathspecs_capped", False) is True

