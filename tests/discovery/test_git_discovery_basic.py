import os
import shutil
from pathlib import Path

import pytest

from chunkhound.utils.git_discovery import list_repo_files_via_git

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git binary required for git discovery tests"
)


def _init_repo(repo: Path) -> None:
    os.makedirs(repo, exist_ok=True)
    # Initialize a barebones repo
    assert shutil.which("git") is not None
    import subprocess

    subprocess.run(
        ["git", "init"],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_git_discovery_in_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    # .gitignore to exclude build/
    (repo / ".gitignore").write_text("build/\n", encoding="utf-8")
    # Files
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "build").mkdir(parents=True)
    (repo / "build" / "ignored.txt").write_text("ignore\n", encoding="utf-8")

    # Discover via git from the repo root
    files, _stats = list_repo_files_via_git(
        repo, repo, include_patterns=["**/*.py"], config_excludes=[]
    )
    rels = sorted([p.resolve().relative_to(repo).as_posix() for p in files])
    assert rels == ["src/a.py"]


def test_git_discovery_from_subdir(tmp_path: Path) -> None:
    repo = tmp_path / "repo2"
    _init_repo(repo)

    # Create structure
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "mod.py").write_text("print('x')\n", encoding="utf-8")
    (repo / "tools" / "gen.sh").parent.mkdir(parents=True)
    (repo / "tools" / "gen.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    start = repo / "src"
    files, _stats = list_repo_files_via_git(
        repo, start, include_patterns=["**/*.py"], config_excludes=[]
    )
    rels = sorted([p.resolve().relative_to(repo).as_posix() for p in files])
    assert rels == ["src/pkg/mod.py"]
