from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git required for repo-root detection tests",
)


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    assert _git(["init"], repo).returncode == 0
    assert _git(["config", "user.email", "test@example.com"], repo).returncode == 0
    assert _git(["config", "user.name", "Test User"], repo).returncode == 0


def test_detect_repo_roots_prunes_root_gitignored_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    (repo / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (repo / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "ignored" / "deep" / "deeper").mkdir(parents=True, exist_ok=True)
    (repo / "ignored" / "deep" / "deeper" / "payload.txt").write_text(
        "x\n",
        encoding="utf-8",
    )

    assert _git(["add", ".gitignore", "main.py"], repo).returncode == 0
    assert _git(["commit", "-m", "init"], repo).returncode == 0

    from chunkhound.utils import ignore_engine as ignore_engine_module

    real_walk = os.walk
    visited: list[Path] = []

    def _recording_walk(top: str | os.PathLike[str], topdown: bool = True):
        for dirpath, dirnames, filenames in real_walk(top, topdown=topdown):
            visited.append(Path(dirpath).resolve())
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(ignore_engine_module.os, "walk", _recording_walk)

    roots = ignore_engine_module.detect_repo_roots(
        repo,
        prune_ignored_gitfile_roots=True,
    )

    assert roots == [repo.resolve()]

    ignored_root = (repo / "ignored").resolve()
    assert ignored_root not in visited
    assert not any(str(path).startswith(str(ignored_root) + os.sep) for path in visited)
