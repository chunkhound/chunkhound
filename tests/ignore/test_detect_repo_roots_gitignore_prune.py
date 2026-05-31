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


def _detect_with_recorded_walk(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prune_ignored_gitfile_roots: bool = False,
    workspace_root_only_gitignore: bool = False,
) -> tuple[list[Path], list[Path]]:
    from chunkhound.utils import ignore_engine as ignore_engine_module

    real_walk = os.walk
    visited: list[Path] = []

    def _recording_walk(top: str | os.PathLike[str], topdown: bool = True):
        for dirpath, dirnames, filenames in real_walk(top, topdown=topdown):
            visited.append(Path(dirpath).resolve())
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(ignore_engine_module.os, "walk", _recording_walk)
    roots = ignore_engine_module.detect_repo_roots(
        root,
        prune_ignored_gitfile_roots=prune_ignored_gitfile_roots,
        workspace_root_only_gitignore=workspace_root_only_gitignore,
    )
    return roots, visited


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

    roots, visited = _detect_with_recorded_walk(
        repo,
        monkeypatch,
        prune_ignored_gitfile_roots=True,
    )

    assert roots == [repo.resolve()]

    ignored_root = (repo / "ignored").resolve()
    assert ignored_root not in visited
    assert not any(str(path).startswith(str(ignored_root) + os.sep) for path in visited)


def test_detect_repo_roots_prunes_nested_gitignore_scopes_without_root_patterns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    (repo / "data" / ".gitignore").parent.mkdir(parents=True, exist_ok=True)
    (repo / "data" / ".gitignore").write_text("binaries/\n", encoding="utf-8")
    (repo / "data" / "keep.txt").write_text("keep\n", encoding="utf-8")
    (repo / "data" / "binaries" / "deep" / "payload.bin").parent.mkdir(
        parents=True, exist_ok=True
    )
    (repo / "data" / "binaries" / "deep" / "payload.bin").write_text(
        "x\n",
        encoding="utf-8",
    )

    roots, visited = _detect_with_recorded_walk(
        repo,
        monkeypatch,
        prune_ignored_gitfile_roots=True,
    )

    assert roots == [repo.resolve()]

    ignored_root = (repo / "data" / "binaries").resolve()
    assert ignored_root not in visited
    assert not any(str(path).startswith(str(ignored_root) + os.sep) for path in visited)


def test_detect_repo_roots_prunes_nonrepo_workspace_gitignore_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / ".gitignore").write_text("datasets/\n", encoding="utf-8")
    (workspace / "datasets" / "raw" / "payload.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (workspace / "datasets" / "raw" / "payload.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    repo = workspace / "repo"
    _init_repo(repo)

    roots, visited = _detect_with_recorded_walk(
        workspace,
        monkeypatch,
        prune_ignored_gitfile_roots=True,
        workspace_root_only_gitignore=True,
    )

    assert roots == [repo.resolve()]

    ignored_root = (workspace / "datasets").resolve()
    assert ignored_root not in visited
    assert not any(str(path).startswith(str(ignored_root) + os.sep) for path in visited)


def test_detect_repo_roots_ignores_external_symlink_dirs_during_state_propagation(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".gitignore").write_text("ignored/\n", encoding="utf-8")

    external_repo = tmp_path / "external-repo"
    _init_repo(external_repo)

    symlink_path = repo / "linked-external"
    try:
        symlink_path.symlink_to(external_repo, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation required for test: {exc}")

    from chunkhound.utils import ignore_engine as ignore_engine_module

    roots = ignore_engine_module.detect_repo_roots(
        repo,
        prune_ignored_gitfile_roots=True,
    )

    assert roots == [repo.resolve()]
