from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.services.realtime_indexing_service import SimpleEventHandler


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _git_init_and_commit(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init"],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "CI")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")


@pytest.mark.skipif(
    subprocess.run(["which", "git"], stdout=subprocess.DEVNULL).returncode != 0,
    reason="git required",
)
def test_realtime_nested_subrepo_boundary_respected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sub = root / "subrepo"

    # Parent repo ignoring folder name 'subrepo/'
    (root / ".gitignore").parent.mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text("subrepo/\n", encoding="utf-8")
    _git_init_and_commit(root)

    # Nested subrepo with a file
    (sub / "pkg").mkdir(parents=True, exist_ok=True)
    (sub / "pkg" / "mod.py").write_text("print('ok')\n", encoding="utf-8")
    _git_init_and_commit(sub)

    cfg = Config(
        **{
            "database": {"provider": "duckdb", "path": str(tmp_path / "db.duckdb")},
            "indexing": {
                "include": ["**/*.py"],
                "exclude": [],
                "exclude_sentinel": ".gitignore",
            },
            "target_dir": root,
        }
    )

    handler = SimpleEventHandler(event_queue=None, config=cfg, loop=None)

    # Path inside nested repo should be included despite parent ignore of folder name
    p = sub / "pkg" / "mod.py"
    assert handler._should_index(p) is True


@pytest.mark.skipif(
    subprocess.run(["which", "git"], stdout=subprocess.DEVNULL).returncode != 0,
    reason="git required",
)
def test_realtime_nonrepo_workspace_gitignore_overlay(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    # Workspace-level .gitignore (not a repo) excludes datasets/
    (ws / ".gitignore").parent.mkdir(parents=True, exist_ok=True)
    (ws / ".gitignore").write_text("datasets/\n", encoding="utf-8")

    # Non-repo datasets file
    (ws / "datasets").mkdir(parents=True, exist_ok=True)
    nonrepo_file = ws / "datasets" / "data.json"
    nonrepo_file.write_text("{}\n", encoding="utf-8")

    # Repo subtree with a python file
    repo = ws / "repo"
    (repo / "src").mkdir(parents=True, exist_ok=True)
    tracked = repo / "src" / "a.py"
    tracked.write_text("print('ok')\n", encoding="utf-8")
    _git_init_and_commit(repo)

    # Build config with overlay flag on; engine currently reads env for this behavior
    cfg = Config(
        **{
            "database": {"provider": "duckdb", "path": str(tmp_path / "db.duckdb")},
            "indexing": {
                "include": ["**/*.py", "**/*.json"],
                "exclude": [],
                "exclude_sentinel": ".gitignore",
                "workspace_gitignore_nonrepo": True,
            },
            "target_dir": ws,
        }
    )

    handler = SimpleEventHandler(event_queue=None, config=cfg, loop=None)

    # Non-repo datasets file should be excluded; repo file should be included
    assert handler._should_index(nonrepo_file) is False
    assert handler._should_index(tracked) is True
