from __future__ import annotations

from pathlib import Path

from tests.ignore._git_oracle import engine_ignored, git_ignored, run_git


def test_backend_matches_git_for_foo_dir_rule(tmp_path: Path, backend: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    assert run_git(["init"], repo).returncode == 0

    # Patterns: ignore foo/ but not bar/
    (repo / ".gitignore").write_text("foo/\n", encoding="utf-8")
    (repo / "foo").mkdir(parents=True, exist_ok=True)
    (repo / "foo" / "x.txt").write_text("x\n", encoding="utf-8")
    (repo / "bar").mkdir(parents=True, exist_ok=True)
    (repo / "bar" / "y.py").write_text("print('y')\n", encoding="utf-8")

    rels = ["foo/x.txt", "bar/y.py"]
    assert engine_ignored(rels, repo, backend) == git_ignored(rels, repo)
