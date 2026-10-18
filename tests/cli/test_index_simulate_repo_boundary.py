from __future__ import annotations

from pathlib import Path

from tests.helpers.cli import run_cli


def test_parent_gitignore_does_not_leak_into_child_repo(tmp_path: Path) -> None:
    repo = tmp_path

    # Parent repo with rule to ignore all .md files
    (repo / ".git").mkdir()
    (repo / ".gitignore").write_text("*.txt\n")

    # Child repo under subrepo, with its own .git and an override
    (repo / "subrepo").mkdir()
    (repo / "subrepo" / ".git").mkdir()
    # Explicit negation in child (also covers non-boundary fallback)
    (repo / "subrepo" / ".gitignore").write_text("!keep.txt\n")
    (repo / "subrepo" / "keep.txt").write_text("x")

    # Enable gitignore-only via sentinel to exercise .gitignore logic explicitly
    (repo / ".chunkhound.json").write_text('{"indexing": {"exclude": ".gitignore"}}\n')

    proc = run_cli(["index", "--simulate", str(repo)], cwd=repo)
    assert proc.returncode == 0, proc.stderr
    out = set(proc.stdout.strip().splitlines())

    # Parent's *.md must NOT exclude files inside the child repo
    assert "subrepo/keep.txt" in out
