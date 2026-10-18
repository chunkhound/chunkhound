from __future__ import annotations

from pathlib import Path

from tests.helpers.cli import init_git_repository, run_cli


def test_simulate_isolates_sibling_gitignores(tmp_path: Path) -> None:
    root = tmp_path

    init_git_repository(root / "repoA")
    init_git_repository(root / "repoB")

    # repoA ignores *.txt
    (root / "repoA" / ".gitignore").write_text("*.txt\n")
    (root / "repoA" / "a.txt").write_text("x")
    (root / "repoA" / "b.md").write_text("x")

    # repoB ignores *.md
    (root / "repoB" / ".gitignore").write_text("*.md\n")
    (root / "repoB" / "a.txt").write_text("x")
    (root / "repoB" / "b.md").write_text("x")

    # Use gitignore-only mode for clarity
    (root / ".chunkhound.json").write_text('{"indexing": {"exclude": ".gitignore"}}\n')

    proc = run_cli(["index", "--simulate", str(root)], cwd=root)
    assert proc.returncode == 0, proc.stderr
    out = set(proc.stdout.strip().splitlines())

    # repoA rules apply only under repoA
    assert "repoA/a.txt" not in out
    assert "repoA/b.md" in out

    # repoB rules apply only under repoB
    assert "repoB/a.txt" in out
    assert "repoB/b.md" not in out
