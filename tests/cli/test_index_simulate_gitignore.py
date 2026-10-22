from __future__ import annotations

from pathlib import Path

from tests.helpers.cli import init_git_repository, run_cli


def test_simulate_respects_nested_gitignore(tmp_path: Path) -> None:
    repo = tmp_path
    init_git_repository(repo)

    (repo / ".gitignore").write_text("""/only-at-root.txt
""")

    (repo / "sub").mkdir(parents=True, exist_ok=True)
    (repo / "sub" / ".gitignore").write_text("""build/
!keep.txt
""")

    # files
    (repo / "a.txt").write_text("x")
    (repo / "only-at-root.txt").write_text("x")
    (repo / "sub" / "b.txt").write_text("x")
    (repo / "sub" / "keep.txt").write_text("x")
    (repo / "sub" / "build").mkdir(parents=True, exist_ok=True)
    (repo / "sub" / "build" / "tool.txt").write_text("x")
    (repo / "other" / "build").mkdir(parents=True, exist_ok=True)
    (repo / "other" / "build" / "tool.txt").write_text("x")

    # Force gitignore-only mode via local config
    (repo / ".chunkhound.json").write_text('{"indexing": {"exclude": ".gitignore"}}\n')

    proc = run_cli(["index", "--simulate", str(repo)], cwd=repo)
    assert proc.returncode == 0, proc.stderr
    out = set(proc.stdout.strip().splitlines())

    # Excluded
    # a.txt is included by defaults and not ignored
    assert "a.txt" in out
    assert "only-at-root.txt" not in out
    assert "sub/b.txt" in out
    assert "sub/build/tool.txt" not in out

    # Included
    assert "sub/keep.txt" in out
    # Default excludes filter generic build/ directories
    assert "other/build/tool.txt" not in out
