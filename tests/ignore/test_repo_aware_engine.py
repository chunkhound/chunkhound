from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_repo_aware_engine_parent_child_rules(tmp_path: Path) -> None:
    root = tmp_path
    # Parent repo ignores *.txt
    (root / ".git").mkdir()
    (root / ".gitignore").write_text("*.txt\n")

    # Child repo with its own .git, no .gitignore
    (root / "child").mkdir()
    (root / "child" / ".git").mkdir()
    (root / "child" / "k.txt").write_text("x")

    from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine

    eng = build_repo_aware_ignore_engine(
        root,
        sources=["gitignore"],
        chignore_file=".chignore",
        config_exclude=["**/.git/**"],
    )

    # Parent match (ignored)
    assert eng.matches(root / "a.txt", is_dir=False) is not None
    # Inside child repo: should NOT be ignored by parent
    assert eng.matches(root / "child" / "k.txt", is_dir=False) is None
