from __future__ import annotations

from pathlib import Path

from chunkhound.utils.file_patterns import walk_directory_tree


def test_walk_directory_tree_with_repo_boundaries(tmp_path: Path) -> None:
    root = tmp_path
    (root / ".git").mkdir()
    (root / ".gitignore").write_text("*.txt\n")
    (root / "subrepo").mkdir()
    (root / "subrepo" / ".git").mkdir()
    (root / "subrepo" / "keep.txt").write_text("x")

    from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine

    eng = build_repo_aware_ignore_engine(
        root,
        sources=["gitignore"],
        chignore_file=".chignore",
        config_exclude=["**/.git/**"],
    )

    files, _ = walk_directory_tree(
        root,
        root,
        patterns=["**/*.txt"],
        exclude_patterns=[],
        parent_gitignores={},
        use_inode_ordering=False,
        ignore_engine=eng,
    )
    rels = sorted([p.resolve().relative_to(root).as_posix() for p in files])
    assert "subrepo/keep.txt" in rels
