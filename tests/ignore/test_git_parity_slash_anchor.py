from __future__ import annotations

from pathlib import Path

from tests.ignore._git_oracle import git_ignored, run_git


def test_root_gitignore_slash_anchor(tmp_path: Path) -> None:
    repo = tmp_path
    assert run_git(["init"], repo).returncode == 0

    # Root .gitignore with pattern containing '/': anchored to root
    (repo / ".gitignore").write_text("pdf-chunker/work/\n")

    # Create paths
    (repo / "pdf-chunker" / "work").mkdir(parents=True, exist_ok=True)
    (repo / "pdf-chunker" / "work" / "x.txt").write_text("x")
    (repo / "services" / "pdf-chunker" / "work").mkdir(parents=True, exist_ok=True)
    (repo / "services" / "pdf-chunker" / "work" / "y.txt").write_text("y")

    rels = [
        "pdf-chunker/work/x.txt",  # should be ignored by Git
        "services/pdf-chunker/work/y.txt",  # should NOT be ignored by Git
    ]

    gmap = git_ignored(rels, repo)

    # Engine decision
    from chunkhound.utils.ignore_engine import build_ignore_engine  # type: ignore

    eng = build_ignore_engine(
        root=repo, sources=["gitignore"], chignore_file=".chignore", config_exclude=None
    )

    emap: dict[str, bool] = {}
    for p in rels:
        emap[p] = eng.matches(repo / p, is_dir=False) is not None

    assert emap == gmap, (emap, gmap)
