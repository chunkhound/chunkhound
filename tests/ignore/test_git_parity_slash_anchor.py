from __future__ import annotations

import subprocess
from pathlib import Path


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.safecrlf=false",
            *args,
        ],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _git_ignored(paths: list[str], repo: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for p in paths:
        proc = _run_git(["check-ignore", "-q", "--no-index", p], repo)
        out[p] = proc.returncode == 0
    return out


def test_root_gitignore_slash_anchor(tmp_path: Path) -> None:
    repo = tmp_path
    assert _run_git(["init"], repo).returncode == 0

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

    gmap = _git_ignored(rels, repo)

    # Engine decision
    from chunkhound.utils.ignore_engine import build_ignore_engine  # type: ignore

    eng = build_ignore_engine(
        root=repo, sources=["gitignore"], chignore_file=".chignore", config_exclude=None
    )

    emap: dict[str, bool] = {}
    for p in rels:
        emap[p] = eng.matches(repo / p, is_dir=False) is not None

    assert emap == gmap, (emap, gmap)
