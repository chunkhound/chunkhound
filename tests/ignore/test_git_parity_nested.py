"""Nested .gitignore parity tests.

We validate that IgnoreEngine matches Git's behavior for .gitignore files in
subdirectories, including negation and anchored patterns.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

import pytest


def run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
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


def git_ignored(paths: Iterable[str], repo: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for p in paths:
        proc = run_git(["check-ignore", "-q", "--no-index", p], repo)
        out[p] = proc.returncode == 0
    return out


@pytest.mark.asyncio
async def test_git_parity_nested_negation_and_anchoring(tmp_path: Path) -> None:
    repo = tmp_path
    assert run_git(["init"], repo).returncode == 0

    # Root .gitignore: ignore logs everywhere and a root-only file
    (repo / ".gitignore").write_text("""*.log
/only-at-root.txt
""")

    # Subdirectory .gitignore: ignore 'build/' inside sub, but re-include keep.log
    (repo / "sub").mkdir(parents=True, exist_ok=True)
    (repo / "sub" / ".gitignore").write_text("""build/
!keep.log
""")

    # Create files to test
    (repo / "a.log").write_text("x")
    (repo / "only-at-root.txt").write_text("x")
    (repo / "sub" / "b.log").write_text("x")
    (repo / "sub" / "keep.log").write_text("x")
    (repo / "sub" / "build").mkdir(parents=True, exist_ok=True)
    (repo / "sub" / "build" / "tool.txt").write_text("x")
    (repo / "other" / "build").mkdir(parents=True, exist_ok=True)
    (repo / "other" / "build" / "tool.txt").write_text("x")

    rels = [
        "a.log",  # ignored by root *.log
        "only-at-root.txt",  # ignored by root anchored path
        "sub/b.log",  # ignored by root *.log (still ignored)
        "sub/keep.log",  # re-included by sub/.gitignore
        "sub/build/tool.txt",  # ignored by sub/.gitignore build/
        "other/build/tool.txt",  # not ignored (no sub/.gitignore in 'other')
    ]

    gmap = git_ignored(rels, repo)

    from chunkhound.utils.ignore_engine import build_ignore_engine  # type: ignore

    engine = build_ignore_engine(
        root=repo, sources=["gitignore"], chignore_file=".chignore", config_exclude=None
    )

    emap: dict[str, bool] = {}
    for p in rels:
        emap[p] = engine.matches(repo / p, is_dir=False) is not None

    assert emap == gmap
