"""Git-as-oracle parity tests for future IgnoreEngine implementation.

These tests construct a temporary git repository with a .gitignore file
and a small file tree, then compare decisions from `git check-ignore`
with our yet-to-be-implemented IgnoreEngine. For now, these will fail
until the engine is implemented, as part of TDD.
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
    """Return a mapping {path: is_ignored} using git check-ignore -q.

    We call git one path at a time to keep parsing simple and stable.
    """
    out: dict[str, bool] = {}
    for p in paths:
        proc = run_git(["check-ignore", "-q", "--no-index", p], repo)
        out[p] = proc.returncode == 0
    return out


@pytest.mark.asyncio
async def test_git_parity_basic(tmp_path: Path) -> None:
    repo = tmp_path

    # Initialize repo
    assert run_git(["init"], repo).returncode == 0

    # Write .gitignore with some common rules
    (repo / ".gitignore").write_text(
        "\n".join(
            [
                "*.log",
                "build/",
                "/root-only.txt",
                "**/*.min.js",
            ]
        )
        + "\n"
    )

    # Create files/dirs
    (repo / "a.log").write_text("x")
    (repo / "src").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "b.log").write_text("x")
    (repo / "build").mkdir(parents=True, exist_ok=True)
    (repo / "build" / "app.js").write_text("x")
    (repo / "sub").mkdir(parents=True, exist_ok=True)
    (repo / "sub" / "build").mkdir(parents=True, exist_ok=True)
    (repo / "sub" / "build" / "lib.js").write_text("x")
    (repo / "root-only.txt").write_text("x")
    (repo / "src" / "root-only.txt").write_text("x")
    (repo / "lib").mkdir(parents=True, exist_ok=True)
    (repo / "lib" / "min.min.js").write_text("x")

    rels = [
        "a.log",
        "src/b.log",
        "build/app.js",
        "sub/build/lib.js",
        "root-only.txt",
        "src/root-only.txt",
        "lib/min.min.js",
        "src/keep.js",
    ]

    git_map = git_ignored(rels, repo)

    # Import (non-existent) engine to enforce TDD failure until implemented
    from chunkhound.utils.ignore_engine import build_ignore_engine  # type: ignore

    engine = build_ignore_engine(
        root=repo,
        sources=["gitignore"],
        chignore_file=".chignore",
        config_exclude=None,
    )

    engine_map: dict[str, bool] = {}
    for p in rels:
        engine_map[p] = engine.matches((repo / p), is_dir=False) is not None

    assert engine_map == git_map
