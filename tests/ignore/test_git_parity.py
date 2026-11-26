"""Git-as-oracle parity tests for future IgnoreEngine implementation.

These tests construct a temporary git repository with a .gitignore file
and a small file tree, then compare decisions from `git check-ignore`
with our yet-to-be-implemented IgnoreEngine. For now, these will fail
until the engine is implemented, as part of TDD.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.ignore._git_oracle import engine_ignored, git_ignored, run_git


@pytest.mark.asyncio
async def test_git_parity_basic(tmp_path: Path, backend: str) -> None:
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

    engine_map = engine_ignored(rels, repo, backend)

    assert engine_map == git_map
