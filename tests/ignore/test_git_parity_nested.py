"""Nested .gitignore parity tests.

We validate that IgnoreEngine matches Git's behavior for .gitignore files in
subdirectories, including negation and anchored patterns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.ignore._git_oracle import engine_ignored, git_ignored, run_git, write_file

KEEP_LOG = "sub/keep.log"

# Known libgit2 disagreement: root '*.log' is not re-included by sub/.gitignore
# '!keep.log' in the libgit2 backend. Only this path is isolated; every other
# path stays strictly checked.
LIBGIT2_KNOWN_MISMATCH = (
    "libgit2 re-ignores sub/keep.log: root '*.log' is not "
    "re-included by sub/.gitignore '!keep.log' (git says not ignored)"
)


def _build_nested_repo(tmp_path: Path) -> tuple[Path, list[str]]:
    repo = tmp_path
    assert run_git(["init"], repo).returncode == 0

    # Root .gitignore: ignore logs everywhere and a root-only file
    write_file(repo, ".gitignore", "*.log\n/only-at-root.txt\n")

    # Subdirectory .gitignore: ignore 'build/' inside sub, but re-include keep.log
    write_file(repo, "sub/.gitignore", "build/\n!keep.log\n")

    write_file(repo, "a.log")
    write_file(repo, "only-at-root.txt")
    write_file(repo, "sub/b.log")
    write_file(repo, KEEP_LOG)
    write_file(repo, "sub/build/tool.txt")
    write_file(repo, "other/build/tool.txt")

    rels = [
        "a.log",  # ignored by root *.log
        "only-at-root.txt",  # ignored by root anchored path
        "sub/b.log",  # ignored by root *.log (still ignored)
        KEEP_LOG,  # re-included by sub/.gitignore
        "sub/build/tool.txt",  # ignored by sub/.gitignore build/
        "other/build/tool.txt",  # not ignored (no sub/.gitignore in 'other')
    ]
    return repo, rels


@pytest.mark.asyncio
async def test_git_parity_nested_negation_and_anchoring(
    tmp_path: Path, backend: str
) -> None:
    repo, rels = _build_nested_repo(tmp_path)
    if backend == "libgit2":
        # The known mismatch is isolated in test_libgit2_nested_keep_log_known_gap.
        rels = [r for r in rels if r != KEEP_LOG]

    gmap = git_ignored(rels, repo)
    emap = engine_ignored(rels, repo, backend)

    assert emap == gmap


@pytest.mark.asyncio
async def test_libgit2_nested_keep_log_known_gap(
    tmp_path: Path, backend: str, request: pytest.FixtureRequest
) -> None:
    if backend != "libgit2":
        pytest.skip("known gap is libgit2-only")
    request.applymarker(
        pytest.mark.xfail(
            strict=True, raises=AssertionError, reason=LIBGIT2_KNOWN_MISMATCH
        )
    )
    repo, _ = _build_nested_repo(tmp_path)

    assert engine_ignored([KEEP_LOG], repo, backend) == git_ignored([KEEP_LOG], repo)
