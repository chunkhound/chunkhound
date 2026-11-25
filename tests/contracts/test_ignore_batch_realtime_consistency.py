"""Batch discovery and realtime admission must agree on the same tree.

For each shared layout, every file is classified by:
- batch: IndexingCoordinator._discover_files (walk / git discovery path)
- realtime: RealtimePathFilter.should_index (per-event path)

Both sides use the same backend. Layout-specific gaps are strict xfails that
name the concrete disagreement; they are the same gaps the ignore matrix
records (see tests/ignore/test_ignore_matrix_new_cases.py).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.realtime_path_filter import RealtimePathFilter
from tests.ignore import _backends
from tests.realtime.test_filter_consistency_realtime_cleanup import (
    _build_config,
    _build_gitignored_worktree_layout,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")

# Shared ``backend`` fixture: parametrizes over IGNORE_BACKENDS and skips
# libgit2 when pygit2 is missing (tests/ignore/_backends.py).
backend = _backends.backend


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True)


def _layout_nested_gitignore(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git(root, "init")
    _write(root / ".gitignore", "*.log\nbuild/\n")
    _write(root / "src" / ".gitignore", "gen/\n!keep.log\n")
    _write(root / "src" / "a.py")
    _write(root / "src" / "gen" / "g.py")
    _write(root / "src" / "keep.log")
    _write(root / "src" / "b.log")
    _write(root / "build" / "x.py")
    _write(root / "lib" / "c.py")


def _layout_char_class(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git(root, "init")
    _write(root / ".gitignore", "[ab].py\n[!c]x.py\nq?.py\n")
    for name in ["a.py", "b.py", "c.py", "ax.py", "cx.py", "q1.py", "qq.py"]:
        _write(root / name)


def _layout_gitignored_worktree(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Shared layout from the realtime consistency suite: .gitignore excludes a
    # subtree that contains a file, and a sibling file must still be admitted.
    _build_gitignored_worktree_layout(root)


def _layout_info_exclude(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git(root, "init")
    (root / ".git" / "info" / "exclude").write_text("sec/\n", encoding="utf-8")
    _write(root / "sec" / "a.py")
    _write(root / "pub" / "a.py")


def _layout_chignore(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git(root, "init")
    _write(root / ".chignore", "*.gen.py\n")
    _write(root / "a.py")
    _write(root / "a.gen.py")


# (layout name, builder, gap). gap is None or (backend, reason): the named
# backend has a known disagreement and is an xfail; the other backend must agree.
LAYOUTS = [
    (
        "nested_gitignore",
        _layout_nested_gitignore,
        (
            "libgit2",
            "libgit2 realtime drops src/keep.log (re-included by "
            "'!keep.log'); batch admits it",
        ),
    ),
    ("char_class", _layout_char_class, None),
    ("gitignored_worktree", _layout_gitignored_worktree, None),
    (
        "info_exclude",
        _layout_info_exclude,
        (
            "python",
            "python realtime admits sec/a.py (python engine does not read "
            ".git/info/exclude; ignore_engine.py), but batch excludes it "
            "(git ls-files --exclude-standard)",
        ),
    ),
    ("chignore", _layout_chignore, None),
]


@pytest.mark.parametrize(
    "name,builder,gap",
    LAYOUTS,
    ids=[layout[0] for layout in LAYOUTS],
)
@pytest.mark.asyncio
async def test_batch_and_realtime_agree_on_every_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    backend: str,
    name: str,
    builder,
    gap: tuple[str, str] | None,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    builder(root, monkeypatch)
    if gap is not None and gap[0] == backend:
        request.applymarker(pytest.mark.xfail(strict=True, reason=gap[1]))

    # Same include scope on both sides so only ignore semantics can differ.
    config = _build_config(root, tmp_path / "db.duckdb")
    config.indexing.include = ["**/*"]
    config.indexing.index_unknown_files = True
    config.indexing.gitignore_backend = backend
    provider = DuckDBProvider(tmp_path / "db.duckdb", base_directory=root)
    provider.connect()
    try:
        coordinator = IndexingCoordinator(
            database_provider=provider,
            base_directory=root,
            config=config,
        )
        realtime = RealtimePathFilter(config=config, root_path=root)
        batch = await coordinator._discover_files(root, ["**/*"], [])
        batch_rels = {p.relative_to(root).as_posix() for p in batch}

        all_files = sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file()
            and not any(part.startswith(".") for part in p.relative_to(root).parts)
        )
        batch_rels = {
            rel
            for rel in batch_rels
            if not any(part.startswith(".") for part in Path(rel).parts)
        }
        rt_rels = {rel for rel in all_files if realtime.should_index(root / rel)}
        assert batch_rels == rt_rels, {
            "only_batch": sorted(batch_rels - rt_rels),
            "only_realtime": sorted(rt_rels - batch_rels),
        }
    finally:
        provider.disconnect()


@pytest.mark.asyncio
async def test_info_exclude_per_side_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Records what each side does today for .git/info/exclude 'sec/'.

    Batch in a git repo goes through git ls-files --exclude-standard (git_only),
    so it drops sec/a.py for both backends. Realtime python admits it because
    the python engine never reads info/exclude; libgit2 realtime drops it.
    """
    root = tmp_path / "repo"
    root.mkdir()
    _layout_info_exclude(root, monkeypatch)
    config = _build_config(root, tmp_path / "db.duckdb")
    config.indexing.include = ["**/*"]
    config.indexing.index_unknown_files = True
    config.indexing.gitignore_backend = backend
    provider = DuckDBProvider(tmp_path / "db.duckdb", base_directory=root)
    provider.connect()
    try:
        coordinator = IndexingCoordinator(
            database_provider=provider, base_directory=root, config=config
        )
        realtime = RealtimePathFilter(config=config, root_path=root)
        batch = await coordinator._discover_files(root, ["**/*"], [])
        batch_rels = {p.relative_to(root).as_posix() for p in batch}
        assert batch_rels == {"pub/a.py"}
        expect_rt_sec = backend == "python"
        assert realtime.should_index(root / "sec" / "a.py") is expect_rt_sec
        assert realtime.should_index(root / "pub" / "a.py") is True
    finally:
        provider.disconnect()
