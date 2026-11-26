"""Batch discovery and realtime admission must agree on the same tree.

For each shared layout, every file is classified by:
- batch: IndexingCoordinator._discover_files (walk / git discovery path)
- realtime: RealtimePathFilter.should_index (per-event path)

Every layout here is a git repo, so batch resolves to ``git_only`` and
enumerates with ``git ls-files`` (indexing_coordinator.py, _discover_files_via_git).
That path does not build a gitignore engine, so the gitignore backend cannot
change batch output. Therefore:
- batch is checked once per layout and must be identical for both backends
  (test_batch_is_backend_independent_in_git_layouts);
- the backend is varied on the realtime side only, and realtime is compared
  against that single batch result (test_batch_and_realtime_agree_on_every_file).

Layout-specific gaps are strict xfails that name the concrete disagreement;
they are the same gaps the ignore matrix records
(see tests/ignore/test_ignore_matrix_new_cases.py).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
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


# (layout name, builder, gap). gap is None or (backend, reason, paths): the named
# backend disagrees with batch on exactly ``paths``. Those paths are isolated into
# their own strict xfail; every other file is compared without any xfail.
LAYOUTS = [
    (
        "nested_gitignore",
        _layout_nested_gitignore,
        (
            "libgit2",
            "libgit2 realtime drops src/keep.log (re-included by "
            "'!keep.log'); batch admits it",
            {"src/keep.log"},
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
            {"sec/a.py"},
        ),
    ),
    ("chignore", _layout_chignore, None),
]


def _all_visible_files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )


def _scoped_config(root: Path, tmp_path: Path, backend: str) -> Config:
    # Same include scope on both sides so only ignore semantics can differ.
    config = _build_config(root, tmp_path / "db.duckdb")
    config.indexing.include = ["**/*"]
    config.indexing.index_unknown_files = True
    config.indexing.gitignore_backend = backend
    return config


async def _batch_rels(root: Path, tmp_path: Path, backend: str) -> set[str]:
    config = _scoped_config(root, tmp_path, backend)
    provider = DuckDBProvider(tmp_path / "db.duckdb", base_directory=root)
    provider.connect()
    try:
        coordinator = IndexingCoordinator(
            database_provider=provider, base_directory=root, config=config
        )
        batch = await coordinator._discover_files(root, ["**/*"], [])
        rels = {p.relative_to(root).as_posix() for p in batch}
    finally:
        provider.disconnect()
    return {r for r in rels if not any(part.startswith(".") for part in Path(r).parts)}


def _realtime_rels(root: Path, tmp_path: Path, backend: str) -> set[str]:
    config = _scoped_config(root, tmp_path, backend)
    realtime = RealtimePathFilter(config=config, root_path=root)
    return {r for r in _all_visible_files(root) if realtime.should_index(root / r)}


@pytest.mark.parametrize(
    "name,builder,gap",
    LAYOUTS,
    ids=[layout[0] for layout in LAYOUTS],
)
@pytest.mark.asyncio
async def test_batch_is_backend_independent_in_git_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    builder,
    gap: tuple[str, str, set[str]] | None,
) -> None:
    """Batch output is the same for python and libgit2 in git repos.

    This is the code's current behavior (git ls-files path), not a git-vs-
    ChunkHound check. If batch ever starts reading the gitignore backend, this
    fails and the realtime comparison below must be revisited.
    """
    root_py = tmp_path / "py" / "repo"
    root_lg = tmp_path / "lg" / "repo"
    for root in (root_py, root_lg):
        root.mkdir(parents=True)
        builder(root, monkeypatch)
    batch_py = await _batch_rels(root_py, tmp_path / "py", "python")
    batch_lg = await _batch_rels(root_lg, tmp_path / "lg", "libgit2")
    assert batch_py == batch_lg, {
        "only_python": sorted(batch_py - batch_lg),
        "only_libgit2": sorted(batch_lg - batch_py),
    }


@pytest.mark.parametrize(
    "name,builder,gap",
    LAYOUTS,
    ids=[layout[0] for layout in LAYOUTS],
)
@pytest.mark.asyncio
async def test_batch_and_realtime_agree_on_every_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    name: str,
    builder,
    gap: tuple[str, str, set[str]] | None,
) -> None:
    """Realtime (this backend) must admit exactly the files batch admits.

    Known-gap paths are excluded here so any other disagreement still fails.
    They are checked by test_known_batch_realtime_gap_paths.

    Batch is computed with the python backend: in git layouts batch is
    backend-independent (see test_batch_is_backend_independent_in_git_layouts),
    so the backend under test varies the realtime side only.
    """
    root = tmp_path / "repo"
    root.mkdir()
    builder(root, monkeypatch)
    known = gap[2] if gap is not None and gap[0] == backend else set()

    batch_rels = await _batch_rels(root, tmp_path, "python") - known
    rt_rels = _realtime_rels(root, tmp_path, backend) - known
    assert batch_rels == rt_rels, {
        "only_batch": sorted(batch_rels - rt_rels),
        "only_realtime": sorted(rt_rels - batch_rels),
    }


@pytest.mark.parametrize(
    "name,builder,gap",
    [layout for layout in LAYOUTS if layout[2] is not None],
    ids=[layout[0] for layout in LAYOUTS if layout[2] is not None],
)
@pytest.mark.asyncio
async def test_known_batch_realtime_gap_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    backend: str,
    name: str,
    builder,
    gap: tuple[str, str, set[str]],
) -> None:
    """Only the documented disagreeing paths are xfail; the gap is strict."""
    gap_backend, reason, known = gap
    if backend != gap_backend:
        pytest.skip(f"known gap is {gap_backend}-only")
    request.applymarker(
        pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)
    )
    root = tmp_path / "repo"
    root.mkdir()
    builder(root, monkeypatch)

    batch_rels = await _batch_rels(root, tmp_path, "python")
    rt_rels = _realtime_rels(root, tmp_path, backend)
    assert {p: p in batch_rels for p in known} == {p: p in rt_rels for p in known}


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
    batch_rels = await _batch_rels(root, tmp_path, backend)
    assert batch_rels == {"pub/a.py"}
    expect_rt_sec = backend == "python"
    config = _scoped_config(root, tmp_path, backend)
    realtime = RealtimePathFilter(config=config, root_path=root)
    assert realtime.should_index(root / "sec" / "a.py") is expect_rt_sec
    assert realtime.should_index(root / "pub" / "a.py") is True
