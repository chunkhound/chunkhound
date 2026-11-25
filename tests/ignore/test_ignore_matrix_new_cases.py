"""Ignore-semantics cases that were missing from the parity suite.

Each case runs on every ``backend`` (see conftest.py). Expectations come from
``git check-ignore --no-index`` unless the mode's documented behavior differs
(docs: wiki "Rust Ignore-Matcher Migration — Design", Backend modes). Known
disagreements are strict xfails that name the concrete difference, so a fix
flips them to XPASS and forces the marker to be removed.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from tests.ignore._git_oracle import (
    assert_engine_backend,
    build_backend_engine,
    engine_ignored,
    git_ignored,
    run_git,
    write_file,
)


def _xfail(request: pytest.FixtureRequest, reason: str) -> None:
    request.applymarker(
        pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert run_git(["init"], repo).returncode == 0
    return repo


def _assert_matches_git(repo: Path, rels: list[str], backend: str) -> None:
    assert engine_ignored(rels, repo, backend) == git_ignored(rels, repo)


# --- character classes, '?', negation, '**', anchoring -----------------------


def test_character_class_and_question_mark(tmp_path: Path, backend: str) -> None:
    repo = _init_repo(tmp_path)
    write_file(repo, ".gitignore", "[ab].log\n[!c]x\nq?.txt\n")
    rels = ["a.log", "b.log", "c.log", "ax", "cx", "q1.txt", "qq.txt", "q.txt"]
    for rel in rels:
        write_file(repo, rel)
    _assert_matches_git(repo, rels, backend)


def test_negation_and_doublestar(tmp_path: Path, backend: str) -> None:
    repo = _init_repo(tmp_path)
    write_file(
        repo, ".gitignore", "*.log\n!keep.log\nbuild/*\n!build/keep\na/**/z\n**/y/**\n"
    )
    rels = [
        "a.log",
        "keep.log",
        "build/x",
        "build/keep",
        "a/z",
        "a/b/z",
        "a/b/c/z",
        "q/y/x",
    ]
    for rel in rels:
        write_file(repo, rel)
    _assert_matches_git(repo, rels, backend)


def test_anchoring_root_vs_nested(tmp_path: Path, backend: str) -> None:
    repo = _init_repo(tmp_path)
    write_file(repo, ".gitignore", "/root.txt\nsub/x.txt\n")
    rels = ["root.txt", "d/root.txt", "sub/x.txt", "d/sub/x.txt"]
    for rel in rels:
        write_file(repo, rel)
    _assert_matches_git(repo, rels, backend)


# --- .git/info/exclude and .chignore -----------------------------------------


def test_info_exclude_is_honored(tmp_path: Path, backend: str) -> None:
    repo = _init_repo(tmp_path)
    (repo / ".git" / "info" / "exclude").write_text("sec/\n", encoding="utf-8")
    write_file(repo, "sec/a.key")
    write_file(repo, "pub/a.key")
    rels = ["sec/a.key", "pub/a.key"]
    assert git_ignored(rels, repo) == {"sec/a.key": True, "pub/a.key": False}
    if backend == "python":
        # Current code: python engine reads only .gitignore files
        # (ignore_engine.py:_read_transformed_gitignore_patterns), so
        # .git/info/exclude is NOT applied and sec/a.key is not ignored.
        assert engine_ignored(rels, repo, backend) == {
            "sec/a.key": False,
            "pub/a.key": False,
        }
    else:
        # libgit2 asks pygit2 repo ignore rules, which read info/exclude.
        _assert_matches_git(repo, rels, backend)


def test_chignore_file_is_not_applied(
    tmp_path: Path, backend: str, request: pytest.FixtureRequest
) -> None:
    """Documented split: libgit2 does NOT apply .chignore; python mode should.

    The wiki says python honors .chignore, but the 'chignore' source was
    removed from build_ignore_engine. So python currently ignores nothing from
    .chignore; that gap is an xfail naming it.
    """
    repo = _init_repo(tmp_path)
    write_file(repo, ".chignore", "*.tmp\n")
    write_file(repo, "a.tmp")
    eng = build_backend_engine(repo, backend)
    matched = eng.matches(repo / "a.tmp", is_dir=False) is not None
    if backend == "libgit2":
        assert matched is False, "libgit2 mode must not apply .chignore"
    else:
        _xfail(
            request,
            "wiki says python mode honors .chignore, but the 'chignore' source "
            "was removed; a.tmp is not ignored",
        )
        assert matched is True


# --- tracked-but-ignored ------------------------------------------------------


def test_tracked_but_ignored_file_counts_as_ignored(
    tmp_path: Path, backend: str
) -> None:
    """A file in the git index that .gitignore matches is treated as ignored.

    Documented for libgit2 (index is not consulted). Git itself with --no-index
    agrees; plain `git check-ignore` (with index) would say not ignored.
    """
    repo = _init_repo(tmp_path)
    write_file(repo, ".gitignore", "*.log\n")
    write_file(repo, "t.log")
    write_file(repo, "u.log")
    assert run_git(["add", "-f", "t.log"], repo).returncode == 0
    rels = ["t.log", "u.log"]
    assert git_ignored(rels, repo) == {"t.log": True, "u.log": True}
    _assert_matches_git(repo, rels, backend)


# --- config excludes through the repo-aware engine ---------------------------


def test_config_excludes_apply_through_repo_aware_engine(
    tmp_path: Path, backend: str
) -> None:
    repo = _init_repo(tmp_path)
    write_file(repo, "pkg/node_modules/dep/index.js")
    write_file(repo, "src/app.min.js")
    write_file(repo, "src/app.js")
    eng = build_backend_engine(
        repo,
        backend,
        sources=("gitignore", "config"),
        config_exclude=("**/node_modules/**", "*.min.js"),
    )
    assert eng.matches(repo / "pkg/node_modules/dep/index.js", is_dir=False)
    assert eng.matches(repo / "src/app.min.js", is_dir=False)
    assert eng.matches(repo / "src/app.js", is_dir=False) is None


# --- Windows-style separators -------------------------------------------------


@pytest.mark.skipif(
    os.name != "nt",
    reason="Backslash separators only exist as path separators on Windows",
)
def test_windows_separators_match_on_backend(tmp_path: Path, backend: str) -> None:
    repo = _init_repo(tmp_path)
    write_file(repo, ".gitignore", "build/\n")
    write_file(repo, "build/x.txt")
    eng = build_backend_engine(repo, backend)
    win_style = str(repo / "build" / "x.txt").replace("/", "\\")
    assert eng.matches(Path(win_style), is_dir=False) is not None


# --- submodule -----------------------------------------------------------------


def _make_submodule_repo(tmp_path: Path) -> Path:
    lib = tmp_path / "lib"
    lib.mkdir()
    assert run_git(["init"], lib).returncode == 0
    write_file(lib, "s.py")
    run_git(["add", "."], lib)
    run_git(["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"], lib)

    top = tmp_path / "top"
    top.mkdir()
    assert run_git(["init"], top).returncode == 0
    write_file(top, ".gitignore", "*.txt\n")
    add = subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "submodule",
            "add",
            "-q",
            str(lib),
            "vendor/lib",
        ],
        cwd=str(top),
        capture_output=True,
        text=True,
        check=False,
    )
    assert add.returncode == 0, add.stderr
    return top


def test_submodule_contents_follow_nested_repo_boundary(
    tmp_path: Path, backend: str
) -> None:
    """Code today: a submodule's files are not ignored by the parent's rules.

    The parent has '*.txt' and vendor/lib/s.txt matches it. Python and libgit2
    treat vendor/lib as its own repo root (_detect_repo_roots), so the
    parent's rule does not reach inside it and s.txt is admitted on both.
    """
    top = _make_submodule_repo(tmp_path)
    write_file(top, "vendor/lib/s.txt")
    assert engine_ignored(["vendor/lib/s.txt"], top, backend) == {
        "vendor/lib/s.txt": False
    }


def test_submodule_contents_match_git_for_parent_rule(
    tmp_path: Path, backend: str, request: pytest.FixtureRequest
) -> None:
    """Git oracle: git applies the parent '*.txt' inside the submodule path."""
    _xfail(
        request,
        "git ignores vendor/lib/s.txt via the parent '*.txt' rule; python and "
        "libgit2 treat vendor/lib as a nested repo root "
        "(ignore_engine.py, _detect_repo_roots / _nearest_repo) and admit it",
    )
    top = _make_submodule_repo(tmp_path)
    write_file(top, "vendor/lib/s.txt")
    _assert_matches_git(top, ["vendor/lib/s.txt"], backend)


def test_submodule_directory_is_not_ignored_by_parent(
    tmp_path: Path, backend: str, request: pytest.FixtureRequest
) -> None:
    if backend == "libgit2":
        _xfail(
            request,
            "libgit2 reports the submodule directory vendor/lib as ignored; "
            "git and python do not",
        )
    top = _make_submodule_repo(tmp_path)
    assert engine_ignored(["vendor/lib"], top, backend) == git_ignored(
        ["vendor/lib"], top
    )


# --- user exclude patterns through batch discovery ----------------------------


async def _batch_files(
    tmp_path: Path, repo: Path, backend: str
) -> tuple[list[str], IndexingCoordinator]:
    from chunkhound.core.types.common import Language
    from chunkhound.parsers.parser_factory import create_parser_for_language
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider

    write_file(repo, "pkg/vendored/v.py")
    write_file(repo, "pkg/own.py")
    config = Config(
        target_dir=repo,
        database={"path": str(tmp_path / "d.duckdb"), "provider": "duckdb"},
        indexing={"include": ["**/*.py"], "gitignore_backend": backend},
    )
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()
    try:
        coordinator = IndexingCoordinator(
            db,
            repo,
            None,
            {Language.PYTHON: create_parser_for_language(Language.PYTHON)},
            None,
            config,
        )
        files = await coordinator._discover_files(
            repo,
            patterns=["**/*.py"],
            exclude_patterns=["**/vendored/**"],
            parallel_discovery=False,
        )
    finally:
        db.disconnect()
    return sorted(p.relative_to(repo).as_posix() for p in files), coordinator


@pytest.mark.asyncio
async def test_user_exclude_subtree_pattern_walk_uses_requested_backend(
    tmp_path: Path, backend: str
) -> None:
    """Plain-directory walk applies the user exclude and builds the asked backend."""
    repo = tmp_path / "plain"
    repo.mkdir()
    rels, coordinator = await _batch_files(tmp_path, repo, backend)
    (walk_engine,) = coordinator._ignore_engine_cache.values()
    assert_engine_backend(walk_engine, backend)
    assert rels == ["pkg/own.py"]


@pytest.mark.asyncio
async def test_git_repo_batch_enumerates_with_git_ls_files(
    tmp_path: Path, backend: str
) -> None:
    """Inside a git repo, batch uses `git ls-files`; no ignore engine is built.

    So the gitignore backend cannot change batch output here. This is the
    reason batch results are backend-independent in git layouts.
    """
    repo = _init_repo(tmp_path)
    _rels, coordinator = await _batch_files(tmp_path, repo, backend)
    assert coordinator._resolved_discovery_backend == "git_only"
    assert not coordinator._ignore_engine_cache


@pytest.mark.asyncio
async def test_git_repo_batch_drops_user_exclude_subtree(
    tmp_path: Path, backend: str, request: pytest.FixtureRequest
) -> None:
    """Known gap: git discovery ignores the user exclude '**/vendored/**'.

    Inside a git repo, pkg/vendored/v.py is kept, while the non-repo walk drops
    it. The git_only path applies config excludes only through the `git`
    results and does not evaluate the exclude_patterns argument.
    """
    _xfail(
        request,
        "git discovery backend drops user exclude '**/vendored/**': "
        "pkg/vendored/v.py is kept inside a git repo; the non-repo walk "
        "honors it",
    )
    rels, _ = await _batch_files(tmp_path, _init_repo(tmp_path), backend)
    assert rels == ["pkg/own.py"]
