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

from tests.ignore._git_oracle import engine_ignored, git_ignored, run_git


def _xfail(request: pytest.FixtureRequest, reason: str) -> None:
    request.applymarker(pytest.mark.xfail(strict=True, reason=reason))


def _write(repo: Path, rel: str, text: str = "x") -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    _write(repo, ".gitignore", "[ab].log\n[!c]x\nq?.txt\n")
    rels = ["a.log", "b.log", "c.log", "ax", "cx", "q1.txt", "qq.txt", "q.txt"]
    for rel in rels:
        _write(repo, rel)
    _assert_matches_git(repo, rels, backend)


def test_negation_and_doublestar(tmp_path: Path, backend: str) -> None:
    repo = _init_repo(tmp_path)
    _write(
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
        _write(repo, rel)
    _assert_matches_git(repo, rels, backend)


def test_anchoring_root_vs_nested(tmp_path: Path, backend: str) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, ".gitignore", "/root.txt\nsub/x.txt\n")
    rels = ["root.txt", "d/root.txt", "sub/x.txt", "d/sub/x.txt"]
    for rel in rels:
        _write(repo, rel)
    _assert_matches_git(repo, rels, backend)


# --- .git/info/exclude and .chignore -----------------------------------------


def test_info_exclude_is_honored(tmp_path: Path, backend: str) -> None:
    repo = _init_repo(tmp_path)
    (repo / ".git" / "info" / "exclude").write_text("sec/\n", encoding="utf-8")
    _write(repo, "sec/a.key")
    _write(repo, "pub/a.key")
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
    _write(repo, ".chignore", "*.tmp\n")
    _write(repo, "a.tmp")
    from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine

    eng = build_repo_aware_ignore_engine(
        repo,
        sources=["gitignore"],
        chignore_file=".chignore",
        config_exclude=[],
        backend=backend,
    )
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
    _write(repo, ".gitignore", "*.log\n")
    _write(repo, "t.log")
    _write(repo, "u.log")
    assert run_git(["add", "-f", "t.log"], repo).returncode == 0
    rels = ["t.log", "u.log"]
    assert git_ignored(rels, repo) == {"t.log": True, "u.log": True}
    _assert_matches_git(repo, rels, backend)


# --- config excludes through the repo-aware engine ---------------------------


def test_config_excludes_apply_through_repo_aware_engine(
    tmp_path: Path, backend: str
) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "pkg/node_modules/dep/index.js")
    _write(repo, "src/app.min.js")
    _write(repo, "src/app.js")
    from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine

    eng = build_repo_aware_ignore_engine(
        repo,
        sources=["gitignore", "config"],
        chignore_file=".chignore",
        config_exclude=["**/node_modules/**", "*.min.js"],
        backend=backend,
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
    _write(repo, ".gitignore", "build/\n")
    _write(repo, "build/x.txt")
    from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine

    eng = build_repo_aware_ignore_engine(
        repo,
        sources=["gitignore"],
        chignore_file=".chignore",
        config_exclude=[],
        backend=backend,
    )
    win_style = str(repo / "build" / "x.txt").replace("/", "\\")
    assert eng.matches(Path(win_style), is_dir=False) is not None


# --- submodule -----------------------------------------------------------------


def _make_submodule_repo(tmp_path: Path) -> Path:
    lib = tmp_path / "lib"
    lib.mkdir()
    assert run_git(["init"], lib).returncode == 0
    _write(lib, "s.py")
    run_git(["add", "."], lib)
    run_git(["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"], lib)

    top = tmp_path / "top"
    top.mkdir()
    assert run_git(["init"], top).returncode == 0
    _write(top, ".gitignore", "*.txt\n")
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


def test_submodule_contents_are_not_ignored_by_parent(
    tmp_path: Path, backend: str
) -> None:
    top = _make_submodule_repo(tmp_path)
    _assert_matches_git(top, ["vendor/lib/s.py"], backend)


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


@pytest.mark.parametrize("in_git_repo", [False, True], ids=["walk", "git_repo"])
@pytest.mark.asyncio
async def test_user_exclude_subtree_pattern_drops_files_in_batch(
    tmp_path: Path,
    backend: str,
    in_git_repo: bool,
    request: pytest.FixtureRequest,
) -> None:
    """`**/name/**` user excludes drop files under that subtree (batch path).

    Protects the behavior the fnmatch->gitignore converter helper provides.
    Known gap: inside a git repo the git discovery backend ignores the user
    exclude, so pkg/vendored/v.py is kept (the non-repo walk drops it).
    """
    from chunkhound.core.types.common import Language
    from chunkhound.parsers.parser_factory import create_parser_for_language
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider
    from chunkhound.services.indexing_coordinator import IndexingCoordinator

    if in_git_repo:
        _xfail(
            request,
            "git discovery backend drops user exclude '**/vendored/**': "
            "pkg/vendored/v.py is kept inside a git repo; the non-repo walk "
            "honors it",
        )
        repo = _init_repo(tmp_path)
    else:
        repo = tmp_path / "plain"
        repo.mkdir()
    _write(repo, "pkg/vendored/v.py")
    _write(repo, "pkg/own.py")
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()
    try:
        coordinator = IndexingCoordinator(
            db,
            repo,
            None,
            {Language.PYTHON: create_parser_for_language(Language.PYTHON)},
            None,
            None,
        )
        files = await coordinator._discover_files(
            repo,
            patterns=["**/*.py"],
            exclude_patterns=["**/vendored/**"],
            parallel_discovery=False,
        )
    finally:
        db.disconnect()
    rels = sorted(p.relative_to(repo).as_posix() for p in files)
    assert rels == ["pkg/own.py"]
