"""core.excludesFile end to end: real git config, real files, real discovery.

tests/unit/test_global_gitignore.py mocks the git-config lookup. These tests
point GIT_CONFIG_GLOBAL and HOME at a real gitconfig and check what ChunkHound
does with it today.

Two kinds of test live here:
- Code-contract tests (passing): assert what the code does per backend and per
  side (batch, realtime). They are the expectation.
- Git-oracle xfails (strict): assert what git does. They fail today where the
  code disagrees with git, and name the exact disagreement and location.

Where the global file is read: ``_collect_global_gitignore_patterns``
(chunkhound/utils/ignore_engine.py) flattens it to '**/' patterns. Batch uses
them through ``_extend_with_global_gitignore`` (indexing_coordinator.py).
pygit2 reads ``$HOME/.gitconfig``, so the fixture writes the same file to both
locations.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.realtime_path_filter import (
    RealtimePathFilter,
    RealtimePathFilterSettings,
)
from tests.ignore import _backends
from tests.ignore._git_oracle import git_ignored, run_git

# Shared ``backend`` fixture (python, libgit2 with pygit2 skip).
backend = _backends.backend

GLOBAL_PATTERNS = "*.glob\nscratch/\n"
ALL_FILES = ["keep.py", "x.glob", "scratch/a.py"]


@pytest.fixture
def global_excludes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    excludes = home / "global-ignore"
    excludes.write_text(GLOBAL_PATTERNS, encoding="utf-8")
    gitconfig = home / ".gitconfig"
    gitconfig.write_text(f"[core]\n\texcludesFile = {excludes}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("HOME", str(home))
    return excludes


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def _files(root: Path) -> None:
    for rel in ALL_FILES:
        _write(root / rel)


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert run_git(["init"], repo).returncode == 0
    _files(repo)
    return repo


def _plain_dir(tmp_path: Path) -> Path:
    plain = tmp_path / "plain"
    plain.mkdir()
    _files(plain)
    return plain


def _config(root: Path, tmp_path: Path, backend: str) -> Config:
    return Config(
        target_dir=root,
        database={"path": str(tmp_path / "d.duckdb"), "provider": "duckdb"},
        indexing={
            "include": ["**/*"],
            "index_unknown_files": True,
            "gitignore_backend": backend,
        },
    )


async def _batch_rels(root: Path, tmp_path: Path, backend: str) -> list[str]:
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()
    try:
        coordinator = IndexingCoordinator(
            db,
            root,
            None,
            {Language.PYTHON: create_parser_for_language(Language.PYTHON)},
            None,
            _config(root, tmp_path, backend),
        )
        files = await coordinator._discover_files(
            root, patterns=["**/*"], exclude_patterns=[], parallel_discovery=False
        )
    finally:
        db.disconnect()
    return sorted(p.relative_to(root).as_posix() for p in files)


def _realtime_admitted_in_fresh_interpreter(root: Path) -> set[str]:
    """Run the libgit2 realtime check in a new process (clean pygit2 config)."""
    script = (
        "import json,sys\n"
        "from pathlib import Path\n"
        "from tests.ignore.test_global_excludes_end_to_end import "
        "_realtime_admitted\n"
        "print(json.dumps(sorted(_realtime_admitted(Path(sys.argv[1]), 'libgit2'))))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ.copy(),
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


def _realtime_admitted(root: Path, backend: str) -> set[str]:
    flt = RealtimePathFilter(
        config=None,
        root_path=root,
        settings=RealtimePathFilterSettings(
            include_patterns=("**/*",),
            ignore_sources=("gitignore",),
            gitignore_backend=backend,
            index_unknown_files=True,
        ),
    )
    return {rel for rel in ALL_FILES if flt.should_index(root / rel)}


# --- oracle precondition (git itself) -----------------------------------------


def test_fixture_gives_git_the_global_file(
    tmp_path: Path, global_excludes: Path
) -> None:
    """Precondition: git honors the fixture's core.excludesFile."""
    repo = _git_repo(tmp_path)
    assert git_ignored(ALL_FILES, repo) == {
        "keep.py": False,
        "x.glob": True,
        "scratch/a.py": True,
    }


# --- batch, git repo (git ls-files path) --------------------------------------


@pytest.mark.asyncio
async def test_batch_in_git_repo_applies_glob_but_not_dir_pattern(
    tmp_path: Path, global_excludes: Path, backend: str
) -> None:
    """Code today: in a git repo batch drops '*.glob' but keeps scratch/a.py.

    Batch uses git ls-files (git_only), so the gitignore backend does not reach
    it. The global 'scratch/' is flattened to '**/scratch/', which does not
    match files under scratch/ (file_patterns.should_exclude_path).
    """
    repo = _git_repo(tmp_path)
    assert await _batch_rels(repo, tmp_path, backend) == ["keep.py", "scratch/a.py"]


@pytest.mark.asyncio
async def test_batch_in_git_repo_drops_scratch_dir_like_git(
    tmp_path: Path, global_excludes: Path, backend: str, request: pytest.FixtureRequest
) -> None:
    request.applymarker(
        pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason=(
                "git ignores scratch/a.py via core.excludesFile 'scratch/', but "
                "batch keeps it: _extend_with_global_gitignore "
                "(indexing_coordinator.py) flattens it to '**/scratch/', which "
                "file_patterns.should_exclude_path does not match for files "
                "under the directory"
            ),
        )
    )
    repo = _git_repo(tmp_path)
    assert await _batch_rels(repo, tmp_path, backend) == ["keep.py"]


# --- batch, plain directory (no git repo) -------------------------------------


@pytest.mark.asyncio
async def test_batch_in_plain_dir_does_not_apply_core_excludes(
    tmp_path: Path, global_excludes: Path, backend: str
) -> None:
    """Code today: a non-repo directory gets no core.excludesFile filtering.

    There is no git repo, so git has no opinion here and this is not a git-oracle
    xfail. The global patterns are added to a local list in
    _extend_with_global_gitignore, but the walker receives exclude_patterns, and
    build_repo_aware_ignore_engine only reads .gitignore and config excludes for
    a non-repo root. So all three files are kept on both backends.
    """
    plain = _plain_dir(tmp_path)
    assert await _batch_rels(plain, tmp_path, backend) == [
        "keep.py",
        "scratch/a.py",
        "x.glob",
    ]


# --- realtime -----------------------------------------------------------------


def test_realtime_admission_of_core_excludes_files_per_backend(
    tmp_path: Path, global_excludes: Path, backend: str
) -> None:
    """Code today: realtime honors core.excludesFile only on the libgit2 backend.

    python: the engine (build_repo_aware_ignore_engine) reads .gitignore and
    config excludes only, so all files are admitted.
    libgit2: pygit2 reads $HOME/.gitconfig (the fixture sets HOME), so the
    git-ignored files are dropped, matching git.

    libgit2 is checked in a fresh interpreter. pygit2 caches the global config
    per process, so an earlier libgit2 test (e.g. test_git_parity_slash_anchor)
    would otherwise leave a stale HOME in effect and change this result.
    """
    repo = _git_repo(tmp_path)
    if backend == "libgit2":
        admitted = _realtime_admitted_in_fresh_interpreter(repo)
    else:
        admitted = _realtime_admitted(repo, backend)
    expected = {"keep.py"} if backend == "libgit2" else set(ALL_FILES)
    assert admitted == expected


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "git ignores x.glob and scratch/a.py via core.excludesFile, but python "
        "realtime admits both: the python engine (build_repo_aware_ignore_engine, "
        "realtime_path_filter.py) has no global-excludes source"
    ),
)
def test_realtime_python_drops_core_excludes_files_like_git(
    tmp_path: Path, global_excludes: Path
) -> None:
    repo = _git_repo(tmp_path)
    assert _realtime_admitted(repo, "python") == {"keep.py"}
