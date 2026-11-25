"""core.excludesFile end to end: real git config, real files, real discovery.

tests/unit/test_global_gitignore.py mocks the git-config lookup. These tests
point GIT_CONFIG_GLOBAL at a real gitconfig and check user-visible results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.ignore._git_oracle import git_ignored, run_git


@pytest.fixture
def global_excludes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    excludes = home / "global-ignore"
    excludes.write_text("*.glob\nscratch/\n", encoding="utf-8")
    gitconfig = home / "gitconfig"
    gitconfig.write_text(f"[core]\n\texcludesFile = {excludes}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("HOME", str(home))
    return excludes


def _repo_with_files(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert run_git(["init"], repo).returncode == 0
    (repo / "keep.py").write_text("x", encoding="utf-8")
    (repo / "x.glob").write_text("x", encoding="utf-8")
    (repo / "scratch").mkdir()
    (repo / "scratch" / "a.py").write_text("x", encoding="utf-8")
    return repo


def test_git_honors_core_excludes_file_as_oracle(
    tmp_path: Path, global_excludes: Path
) -> None:
    repo = _repo_with_files(tmp_path)
    rels = ["x.glob", "scratch/a.py", "keep.py"]
    assert git_ignored(rels, repo) == {
        "x.glob": True,
        "scratch/a.py": True,
        "keep.py": False,
    }


@pytest.mark.asyncio
async def test_batch_discovery_applies_core_excludes_file(
    tmp_path: Path, global_excludes: Path, request: pytest.FixtureRequest
) -> None:
    """Batch discovery (IndexingCoordinator) must drop globally excluded files.

    Known gap: the global 'scratch/' (directory-only) pattern is flattened to
    '**/scratch/' and does not exclude scratch/a.py, while git ignores it.
    '*.glob' (file pattern) is excluded correctly.
    """
    request.applymarker(
        pytest.mark.xfail(
            strict=True,
            reason=(
                "batch keeps scratch/a.py: global 'scratch/' is flattened to "
                "'**/scratch/' in the flat exclude list; git ignores scratch/a.py"
            ),
        )
    )
    from chunkhound.core.types.common import Language
    from chunkhound.parsers.parser_factory import create_parser_for_language
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider
    from chunkhound.services.indexing_coordinator import IndexingCoordinator

    repo = _repo_with_files(tmp_path)
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()
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
        patterns=["**/*"],
        exclude_patterns=[],
        parallel_discovery=False,
    )
    rels = sorted(p.relative_to(repo).as_posix() for p in files)
    assert rels == ["keep.py"]


def test_realtime_filter_applies_core_excludes_file(
    tmp_path: Path, global_excludes: Path, request: pytest.FixtureRequest
) -> None:
    """Realtime admission must agree with batch: drop globally excluded files.

    Known gap: RealtimePathFilter builds its engine from .gitignore and config
    excludes only, so x.glob is admitted even though batch discovery drops it.
    """
    from chunkhound.services.realtime_path_filter import (
        RealtimePathFilter,
        RealtimePathFilterSettings,
    )

    request.applymarker(
        pytest.mark.xfail(
            strict=True,
            reason=(
                "realtime admits x.glob (core.excludesFile '*.glob') that batch "
                "discovery excludes; core.excludesFile is not in the realtime engine"
            ),
        )
    )
    repo = _repo_with_files(tmp_path)
    flt = RealtimePathFilter(
        config=None,
        root_path=repo,
        settings=RealtimePathFilterSettings(
            include_patterns=("**/*",),
            ignore_sources=("gitignore",),
            index_unknown_files=True,
        ),
    )
    assert flt.should_index(repo / "x.glob") is False
    assert flt.should_index(repo / "keep.py") is True
