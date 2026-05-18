from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.services.indexing_coordinator import IndexingCoordinator


class _FakeDb:
    db_path = None


class _IndexingConfig:
    def __init__(self, sources: list[str]) -> None:
        self.discovery_backend = "auto"
        self._sources = sources

    def resolve_ignore_sources(self) -> list[str]:
        return list(self._sources)

    def get_effective_config_excludes(self) -> list[str]:
        return []


@pytest.mark.asyncio
async def test_auto_backend_prunes_gitignored_repo_roots_when_gitignore_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "main.py"
    file_path.write_text("print('ok')\n", encoding="utf-8")

    coordinator = IndexingCoordinator(
        database_provider=_FakeDb(),
        base_directory=tmp_path,
    )

    calls: list[bool] = []

    def _fake_repo_roots(
        directory: Path,
        config_exclude: list[str],
        prune_ignored_gitfile_roots: bool = False,
    ) -> list[Path]:
        del config_exclude
        calls.append(prune_ignored_gitfile_roots)
        return [Path(directory).resolve()]

    monkeypatch.setattr(coordinator, "_get_or_detect_repo_roots", _fake_repo_roots)
    monkeypatch.setattr(
        coordinator,
        "_discover_files_via_git",
        lambda directory, patterns, exclude_patterns, fallback_to_python: [file_path],
    )

    files = await coordinator._discover_files(
        tmp_path,
        patterns=["**/*.py"],
        exclude_patterns=[],
        parallel_discovery=False,
    )

    assert files == [file_path]
    assert calls == [True]
    assert getattr(coordinator, "_resolved_discovery_backend", None) == "git_only"


@pytest.mark.asyncio
async def test_auto_backend_skips_gitignore_prune_without_gitignore_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "main.py"
    file_path.write_text("print('ok')\n", encoding="utf-8")

    coordinator = IndexingCoordinator(
        database_provider=_FakeDb(),
        base_directory=tmp_path,
        config=SimpleNamespace(indexing=_IndexingConfig(["config"])),
    )

    calls: list[bool] = []

    def _fake_repo_roots(
        directory: Path,
        config_exclude: list[str],
        prune_ignored_gitfile_roots: bool = False,
    ) -> list[Path]:
        del config_exclude
        calls.append(prune_ignored_gitfile_roots)
        return [Path(directory).resolve()]

    monkeypatch.setattr(coordinator, "_get_or_detect_repo_roots", _fake_repo_roots)
    monkeypatch.setattr(
        coordinator,
        "_discover_files_via_git",
        lambda directory, patterns, exclude_patterns, fallback_to_python: [file_path],
    )

    files = await coordinator._discover_files(
        tmp_path,
        patterns=["**/*.py"],
        exclude_patterns=[],
        parallel_discovery=False,
    )

    assert files == [file_path]
    assert calls == [False, False]
    assert getattr(coordinator, "_resolved_discovery_backend", None) == "git_only"
