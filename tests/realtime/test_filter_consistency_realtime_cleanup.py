from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.models import File
from chunkhound.core.types.common import Language
from chunkhound.database_factory import create_services
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.realtime_indexing_service import (
    RealtimeIndexingService,
    SimpleEventHandler,
    WatchmanRealtimeAdapter,
)
from chunkhound.services.realtime_path_filter import RealtimePathFilter
from chunkhound.watchman import WatchmanSubscriptionScope


def _git_init(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )


def _build_gitignored_worktree_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    _git_init(root)
    (root / ".gitignore").write_text(".gitignored/\n", encoding="utf-8")

    ignored_file = root / ".gitignored" / "worktrees" / "feature" / "tracked.py"
    ignored_file.parent.mkdir(parents=True, exist_ok=True)
    ignored_file.write_text("def tracked():\n    return 1\n", encoding="utf-8")

    included_file = root / "src" / "included.py"
    included_file.parent.mkdir(parents=True, exist_ok=True)
    included_file.write_text("def included():\n    return 1\n", encoding="utf-8")

    return root, ignored_file, included_file


def _build_config(
    root: Path,
    db_path: Path,
    *,
    realtime_backend: str = "watchdog",
) -> Config:
    return Config(
        target_dir=root,
        database={"path": str(db_path), "provider": "duckdb"},
        indexing={
            "include": ["**/*.py"],
            "exclude": [],
            "exclude_sentinel": ".gitignore",
            "realtime_backend": realtime_backend,
        },
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
@pytest.mark.asyncio
async def test_watchdog_event_filter_blocks_gitignored_worktree_file(
    tmp_path: Path,
) -> None:
    root, ignored_file, included_file = _build_gitignored_worktree_layout(tmp_path)
    config = _build_config(root, tmp_path / "watchdog.duckdb")
    event_queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue()
    filtered_events: list[tuple[str, Path]] = []

    handler = SimpleEventHandler(
        event_queue,
        config=config,
        loop=asyncio.get_running_loop(),
        root_path=root,
        filtered_event_callback=lambda event_type, file_path: filtered_events.append(
            (event_type, file_path)
        ),
    )

    handler.on_any_event(
        SimpleNamespace(
            event_type="created",
            is_directory=False,
            src_path=str(ignored_file),
        )
    )
    handler.on_any_event(
        SimpleNamespace(
            event_type="created",
            is_directory=False,
            src_path=str(included_file),
        )
    )

    await asyncio.sleep(0)

    assert filtered_events == [("created", ignored_file.resolve())]
    assert await event_queue.get() == ("created", included_file.resolve())
    assert event_queue.empty()


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
@pytest.mark.asyncio
async def test_watchman_translation_filters_gitignored_worktree_file(
    tmp_path: Path,
) -> None:
    root, ignored_file, _ = _build_gitignored_worktree_layout(tmp_path)
    db_path = tmp_path / "watchman.duckdb"
    config = _build_config(root, db_path, realtime_backend="watchman")
    services = create_services(db_path, config)
    services.provider.connect()

    try:
        service = RealtimeIndexingService(services, config)
        adapter = WatchmanRealtimeAdapter(service)
        adapter._path_filter = RealtimePathFilter(config=config, root_path=root)

        adapter._translate_subscription_pdu(
            {
                "subscription": "chunkhound-live-indexing",
                "clock": "c:0:1",
                "files": [
                    {
                        "name": ".gitignored/worktrees/feature/tracked.py",
                        "exists": True,
                        "new": True,
                        "type": "f",
                    }
                ],
            },
            WatchmanSubscriptionScope(
                requested_path=root,
                watch_root=root.resolve(),
                relative_root=None,
                scope_kind="primary",
            ),
        )

        stats = await service.get_health()
        assert service.event_queue.empty()
        assert stats["pipeline"]["filtered_event_count"] == 1
        assert stats["pipeline"]["last_source_event_path"] == str(ignored_file)
        assert stats["pipeline"]["last_accepted_event_path"] is None
    finally:
        services.provider.disconnect()


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
@pytest.mark.asyncio
async def test_dir_index_expansion_does_not_enqueue_gitignored_worktree_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, ignored_file, _ = _build_gitignored_worktree_layout(tmp_path)
    db_path = tmp_path / "dir-index.duckdb"
    config = _build_config(root, db_path)
    services = create_services(db_path, config)
    services.provider.connect()

    try:
        service = RealtimeIndexingService(services, config)
        service.watch_path = root
        queued_children: list[tuple[Path, str]] = []

        async def fake_add_file(file_path: Path, priority: str = "change") -> bool:
            queued_children.append((file_path, priority))
            return True

        monkeypatch.setattr(service, "add_file", fake_add_file)

        await service._index_directory(ignored_file.parent)

        assert queued_children == []
    finally:
        services.provider.disconnect()


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
@pytest.mark.asyncio
async def test_discovery_realtime_and_cleanup_agree_on_gitignored_worktree_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, ignored_file, included_file = _build_gitignored_worktree_layout(tmp_path)
    missing_file = root / "src" / "missing.py"
    db_path = tmp_path / "cleanup.duckdb"
    config = _build_config(root, db_path)
    provider = DuckDBProvider(db_path, base_directory=root)
    provider.connect()

    try:
        coordinator = IndexingCoordinator(
            database_provider=provider,
            base_directory=root,
            config=config,
        )
        realtime_filter = RealtimePathFilter(config=config, root_path=root)

        discovered = await coordinator._discover_files(root, ["**/*.py"], [])
        discovered_paths = {path.relative_to(root).as_posix() for path in discovered}

        ignored_rel = ignored_file.relative_to(root).as_posix()
        included_rel = included_file.relative_to(root).as_posix()
        missing_rel = missing_file.relative_to(root).as_posix()

        assert ignored_rel not in discovered_paths
        assert included_rel in discovered_paths
        assert realtime_filter.should_index(ignored_file) is False
        assert realtime_filter.should_index(included_file) is True
        assert (
            coordinator._classify_cleanup_candidate(
                ignored_rel,
                discovered_paths,
                realtime_filter,
            )
            == "excluded_by_current_policy"
        )
        assert (
            coordinator._classify_cleanup_candidate(
                missing_rel,
                discovered_paths,
                realtime_filter,
            )
            == "missing_on_disk"
        )

        provider.insert_file(
            File(
                path=ignored_rel,
                mtime=float(ignored_file.stat().st_mtime),
                language=Language.PYTHON,
                size_bytes=int(ignored_file.stat().st_size),
            )
        )
        provider.insert_file(
            File(
                path=missing_rel,
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=0,
            )
        )

        deleted_paths: list[str] = []

        def fake_delete_files_batch(file_paths: list[str]) -> int:
            deleted_paths.extend(file_paths)
            return len(file_paths)

        monkeypatch.setattr(
            provider,
            "delete_files_batch",
            fake_delete_files_batch,
        )

        cleaned = coordinator._cleanup_orphaned_files(root, discovered, ["**/*.py"], [])

        assert cleaned == 2
        assert deleted_paths == [missing_rel, ignored_rel]
    finally:
        provider.disconnect()
