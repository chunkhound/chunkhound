"""Contract test: `_quickresearch` ignores user include/exclude filters.

The `_quickresearch` CLI subcommand is invoked only by `chunkhound websearch`
and the websearch MCP tool to index a tempdir of fetched pages. User-supplied
`indexing.include` / `indexing.exclude` (from `.chunkhound.json`, `--config`,
or `CHUNKHOUND_INDEXING__*` env vars) would silently zero out those files.

This test pins the externally-observable invariant: by the time
`DirectoryIndexingService` is constructed inside `quickresearch_command`, the
five filter-related `IndexingConfig` fields are reset to library defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from chunkhound.api.cli.commands import quickresearch as qr_mod
from chunkhound.core.config.config import Config
from chunkhound.core.config.indexing_config import IndexingConfig


@pytest.mark.asyncio
async def test_quickresearch_ignores_user_include_exclude_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Build a config carrying every filter knob the indexing pipeline reads.
    config = Config()
    config.indexing.include = ["**/*.py"]
    config.indexing.exclude = ["**/foo/**"]
    config.indexing.exclude_sentinel = ".gitignore"
    config.indexing.exclude_mode = "config_only"
    config.indexing.exclude_user_supplied = True

    captured: dict[str, IndexingConfig] = {}

    def fake_setup(formatter: Any, cfg: Any) -> tuple[Any, Any]:
        return (object(), object())

    def fake_create_services(db: Any, cfg: Any, embedding_manager: Any) -> Any:
        return argparse.Namespace(indexing_coordinator=object())

    class FakeIndexingService:
        def __init__(
            self,
            *,
            indexing_coordinator: Any,
            config: Any,
            progress_callback: Any,
        ) -> None:
            # Snapshot the indexing config at construction time — this is the
            # moment the pipeline starts consulting include/exclude.
            captured["indexing"] = config.indexing

        async def process_directory(self, path: Any) -> None:
            return None

    async def fake_run_research(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(qr_mod, "setup_embedding_llm", fake_setup)
    monkeypatch.setattr(qr_mod, "create_services", fake_create_services)
    monkeypatch.setattr(qr_mod, "DirectoryIndexingService", FakeIndexingService)
    monkeypatch.setattr(qr_mod, "run_research", fake_run_research)

    args = argparse.Namespace(
        query="q",
        path=tmp_path,
        path_filter=None,
        verbose=False,
    )

    await qr_mod.quickresearch_command(args, config)

    snapshot = captured["indexing"]
    defaults = IndexingConfig()
    assert snapshot.include == defaults.include
    assert snapshot.exclude == defaults.exclude
    assert snapshot.exclude_sentinel is None
    assert snapshot.exclude_mode is None
    assert snapshot.exclude_user_supplied is False
