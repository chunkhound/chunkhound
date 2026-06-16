"""Unit test: DirectoryIndexingService should not over-prefix include patterns.

Before the fix, default patterns like "**/*.py" were prefixed again to
"**/**/.py", which prevented matching root-level files. This test asserts that
the service leaves patterns starting with "**/" unchanged when delegating to
the coordinator.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.services.directory_indexing_service import DirectoryIndexingService


class _DummyConfig:
    def __init__(self) -> None:
        self.indexing = IndexingConfig()


@pytest.mark.asyncio
async def test_patterns_not_double_prefixed(tmp_path: Path):
    coord = AsyncMock()
    async def _process_directory(
        directory,
        patterns=None,
        exclude_patterns=None,
        config_file_size_threshold_kb=20,
    ):
        del directory, exclude_patterns, config_file_size_threshold_kb
        return {"status": "no_files", "patterns": list(patterns or [])}

    coord.process_directory.side_effect = _process_directory
    svc = DirectoryIndexingService(indexing_coordinator=coord, config=_DummyConfig())

    res = await svc._process_directory_files(
        tmp_path,
        include_patterns=svc.config.indexing.include,
        exclude_patterns=svc.config.indexing.exclude,
    )

    patts = res.get("patterns", [])
    # Key assertion: none of the patterns should contain "**/**/"
    assert all("**/**/" not in p for p in patts), (
        f"Found over-prefixed pattern(s): {[p for p in patts if '**/**/' in p]}"
    )

    # And a sanity check: python pattern should remain exactly "**/*.py"
    assert "**/*.py" in patts, "Expected default python pattern '**/*.py'"


@pytest.mark.asyncio
async def test_process_directory_reports_discovery_phase_first(tmp_path: Path) -> None:
    coord = AsyncMock()
    coord._db = SimpleNamespace(
        drop_all_hnsw_indexes=lambda: None,
        ensure_all_hnsw_indexes=lambda: None,
    )
    coord.process_directory.return_value = {"status": "no_files"}
    coord.compact_database_with_metrics.return_value = {
        "status": "skipped",
        "reason": "unsupported",
    }
    messages: list[str] = []
    svc = DirectoryIndexingService(
        indexing_coordinator=coord,
        config=_DummyConfig(),
        progress_callback=messages.append,
    )

    await svc.process_directory(tmp_path, no_embeddings=True)

    assert messages[0] == "Discovering files..."
