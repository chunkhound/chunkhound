"""Unit test: DirectoryIndexingService should not over-prefix include patterns.

Before the fix, default patterns like "**/*.py" were prefixed again to
"**/**/.py", which prevented matching root-level files. This test asserts that
the service leaves patterns starting with "**/" unchanged when delegating to
the coordinator.
"""

import pytest
from pathlib import Path

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.services.directory_indexing_service import DirectoryIndexingService

from tests.unit.conftest import CaptureCoordinator


class _DummyConfig:
    def __init__(self, force_reindex: bool = False) -> None:
        self.indexing = IndexingConfig(force_reindex=force_reindex)


@pytest.mark.asyncio
async def test_patterns_not_double_prefixed(tmp_path: Path):
    coord = CaptureCoordinator()
    svc = DirectoryIndexingService(indexing_coordinator=coord, config=_DummyConfig())

    await svc.process_directory(tmp_path)

    patts = coord.last_patterns or []
    # Key assertion: none of the patterns should contain "**/**/"
    assert all("**/**/" not in p for p in patts), (
        f"Found over-prefixed pattern(s): {[p for p in patts if '**/**/' in p]}"
    )

    # And a sanity check: python pattern should remain exactly "**/*.py"
    assert "**/*.py" in patts, "Expected default python pattern '**/*.py'"


@pytest.mark.asyncio
async def test_force_reindex_forwarded_to_coordinator(tmp_path: Path):
    """force_reindex=True in config is forwarded to the coordinator."""
    coord = CaptureCoordinator()
    svc = DirectoryIndexingService(
        indexing_coordinator=coord, config=_DummyConfig(force_reindex=True)
    )

    await svc.process_directory(tmp_path)

    assert coord.last_force_reindex is True


@pytest.mark.asyncio
async def test_default_config_forwards_force_reindex_false(tmp_path: Path) -> None:
    """Default config must forward force_reindex=False to the coordinator."""
    coord = CaptureCoordinator()
    svc = DirectoryIndexingService(
        indexing_coordinator=coord,
        config=_DummyConfig(force_reindex=False),
    )

    await svc.process_directory(tmp_path)

    assert coord.call_count > 0, "coordinator was never called"
    assert coord.last_force_reindex is False
