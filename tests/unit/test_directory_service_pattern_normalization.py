"""Unit test: DirectoryIndexingService should not over-prefix include patterns.

Before the fix, default patterns like "**/*.py" were prefixed again to
"**/**/.py", which prevented matching root-level files. This test asserts that
the service leaves patterns starting with "**/" unchanged when delegating to
the coordinator.
"""

from pathlib import Path

import pytest

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
async def test_process_directory_reports_discovery_phase_first(tmp_path: Path) -> None:
    coord = CaptureCoordinator()
    messages: list[str] = []
    svc = DirectoryIndexingService(
        indexing_coordinator=coord,
        config=_DummyConfig(),
        progress_callback=messages.append,
    )

    await svc.process_directory(tmp_path, no_embeddings=True)

    assert messages[0] == "Discovering files..."
