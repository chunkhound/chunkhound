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


class _CaptureCoordinator:
    def __init__(self) -> None:
        self.last_patterns: list[str] | None = None

    async def process_directory(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        config_file_size_threshold_kb: int = 20,
    ) -> dict:
        self.last_patterns = list(patterns or [])
        # Return a benign payload so the service doesn't raise
        return {"status": "no_files", "patterns": self.last_patterns}


class _DummyConfig:
    def __init__(self) -> None:
        self.indexing = IndexingConfig()


@pytest.mark.asyncio
async def test_patterns_not_double_prefixed(tmp_path: Path):
    coord = _CaptureCoordinator()
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
