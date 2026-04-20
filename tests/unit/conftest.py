"""Shared test fixtures and helpers for the unit test suite."""

from __future__ import annotations

from pathlib import Path


class CaptureCoordinator:
    """Test double for IndexingCoordinator that records calls."""

    def __init__(self) -> None:
        self.last_force_reindex: bool | None = None
        self.last_patterns: list[str] | None = None
        self.call_count: int = 0

    async def process_directory(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        config_file_size_threshold_kb: int = 20,
        force_reindex: bool = False,
        **kwargs,
    ) -> dict:
        self.last_patterns = list(patterns or [])
        self.last_force_reindex = force_reindex
        self.call_count += 1
        return {"status": "no_files", "files_found": 0, "files_indexed": 0, "chunks_created": 0}

    async def generate_missing_embeddings(
        self, *, exclude_patterns=None, metrics_collector=None, **kwargs
    ) -> dict:
        return {"status": "up_to_date", "generated": 0}
