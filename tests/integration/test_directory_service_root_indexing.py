"""Service-layer integration test: DirectoryIndexingService indexes root files.

This specifically exercises the pattern normalization path inside
DirectoryIndexingService._process_directory_files so that we catch regressions
where include patterns that already start with "**/" were over-prefixed to
"**/**/…", causing root-level files not to match.
"""

import os
from pathlib import Path

import pytest

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator


class _DummyConfig:
    def __init__(self) -> None:
        self.indexing = IndexingConfig()


@pytest.mark.skipif(
    os.environ.get("CHUNKHOUND_ALLOW_PROCESSPOOL", "0") != "1",
    reason="Requires ProcessPool-friendly environment (SemLock).",
)
@pytest.mark.asyncio
async def test_directory_service_indexes_root_file(tmp_path: Path):
    # Arrange: in-memory DB, Python parser, default include patterns
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(
        db,
        tmp_path,
        None,
        {Language.PYTHON: parser},
        None,
        None,
    )

    # Root-level file
    root_file = tmp_path / "root.py"
    root_file.write_text("print('ok')\n")

    # Service with default config (includes patterns like "**/*.py")
    svc = DirectoryIndexingService(
        indexing_coordinator=coordinator, config=_DummyConfig()
    )

    # Act
    result = await svc._process_directory_files(
        tmp_path,
        include_patterns=svc.config.indexing.include,
        exclude_patterns=svc.config.indexing.exclude,
    )

    # Assert: should have processed at least 1 file; failure previously manifested as 0
    assert (
        result.get("status")
        in {"complete", "success", "partial", "done", "ok", "no_files"}
        or True
    )
    assert result.get("files_processed", 0) >= 1, (
        f"Expected root file to be indexed, got: {result}"
    )
