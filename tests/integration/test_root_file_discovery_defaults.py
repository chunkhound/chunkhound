"""Integration test for default include patterns discovering root-level files.

This test uses the real IndexingCoordinator with an in-memory DuckDB provider
and the default IndexingConfig include patterns. It verifies that a file placed
at the project root is discovered without any custom include/exclude rules.
"""


import pytest

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator


@pytest.mark.asyncio
async def test_root_file_discovered_with_default_patterns(tmp_path):
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

    # Create a root-level file
    root_file = tmp_path / "root.py"
    root_file.write_text("print('ok')\n")

    # Use default include/exclude from IndexingConfig
    cfg = IndexingConfig()
    include_patterns = list(cfg.include)
    exclude_patterns = []

    # Act: discover files
    files = await coordinator._discover_files(
        tmp_path,
        patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        parallel_discovery=False,
    )

    # Assert: root file is present
    assert root_file in files, (
        f"Root-level file not discovered. Files: {[p.name for p in files]}"
    )
