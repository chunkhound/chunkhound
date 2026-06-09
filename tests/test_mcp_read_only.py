"""Contract tests for the --read-only MCP flag.

Covers three user-visible invariants:
  (a) An existing DB opens read-only and serves regex search.
  (b) Writes through the provider's public API are rejected.
  (c) Config.validate_for_command rejects read_only=True for non-mcp commands.
"""

import asyncio

import duckdb
import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator


@pytest.fixture
def populated_db_path(tmp_path):
    """Build a DuckDB DB on disk, index one Python file, disconnect cleanly."""
    db_path = tmp_path / "chunks.db"
    writer = DuckDBProvider(db_path, base_directory=tmp_path)
    writer.connect()
    try:
        parser = create_parser_for_language(Language.PYTHON)
        coordinator = IndexingCoordinator(
            writer, tmp_path, None, {Language.PYTHON: parser}
        )
        sample = tmp_path / "sample.py"
        sample.write_text(
            "def calculate_tax(income, rate):\n    return income * rate\n"
        )
        asyncio.run(coordinator.process_file(sample))
    finally:
        writer.disconnect()
    return db_path


@pytest.fixture
def read_only_reader(populated_db_path, tmp_path):
    """Open populated_db_path read-only through the production wiring."""
    cfg = DatabaseConfig(read_only=True)
    reader = DuckDBProvider(populated_db_path, base_directory=tmp_path, config=cfg)
    reader.connect()
    try:
        yield reader
    finally:
        reader.disconnect()


def test_read_only_open_runs_regex_search(read_only_reader):
    results, _ = read_only_reader.search_regex("calculate_tax")
    assert any("calculate_tax" in r["content"] for r in results)


def test_read_only_rejects_writes(read_only_reader):
    with pytest.raises(duckdb.Error):
        read_only_reader.execute_query("CREATE TABLE __probe (id INTEGER)")
    with pytest.raises(duckdb.Error):
        read_only_reader.execute_query("INSERT INTO files (path) VALUES ('x')")


def test_validator_rejects_read_only_for_index(tmp_path):
    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / "chunks.db", "read_only": True},
    )
    errors = config.validate_for_command("index")
    assert any(
        "database.read_only=True is only valid for the 'mcp' subcommand" in e
        for e in errors
    ), errors
    assert not any(
        "read_only" in e for e in config.validate_for_command("mcp")
    )
