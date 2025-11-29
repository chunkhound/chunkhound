"""Unit tests for database path consistency across providers.

This test module ensures that DatabaseConfig.get_db_path() returns the correct
final database location for each provider, preventing path duplication bugs.
"""

from pathlib import Path

import pytest

from chunkhound.core.config.database_config import DatabaseConfig


def test_get_db_path_duckdb_no_suffix(tmp_path):
    """Verify DuckDB path is just chunks.db file."""
    test_db_path = tmp_path / "test_db"
    config = DatabaseConfig(path=test_db_path, provider="duckdb")
    actual_path = config.get_db_path()

    assert actual_path == test_db_path / "chunks.db"
    assert actual_path.name == "chunks.db"
    assert ".lancedb" not in actual_path.name


def test_get_db_path_lancedb_includes_suffix(tmp_path):
    """Verify LanceDB path includes .lancedb suffix."""
    test_db_path = tmp_path / "test_db"
    config = DatabaseConfig(path=test_db_path, provider="lancedb")
    actual_path = config.get_db_path()

    assert actual_path == test_db_path / "lancedb.lancedb"
    assert ".lancedb" in actual_path.name
    assert actual_path.name == "lancedb.lancedb"


def test_get_db_path_lancedb_suffix_not_duplicated(tmp_path):
    """Verify .lancedb suffix is not duplicated if path already has it."""
    test_db_path = tmp_path / "test_db"
    config = DatabaseConfig(path=test_db_path, provider="lancedb")
    actual_path = config.get_db_path()

    # Count occurrences of .lancedb in the path
    path_str = str(actual_path)
    count = path_str.count(".lancedb")
    assert count == 1, f"Expected exactly 1 .lancedb suffix, found {count}"


def test_get_db_path_creates_parent_directory(tmp_path):
    """Verify get_db_path creates parent directory if it doesn't exist."""
    test_db_path = tmp_path / "test_db"
    # Parent directory doesn't exist yet
    assert not test_db_path.exists()

    config = DatabaseConfig(path=test_db_path, provider="duckdb")
    db_path = config.get_db_path()

    # Parent directory should now exist
    assert test_db_path.exists()
    assert test_db_path.is_dir()
    # But the database file itself shouldn't exist yet
    assert not db_path.exists()


def test_get_db_path_none_raises_error():
    """Verify error when path is not configured."""
    config = DatabaseConfig(path=None, provider="duckdb")

    with pytest.raises(ValueError, match="Database path not configured"):
        config.get_db_path()


def test_path_property_differs_from_get_db_path(tmp_path):
    """Document that .path != get_db_path() - this is intentional.

    This test documents the critical contract: config.database.path is the
    base directory, while get_db_path() returns the actual database location.
    CLI commands MUST use get_db_path() for existence checks.
    """
    test_db_path = tmp_path / "test_db"
    config = DatabaseConfig(path=test_db_path, provider="duckdb")

    # Base directory path
    base_path = config.path
    # Actual database location
    db_path = config.get_db_path()

    # These should be different
    assert base_path != db_path
    assert db_path == base_path / "chunks.db"


def test_lancedb_path_transformation_matches_provider(tmp_path):
    """Verify DatabaseConfig path transformation matches what LanceDBProvider expects.

    This test ensures that the .lancedb suffix added by DatabaseConfig.get_db_path()
    matches what LanceDBProvider previously did, preventing any path mismatches.
    """
    test_db_path = tmp_path / "test_db"
    config = DatabaseConfig(path=test_db_path, provider="lancedb")
    db_path = config.get_db_path()

    # Simulate what LanceDBProvider.__init__ used to do
    input_path = test_db_path / "lancedb"
    legacy_transform = input_path.parent / f"{input_path.stem}.lancedb"

    # DatabaseConfig should return the same result
    assert db_path == legacy_transform
    assert db_path == test_db_path / "lancedb.lancedb"
