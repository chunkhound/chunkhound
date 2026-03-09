"""
Tests for HNSW persistence configuration.

By default, hnsw_enable_experimental_persistence must NOT be set — this
causes multi-GB DB bloat on network shares (observed: 1.6 GB -> 9.9 GB).
DuckDB falls back to exact brute-force cosine similarity, giving 100%
recall, adequate speed for <200k vectors.

The opt-in flag allows users on local SSD to re-enable HNSW.

Implementation notes (verified from source):
- DuckDBConnectionManager.__init__ sets: self._db_path, self.connection=None,
  self.config=config
- _load_extensions() accesses only self.connection — safe to use __new__
  and set just those two attrs.
"""
from unittest.mock import MagicMock

from chunkhound.core.config.database_config import DatabaseConfig


class TestDatabaseConfigDefaults:
    def test_enable_hnsw_persistence_defaults_to_false(self):
        """HNSW persistence must be opt-in, not opt-out."""
        config = DatabaseConfig()
        assert config.enable_hnsw_persistence is False

    def test_enable_hnsw_persistence_can_be_opted_in(self):
        config = DatabaseConfig(enable_hnsw_persistence=True)
        assert config.enable_hnsw_persistence is True


class TestConnectionManagerHNSWGate:
    """_load_extensions must skip the SET when config.enable_hnsw_persistence is False."""

    def _make_connection_manager(self, enable_hnsw: bool):
        """Minimal DuckDBConnectionManager with mocked connection."""
        from chunkhound.providers.database.duckdb.connection_manager import (
            DuckDBConnectionManager,
        )

        config = DatabaseConfig(enable_hnsw_persistence=enable_hnsw)
        mgr = DuckDBConnectionManager.__new__(DuckDBConnectionManager)
        mgr.config = config            # verified: self.config (not self._config)
        mgr.connection = MagicMock()   # only attr _load_extensions accesses
        return mgr

    def test_hnsw_set_not_called_when_disabled(self):
        """When enable_hnsw_persistence=False, the SET statement must NOT execute."""
        mgr = self._make_connection_manager(enable_hnsw=False)
        mgr._load_extensions()

        executed_sql = [c.args[0] for c in mgr.connection.execute.call_args_list]
        assert not any(
            "hnsw_enable_experimental_persistence" in sql for sql in executed_sql
        ), f"HNSW persistence was set despite being disabled. Calls: {executed_sql}"

    def test_hnsw_set_called_when_enabled(self):
        """When enable_hnsw_persistence=True, the SET statement MUST execute."""
        mgr = self._make_connection_manager(enable_hnsw=True)
        mgr._load_extensions()

        executed_sql = [c.args[0] for c in mgr.connection.execute.call_args_list]
        assert any(
            "hnsw_enable_experimental_persistence" in sql for sql in executed_sql
        ), "HNSW persistence was not set even though it was explicitly enabled."

    def test_vss_extension_always_loaded_regardless_of_hnsw_flag(self):
        """VSS must be installed and loaded whether HNSW is on or off."""
        for flag in (True, False):
            mgr = self._make_connection_manager(enable_hnsw=flag)
            mgr._load_extensions()
            executed_sql = [c.args[0] for c in mgr.connection.execute.call_args_list]
            assert any("INSTALL vss" in sql for sql in executed_sql), (
                f"VSS not installed when enable_hnsw_persistence={flag}"
            )
            assert any("LOAD vss" in sql for sql in executed_sql), (
                f"VSS not loaded when enable_hnsw_persistence={flag}"
            )
