"""Contract tests for the DuckDB HNSW enable/disable configuration."""

import argparse
import json
from types import SimpleNamespace

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig


class TestDatabaseConfigDuckdbHnsw:
    """DatabaseConfig field, env, and CLI contracts for duckdb_hnsw_enabled."""

    def test_default_enabled(self) -> None:
        config = DatabaseConfig()
        assert config.duckdb_hnsw_enabled is True

    def test_json_construction(self) -> None:
        config = DatabaseConfig(duckdb_hnsw_enabled=False)
        assert config.duckdb_hnsw_enabled is False
        assert DatabaseConfig(duckdb_hnsw_enabled=True).duckdb_hnsw_enabled is True

    def test_load_from_env_true_variants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for raw in ("true", "1", "yes", "on", " TRUE "):
            monkeypatch.setenv("CHUNKHOUND_DATABASE__DUCKDB_HNSW_ENABLED", raw)
            config = DatabaseConfig.load_from_env()
            assert config["duckdb_hnsw_enabled"] is True, raw

    def test_load_from_env_false_variants(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for raw in ("false", "0", "no", "off"):
            monkeypatch.setenv("CHUNKHOUND_DATABASE__DUCKDB_HNSW_ENABLED", raw)
            config = DatabaseConfig.load_from_env()
            assert config["duckdb_hnsw_enabled"] is False, raw

    def test_load_from_env_invalid_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHUNKHOUND_DATABASE__DUCKDB_HNSW_ENABLED", "banana")
        with pytest.raises(ValueError, match="boolean"):
            DatabaseConfig.load_from_env()

    def test_load_from_env_unset_omits_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CHUNKHOUND_DATABASE__DUCKDB_HNSW_ENABLED", raising=False)
        config = DatabaseConfig.load_from_env()
        assert "duckdb_hnsw_enabled" not in config

    def test_extract_cli_overrides_preserves_false(self) -> None:
        args = SimpleNamespace(duckdb_hnsw_enabled=False)
        overrides = DatabaseConfig.extract_cli_overrides(args)
        assert overrides == {"duckdb_hnsw_enabled": False}

    def test_extract_cli_overrides_true(self) -> None:
        args = SimpleNamespace(duckdb_hnsw_enabled=True)
        overrides = DatabaseConfig.extract_cli_overrides(args)
        assert overrides == {"duckdb_hnsw_enabled": True}

    def test_extract_cli_overrides_none_skipped(self) -> None:
        args = SimpleNamespace(duckdb_hnsw_enabled=None)
        overrides = DatabaseConfig.extract_cli_overrides(args)
        assert overrides == {}

    def test_cli_flag_directions(self) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        DatabaseConfig.add_cli_arguments(parser)

        assert parser.parse_args([]).duckdb_hnsw_enabled is None
        assert parser.parse_args(["--duckdb-hnsw"]).duckdb_hnsw_enabled is True
        assert parser.parse_args(["--no-duckdb-hnsw"]).duckdb_hnsw_enabled is False

    def test_repr_includes_flag(self) -> None:
        assert "duckdb_hnsw_enabled=True" in repr(DatabaseConfig())
        assert "duckdb_hnsw_enabled=False" in repr(
            DatabaseConfig(duckdb_hnsw_enabled=False)
        )


class TestConfigFileDuckdbHnsw:
    """Config precedence for database.duckdb_hnsw_enabled."""

    def test_json_config_file_disables(self, tmp_path) -> None:
        config_path = tmp_path / ".chunkhound.json"
        config_path.write_text(
            json.dumps({"database": {"duckdb_hnsw_enabled": False}}),
            encoding="utf-8",
        )

        config = Config(target_dir=tmp_path)
        assert config.database.duckdb_hnsw_enabled is False

    def test_env_reaches_config(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch, clean_environment
    ) -> None:
        monkeypatch.setenv("CHUNKHOUND_DATABASE__DUCKDB_HNSW_ENABLED", "false")
        config = Config(target_dir=tmp_path)
        assert config.database.duckdb_hnsw_enabled is False

    def test_invalid_env_fails_explicitly(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch, clean_environment
    ) -> None:
        monkeypatch.setenv("CHUNKHOUND_DATABASE__DUCKDB_HNSW_ENABLED", "nope")
        with pytest.raises(ValueError, match="boolean"):
            Config(target_dir=tmp_path)

    def test_cli_overrides_local_json(self, tmp_path) -> None:
        config_path = tmp_path / ".chunkhound.json"
        config_path.write_text(
            json.dumps({"database": {"duckdb_hnsw_enabled": False}}),
            encoding="utf-8",
        )
        parser = argparse.ArgumentParser(add_help=False)
        DatabaseConfig.add_cli_arguments(parser)
        args = parser.parse_args(["--duckdb-hnsw"])

        config = Config(args, target_dir=tmp_path)
        assert config.database.duckdb_hnsw_enabled is True
