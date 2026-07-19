"""Contract tests for database execute timeout configuration."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.providers.database.serial_executor import (
    _COMPACTION_OPERATION_TIMEOUT_SECONDS,
    SerialDatabaseExecutor,
)


class TestDatabaseConfigExecuteTimeout:
    """DatabaseConfig field, env, and CLI contracts for execute_timeout_seconds."""

    def test_default_is_none(self) -> None:
        config = DatabaseConfig()
        assert config.execute_timeout_seconds is None

    def test_custom_value(self) -> None:
        config = DatabaseConfig(execute_timeout_seconds=120.0)
        assert config.execute_timeout_seconds == 120.0

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError):
            DatabaseConfig(execute_timeout_seconds=0)
        with pytest.raises(ValueError):
            DatabaseConfig(execute_timeout_seconds=-1.0)

    def test_load_from_env_canonical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHUNKHOUND_DATABASE__EXECUTE_TIMEOUT_SECONDS", "90.5")
        monkeypatch.delenv("CHUNKHOUND_DB_EXECUTE_TIMEOUT", raising=False)

        config = DatabaseConfig.load_from_env()
        assert config["execute_timeout_seconds"] == 90.5

    def test_load_from_env_legacy_alias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(
            "CHUNKHOUND_DATABASE__EXECUTE_TIMEOUT_SECONDS", raising=False
        )
        monkeypatch.setenv("CHUNKHOUND_DB_EXECUTE_TIMEOUT", "45")

        config = DatabaseConfig.load_from_env()
        assert config["execute_timeout_seconds"] == 45.0

    def test_load_from_env_canonical_wins_over_legacy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHUNKHOUND_DATABASE__EXECUTE_TIMEOUT_SECONDS", "100")
        monkeypatch.setenv("CHUNKHOUND_DB_EXECUTE_TIMEOUT", "50")

        config = DatabaseConfig.load_from_env()
        assert config["execute_timeout_seconds"] == 100.0

    def test_load_from_env_invalid_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__EXECUTE_TIMEOUT_SECONDS", "not-a-number"
        )

        config = DatabaseConfig.load_from_env()
        assert "execute_timeout_seconds" not in config

    def test_extract_cli_overrides(self) -> None:
        args = SimpleNamespace(
            db=None,
            database_path=None,
            database_provider=None,
            max_disk_usage_gb=None,
            read_only=False,
            fragmentation_threshold_pct=None,
            db_execute_timeout=180.0,
        )

        overrides = DatabaseConfig.extract_cli_overrides(args)
        assert overrides == {"execute_timeout_seconds": 180.0}

    def test_repr_includes_timeout(self) -> None:
        config = DatabaseConfig(execute_timeout_seconds=60.0)
        assert "execute_timeout_seconds=60.0" in repr(config)


class TestConfigFileExecuteTimeout:
    """Config precedence for database.execute_timeout_seconds."""

    def test_json_config_file_sets_timeout(self, tmp_path) -> None:
        config_path = tmp_path / ".chunkhound.json"
        config_path.write_text(
            json.dumps({"database": {"execute_timeout_seconds": 120}}),
            encoding="utf-8",
        )

        config = Config(target_dir=tmp_path)
        assert config.database.execute_timeout_seconds == 120.0

    def test_config_from_env_only_canonical(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch, clean_environment
    ) -> None:
        """Env-only deployments (e.g. MCP) must surface the canonical var."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__EXECUTE_TIMEOUT_SECONDS", "75")
        # Empty project dir: no local JSON, no global discovery interference
        config = Config(target_dir=tmp_path)
        assert config.database.execute_timeout_seconds == 75.0

    def test_config_from_env_only_legacy(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch, clean_environment
    ) -> None:
        """Legacy CHUNKHOUND_DB_EXECUTE_TIMEOUT still reaches Config."""
        monkeypatch.setenv("CHUNKHOUND_DB_EXECUTE_TIMEOUT", "88")
        config = Config(target_dir=tmp_path)
        assert config.database.execute_timeout_seconds == 88.0

    def test_local_json_overrides_env(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHUNKHOUND_DB_EXECUTE_TIMEOUT", "30")
        config_path = tmp_path / ".chunkhound.json"
        config_path.write_text(
            json.dumps({"database": {"execute_timeout_seconds": 200}}),
            encoding="utf-8",
        )

        config = Config(target_dir=tmp_path)
        assert config.database.execute_timeout_seconds == 200.0

    def test_cli_overrides_local_json(self, tmp_path) -> None:
        config_path = tmp_path / ".chunkhound.json"
        config_path.write_text(
            json.dumps({"database": {"execute_timeout_seconds": 50}}),
            encoding="utf-8",
        )
        args = SimpleNamespace(
            command=None,
            config=None,
            path=None,
            debug=False,
            verbose=False,
            db=None,
            database_path=None,
            database_provider=None,
            max_disk_usage_gb=None,
            read_only=False,
            fragmentation_threshold_pct=None,
            db_execute_timeout=300.0,
        )

        config = Config(args, target_dir=tmp_path)
        assert config.database.execute_timeout_seconds == 300.0


class TestSerialExecutorTimeoutResolution:
    """SerialDatabaseExecutor timeout resolution contract."""

    def test_defaults_when_unset(self) -> None:
        executor = SerialDatabaseExecutor()
        assert executor.resolve_timeout("search_semantic") == 30.0
        assert (
            executor.resolve_timeout("compact_database")
            == _COMPACTION_OPERATION_TIMEOUT_SECONDS
        )
        assert (
            executor.resolve_timeout("compact_if_needed")
            == _COMPACTION_OPERATION_TIMEOUT_SECONDS
        )

    def test_explicit_timeout_applies_to_all_ops(self) -> None:
        executor = SerialDatabaseExecutor(execute_timeout_seconds=120.0)
        assert executor.resolve_timeout("search_semantic") == 120.0
        assert executor.resolve_timeout("compact_database") == 120.0
        assert executor.resolve_timeout("ensure_all_hnsw_indexes") == 120.0

    def test_hnsw_rebuild_uses_normal_default(self) -> None:
        executor = SerialDatabaseExecutor()
        assert executor.resolve_timeout("ensure_all_hnsw_indexes") == 30.0
        assert executor.resolve_timeout("drop_all_hnsw_indexes") == 30.0

    def test_execute_sync_passes_timeout_to_future_result(self) -> None:
        """User-visible timeout must reach future.result(timeout=...)."""
        executor = SerialDatabaseExecutor(execute_timeout_seconds=42.0)

        provider = MagicMock()
        provider.get_base_directory.return_value = None
        # Bypass connection/thread-local path; only assert timeout wiring
        captured: list[float] = []

        mock_future = MagicMock()
        mock_future.result.side_effect = lambda timeout=None: (
            captured.append(timeout),
            "ok",
        )[1]

        # Skip real executor work and auto-compaction side effects
        executor._db_executor = MagicMock()
        executor._db_executor.submit.return_value = mock_future
        executor._maybe_run_sampled_auto_compaction = MagicMock()  # type: ignore[method-assign]

        # Provide a no-op executor method so getattr succeeds if called
        provider._executor_search_semantic = MagicMock(return_value="ok")

        result = executor.execute_sync(provider, "search_semantic")

        assert result == "ok"
        assert captured == [42.0]
        mock_future.result.assert_called_once_with(timeout=42.0)

    def test_execute_sync_default_compact_timeout_to_future_result(
        self,
    ) -> None:
        executor = SerialDatabaseExecutor()

        provider = MagicMock()
        provider.get_base_directory.return_value = None
        mock_future = MagicMock()
        mock_future.result.return_value = True

        executor._db_executor = MagicMock()
        executor._db_executor.submit.return_value = mock_future
        executor._maybe_run_sampled_auto_compaction = MagicMock()  # type: ignore[method-assign]
        provider._executor_compact_database = MagicMock(return_value=True)

        executor.execute_sync(provider, "compact_database")

        mock_future.result.assert_called_once_with(
            timeout=_COMPACTION_OPERATION_TIMEOUT_SECONDS
        )


class TestProviderWiresExecutorTimeout:
    """SerialDatabaseProvider passes config timeout into the executor."""

    def test_provider_passes_timeout_from_config(self, tmp_path) -> None:
        from chunkhound.providers.database.serial_database_provider import (
            SerialDatabaseProvider,
        )

        class _StubProvider(SerialDatabaseProvider):
            def _create_connection(self):
                return object()

            def _get_schema_sql(self):
                return None

        cfg = DatabaseConfig(
            path=tmp_path / "db",
            execute_timeout_seconds=99.0,
        )
        provider = _StubProvider(
            db_path=tmp_path / "db" / "chunks.db",
            base_directory=tmp_path,
            config=cfg,
        )
        assert provider._executor.resolve_timeout("any_op") == 99.0

    def test_provider_without_config_uses_defaults(self, tmp_path) -> None:
        from chunkhound.providers.database.serial_database_provider import (
            SerialDatabaseProvider,
        )

        class _StubProvider(SerialDatabaseProvider):
            def _create_connection(self):
                return object()

            def _get_schema_sql(self):
                return None

        provider = _StubProvider(
            db_path=tmp_path / "db" / "chunks.db",
            base_directory=tmp_path,
            config=None,
        )
        assert provider._executor.resolve_timeout("search") == 30.0
        assert (
            provider._executor.resolve_timeout("compact_database")
            == _COMPACTION_OPERATION_TIMEOUT_SECONDS
        )
