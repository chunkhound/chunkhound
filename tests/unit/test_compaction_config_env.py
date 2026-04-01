"""Tests for compaction environment variable configuration."""

import pytest

from chunkhound.core.config.database_config import DatabaseConfig


class TestCompactionEnvConfig:
    @pytest.mark.parametrize("value", ["true", "1", "yes"])
    def test_compaction_enabled_truthy(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_ENABLED", value
        )
        config = DatabaseConfig.load_from_env()
        assert config["compaction_enabled"] is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "anything"])
    def test_compaction_enabled_falsy(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_ENABLED", value
        )
        config = DatabaseConfig.load_from_env()
        assert config["compaction_enabled"] is False

    def test_compaction_threshold_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_THRESHOLD", "0.75"
        )
        config = DatabaseConfig.load_from_env()
        assert config["compaction_threshold"] == 0.75

    def test_compaction_threshold_invalid_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_THRESHOLD", "not-a-number"
        )
        config = DatabaseConfig.load_from_env()
        assert "compaction_threshold" not in config

    def test_compaction_min_size_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_MIN_SIZE_MB", "50"
        )
        config = DatabaseConfig.load_from_env()
        assert config["compaction_min_size_mb"] == 50

    def test_compaction_min_size_invalid_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_MIN_SIZE_MB", "abc"
        )
        config = DatabaseConfig.load_from_env()
        assert "compaction_min_size_mb" not in config
