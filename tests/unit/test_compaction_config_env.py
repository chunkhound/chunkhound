"""Tests for compaction environment variable configuration."""

import pytest
from loguru import logger

from chunkhound.core.config.database_config import DatabaseConfig


@pytest.fixture
def loguru_warnings():
    """Capture loguru warning messages for assertion."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda msg: messages.append(str(msg)),
        level="WARNING",
        format="{message}",
    )
    yield messages
    logger.remove(handler_id)


def test_compaction_enabled_by_default() -> None:
    cfg = DatabaseConfig()
    assert cfg.compaction_enabled is True


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

    @pytest.mark.parametrize("value", ["false", "0", "no"])
    def test_compaction_enabled_falsy(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_ENABLED", value
        )
        config = DatabaseConfig.load_from_env()
        assert config["compaction_enabled"] is False

    @pytest.mark.parametrize("value", ["on", "enabled", "anything"])
    def test_compaction_enabled_unrecognized_warns_and_defaults_true(
        self, monkeypatch: pytest.MonkeyPatch, loguru_warnings: list[str], value: str
    ) -> None:
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_ENABLED", value
        )
        config = DatabaseConfig.load_from_env()
        assert config["compaction_enabled"] is True
        assert any("Unrecognized" in msg for msg in loguru_warnings)

    def test_compaction_threshold_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_THRESHOLD", "0.75"
        )
        config = DatabaseConfig.load_from_env()
        assert config["compaction_threshold"] == 0.75

    def test_compaction_threshold_invalid_warns_and_skips(
        self, monkeypatch: pytest.MonkeyPatch, loguru_warnings: list[str]
    ) -> None:
        """Unparseable threshold logs a warning and uses the default."""
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_THRESHOLD", "not-a-number"
        )
        config = DatabaseConfig.load_from_env()
        assert "compaction_threshold" not in config
        assert any("COMPACTION_THRESHOLD" in msg for msg in loguru_warnings)

    def test_compaction_min_size_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_MIN_SIZE_MB", "50"
        )
        config = DatabaseConfig.load_from_env()
        assert config["compaction_min_size_mb"] == 50

    def test_compaction_enabled_empty_string_uses_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty string env var treated as unset (project convention)."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__COMPACTION_ENABLED", "")
        config = DatabaseConfig.load_from_env()
        assert "compaction_enabled" not in config

    def test_compaction_min_size_invalid_warns_and_skips(
        self, monkeypatch: pytest.MonkeyPatch, loguru_warnings: list[str]
    ) -> None:
        """Unparseable min-size logs a warning and uses the default."""
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_MIN_SIZE_MB", "abc"
        )
        config = DatabaseConfig.load_from_env()
        assert "compaction_min_size_mb" not in config
        assert any("COMPACTION_MIN_SIZE_MB" in msg for msg in loguru_warnings)

    @pytest.mark.parametrize("value", ["-0.1", "1.1", "2.0", "-1.0"])
    def test_compaction_threshold_out_of_range_warns_and_skips(
        self, monkeypatch: pytest.MonkeyPatch, loguru_warnings: list[str], value: str
    ) -> None:
        """Out-of-range threshold logs a warning and uses the default."""
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_THRESHOLD", value
        )
        config = DatabaseConfig.load_from_env()
        assert "compaction_threshold" not in config
        assert any("COMPACTION_THRESHOLD" in msg for msg in loguru_warnings)

    def test_compaction_min_size_negative_warns_and_skips(
        self, monkeypatch: pytest.MonkeyPatch, loguru_warnings: list[str]
    ) -> None:
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_MIN_SIZE_MB", "-5"
        )
        config = DatabaseConfig.load_from_env()
        assert "compaction_min_size_mb" not in config
        assert any("COMPACTION_MIN_SIZE_MB" in msg for msg in loguru_warnings)

    def test_compaction_threshold_boundary_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """0.0 and 1.0 are at the boundary and must be accepted."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__COMPACTION_THRESHOLD", "0.0")
        assert DatabaseConfig.load_from_env()["compaction_threshold"] == 0.0
        monkeypatch.setenv("CHUNKHOUND_DATABASE__COMPACTION_THRESHOLD", "1.0")
        assert DatabaseConfig.load_from_env()["compaction_threshold"] == 1.0
