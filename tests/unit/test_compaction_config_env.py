"""Tests for compaction environment variable configuration."""

import pytest

from chunkhound.core.config.database_config import DatabaseConfig


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

    def test_compaction_threshold_invalid_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unparseable threshold must raise — no silent fallback to default."""
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_THRESHOLD", "not-a-number"
        )
        with pytest.raises(
            ValueError,
            match=r"CHUNKHOUND_DATABASE__COMPACTION_THRESHOLD=.*not a valid number",
        ):
            DatabaseConfig.load_from_env()

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

    def test_compaction_min_size_invalid_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unparseable min-size must raise — no silent fallback to default."""
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_MIN_SIZE_MB", "abc"
        )
        with pytest.raises(
            ValueError,
            match=r"CHUNKHOUND_DATABASE__COMPACTION_MIN_SIZE_MB=.*not a valid integer",
        ):
            DatabaseConfig.load_from_env()

    @pytest.mark.parametrize("value", ["-0.1", "1.1", "2.0", "-1.0"])
    def test_compaction_threshold_out_of_range_raises(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Out-of-range threshold must raise a loud, explicit error.

        "No silent errors" policy — a user typo like COMPACTION_THRESHOLD=1.5
        must not silently fall back to the default.
        """
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_THRESHOLD", value
        )
        with pytest.raises(
            ValueError,
            match=r"CHUNKHOUND_DATABASE__COMPACTION_THRESHOLD=.*out of range",
        ):
            DatabaseConfig.load_from_env()

    def test_compaction_min_size_negative_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "CHUNKHOUND_DATABASE__COMPACTION_MIN_SIZE_MB", "-5"
        )
        with pytest.raises(
            ValueError,
            match=r"CHUNKHOUND_DATABASE__COMPACTION_MIN_SIZE_MB=.*must be >= 0",
        ):
            DatabaseConfig.load_from_env()

    def test_compaction_threshold_boundary_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """0.0 and 1.0 are at the boundary and must be accepted."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__COMPACTION_THRESHOLD", "0.0")
        assert DatabaseConfig.load_from_env()["compaction_threshold"] == 0.0
        monkeypatch.setenv("CHUNKHOUND_DATABASE__COMPACTION_THRESHOLD", "1.0")
        assert DatabaseConfig.load_from_env()["compaction_threshold"] == 1.0
