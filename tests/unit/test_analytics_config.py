"""Unit tests for analytics configuration parsing."""

import json
import re
from pathlib import Path

import pytest
from pydantic import SecretStr

from chunkhound.core.config.analytics_config import AnalyticsConfig
from chunkhound.core.config.config import Config


def test_disabled_by_default() -> None:
    """analytics.enabled must default to False -- opt-in, per design."""
    assert AnalyticsConfig().enabled is False


def test_anonymize_defaults_to_full() -> None:
    assert AnalyticsConfig().anonymize == "full"


def test_anonymize_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        AnalyticsConfig(anonymize="incognito")


def test_save_sensitive_data_defaults_to_false() -> None:
    """Redact action content by default -- admins must explicitly opt in."""
    assert AnalyticsConfig().save_sensitive_data is False


def test_load_from_env_parses_enabled_and_anonymize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__ENABLED", "true")
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__ANONYMIZE", "hashed")

    config = AnalyticsConfig.load_from_env()

    assert config["enabled"] is True
    assert config["anonymize"] == "hashed"


def test_load_from_env_parses_save_sensitive_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__SAVE_SENSITIVE_DATA", "true")

    config = AnalyticsConfig.load_from_env()

    assert config["save_sensitive_data"] is True


def test_load_from_env_parses_s3_and_flush_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CHUNKHOUND_ANALYTICS__S3_ENDPOINT_URL", "https://minio.internal"
    )
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__S3_BUCKET", "usage-events")
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__FLUSH_INTERVAL_SECONDS", "60")
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__FLUSH_BATCH_SIZE", "10")

    config = AnalyticsConfig.load_from_env()

    assert config["s3_endpoint_url"] == "https://minio.internal"
    assert config["s3_bucket"] == "usage-events"
    assert config["flush_interval_seconds"] == 60
    assert config["flush_batch_size"] == 10


def test_load_from_env_parses_s3_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKHOUND_AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("CHUNKHOUND_AWS_SECRET_ACCESS_KEY", "test-secret")

    config = AnalyticsConfig.load_from_env()

    assert config["s3_access_key"] == "AKIA_TEST"
    assert config["s3_secret_key"] == "test-secret"


def test_load_from_env_ignores_malformed_integers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__FLUSH_INTERVAL_SECONDS", "not-a-number")

    config = AnalyticsConfig.load_from_env()

    assert "flush_interval_seconds" not in config


def test_config_composes_analytics_with_defaults(tmp_path: Path) -> None:
    """Config() must always carry a usable, disabled-by-default AnalyticsConfig,
    mirroring every other sub-config's default_factory wiring. Uses an empty
    tmp_path as target_dir so this doesn't pick up this repo's own
    .chunkhound.json (which may set its own analytics values)."""
    config = Config(target_dir=tmp_path)
    assert isinstance(config.analytics, AnalyticsConfig)
    assert config.analytics.enabled is False


def test_config_picks_up_analytics_env_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__ENABLED", "true")
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__ANONYMIZE", "anonymous")

    config = Config(target_dir=tmp_path)

    assert config.analytics.enabled is True
    assert config.analytics.anonymize == "anonymous"


def test_credential_shaped_fields_are_secret_str() -> None:
    """The S3 write credential (`s3_access_key`/`s3_secret_key`) is a
    persisted pydantic field (so it can be set via .chunkhound.json), but
    must be typed as `SecretStr` -- matching `EmbeddingConfig.api_key` --
    so it can never round-trip into a plain string in `repr()` or
    `model_dump()`. Matches on shape (key/secret/credential/token in the
    name) so this still catches a differently-named credential field added
    later without a SecretStr annotation."""
    credential_like = re.compile(r"key|secret|credential|token", re.IGNORECASE)
    offending = [
        name
        for name, field in AnalyticsConfig.model_fields.items()
        if credential_like.search(name) and field.annotation != (SecretStr | None)
    ]
    assert offending == []


def test_config_to_dict_masks_analytics_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The S3 credential is a SecretStr field, so it must never appear in
    plaintext in Config.to_dict() (used for e.g. persisting/echoing config),
    regardless of whether it was sourced from env vars or .chunkhound.json."""
    sentinel_key = "AKIA_SENTINEL_DO_NOT_SERIALIZE"
    sentinel_secret = "SENTINEL_SECRET_DO_NOT_SERIALIZE"
    monkeypatch.setenv("CHUNKHOUND_AWS_ACCESS_KEY_ID", sentinel_key)
    monkeypatch.setenv("CHUNKHOUND_AWS_SECRET_ACCESS_KEY", sentinel_secret)

    config = Config(target_dir=tmp_path)
    serialized = json.dumps(config.to_dict(), default=str)

    assert sentinel_key not in serialized
    assert sentinel_secret not in serialized


def test_repr_masks_s3_credentials() -> None:
    config = AnalyticsConfig(s3_access_key="AKIA_TEST", s3_secret_key="shh")
    assert "AKIA_TEST" not in repr(config)
    assert "shh" not in repr(config)
