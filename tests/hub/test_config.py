import argparse
import json

import pytest

from chunkhound.core.config.config import Config


def test_committed_full_url_overrides_env_without_remote_scrub(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNKHOUND_HUB__URL", "https://env.invalid/catalogue?a=1")
    url = "https://team.invalid/custom/catalogue?signature=secret&branch=main"
    (tmp_path / ".chunkhound.json").write_text(json.dumps({"hub": {"url": url}}))
    config = Config(target_dir=tmp_path, skip_layers={"global", "config_file", "cli"})
    assert config.hub.url == url
    (tmp_path / ".chunkhound.json").unlink()
    config = Config(target_dir=tmp_path, skip_layers={"global", "config_file", "cli"})
    assert config.hub.url == "https://env.invalid/catalogue?a=1"
    monkeypatch.setenv("CHUNKHOUND_HUB__URL", "")
    config = Config(target_dir=tmp_path, skip_layers={"global", "config_file", "cli"})
    assert config.hub.url is None


def test_memory_database_sentinel_survives_config_normalization(tmp_path):
    config = Config(
        target_dir=tmp_path,
        database={"path": ":memory:"},
        skip_layers={"global", "env", "local_config", "config_file", "cli"},
    )
    assert str(config.database.path) == ":memory:"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (None, None),
        ("", None),
        (" \t\n", None),
        ("not a URL", "not a URL"),
        (
            " ftp://team.invalid/catalogue?signature=secret&branch=main ",
            " ftp://team.invalid/catalogue?signature=secret&branch=main ",
        ),
    ],
)
def test_hub_url_is_opaque_except_blank_values(tmp_path, url, expected):
    config = Config(
        target_dir=tmp_path,
        hub={"url": url},
        skip_layers={"global", "env", "local_config", "config_file", "cli"},
    )
    assert config.hub.url == expected


def test_skipping_env_leaves_hub_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNKHOUND_HUB__URL", "https://env.invalid/catalogue?a=1")
    config = Config(
        target_dir=tmp_path,
        skip_layers={"global", "env", "local_config", "config_file", "cli"},
    )
    assert config.hub.url is None


ARCHIVE_DEFAULTS = {
    "max_members": 100_000,
    "max_path_depth": 32,
    "max_path_bytes": 4096,
    "max_total_path_bytes": 32 * 1024 * 1024,
    "max_extracted_bytes": 32 * 1024**3,
    "max_decompressed_bytes": 40 * 1024**3,
}


def test_archive_limits_have_exact_defaults(tmp_path):
    config = Config(
        target_dir=tmp_path,
        skip_layers={"global", "env", "local_config", "config_file", "cli"},
    )
    assert config.hub.archive_limits.model_dump() == ARCHIVE_DEFAULTS
    from chunkhound.core.config.hub_config import ArchiveLimits

    assert ArchiveLimits().model_dump() == ARCHIVE_DEFAULTS


@pytest.mark.parametrize("field", ARCHIVE_DEFAULTS)
@pytest.mark.parametrize(
    "value", [True, False, 0, -1, 1.5, 1.0, float("inf"), None, "unlimited", "12"]
)
def test_archive_limits_reject_non_positive_or_non_integer_config(
    tmp_path, field, value
):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Config(
            target_dir=tmp_path,
            hub={"archive_limits": {field: value}},
            skip_layers={"global", "env", "local_config", "config_file", "cli"},
        )


@pytest.mark.parametrize("field", ARCHIVE_DEFAULTS)
def test_archive_limit_environment_decimal_values_and_skipping_env(
    tmp_path, monkeypatch, field
):
    monkeypatch.setenv(f"CHUNKHOUND_HUB__ARCHIVE_LIMITS__{field.upper()}", "123")
    skips = {"global", "local_config", "config_file", "cli"}
    config = Config(target_dir=tmp_path, skip_layers=skips)
    assert getattr(config.hub.archive_limits, field) == 123
    isolated = Config(target_dir=tmp_path, skip_layers=skips | {"env"})
    assert getattr(isolated.hub.archive_limits, field) == ARCHIVE_DEFAULTS[field]


@pytest.mark.parametrize(
    "value", ["true", "false", "0", "-1", "1.5", "1.0", "inf", "unlimited", ""]
)
def test_archive_limit_invalid_environment_is_rejected(tmp_path, monkeypatch, value):
    from pydantic import ValidationError

    monkeypatch.setenv("CHUNKHOUND_HUB__ARCHIVE_LIMITS__MAX_MEMBERS", value)
    with pytest.raises(ValidationError):
        Config(
            target_dir=tmp_path,
            skip_layers={"global", "local_config", "config_file", "cli"},
        )


def test_archive_limits_follow_existing_nested_layer_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNKHOUND_HUB__ARCHIVE_LIMITS__MAX_MEMBERS", "11")
    monkeypatch.setenv("CHUNKHOUND_HUB__ARCHIVE_LIMITS__MAX_PATH_DEPTH", "12")
    monkeypatch.setenv("CHUNKHOUND_HUB__ARCHIVE_LIMITS__MAX_PATH_BYTES", "unlimited")
    global_file = tmp_path / "global.json"
    global_file.write_text(
        json.dumps(
            {
                "hub": {
                    "archive_limits": {
                        "max_members": 21,
                        "max_path_bytes": 2000,
                    }
                }
            }
        )
    )
    monkeypatch.setenv("CHUNKHOUND_GLOBAL_CONFIG_FILE", str(global_file))
    local = tmp_path / ".chunkhound.json"
    local.write_text(json.dumps({"hub": {"archive_limits": {"max_members": 31}}}))
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"hub": {"archive_limits": {"max_members": 41}}}))
    assert (
        Config(target_dir=tmp_path, skip_layers={"cli"}).hub.archive_limits.max_members
        == 31
    )
    config = Config(
        target_dir=tmp_path,
        args=argparse.Namespace(config=str(explicit)),
        skip_layers={"cli"},
    )
    assert config.hub.archive_limits.max_members == 41
    assert config.hub.archive_limits.max_path_depth == 12
    assert config.hub.archive_limits.max_path_bytes == 2000
    direct = Config(
        target_dir=tmp_path,
        args=argparse.Namespace(config=str(explicit)),
        hub={"archive_limits": {"max_members": 51}},
        skip_layers={"cli"},
    )
    assert direct.hub.archive_limits.max_members == 51
    assert direct.hub.archive_limits.max_path_depth == 12
    global_only = Config(
        target_dir=tmp_path, skip_layers={"local_config", "config_file", "cli"}
    )
    assert global_only.hub.archive_limits.max_members == 21
