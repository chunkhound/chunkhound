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
