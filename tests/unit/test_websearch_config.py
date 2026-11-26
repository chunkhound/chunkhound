"""Contracts for the WebSearchConfig section (file/env/CLI/defaults)."""

from __future__ import annotations

import argparse
import asyncio

import pytest
from pydantic import ValidationError

from chunkhound.api.cli.utils.config_factory import create_validated_config
from chunkhound.core.config.config import Config
from chunkhound.core.config.websearch_config import WebSearchConfig


def test_defaults() -> None:
    cfg = WebSearchConfig()
    assert cfg.backends == ["duckduckgo", "bing"]
    assert cfg.selection == "priority"
    assert cfg.cooldown_seconds == 900


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CHUNKHOUND_WEBSEARCH_BACKENDS", "bing,duckduckgo")
    monkeypatch.setenv("CHUNKHOUND_WEBSEARCH_SELECTION", "random_healthy")
    monkeypatch.setenv("CHUNKHOUND_WEBSEARCH_COOLDOWN_SECONDS", "60")
    cfg = WebSearchConfig()
    assert cfg.backends == ["bing", "duckduckgo"]
    assert cfg.selection == "random_healthy"
    assert cfg.cooldown_seconds == 60


def test_config_loads_env_layer(monkeypatch) -> None:
    monkeypatch.setenv("CHUNKHOUND_WEBSEARCH_BACKENDS", "bing")
    cfg = Config.from_environment()
    assert cfg.websearch.backends == ["bing"]


def test_config_dict_to_model_coercion() -> None:
    cfg = Config(websearch={"backends": ["bing"], "cooldown_seconds": 30})
    assert isinstance(cfg.websearch, WebSearchConfig)
    assert cfg.websearch.backends == ["bing"]
    assert cfg.websearch.cooldown_seconds == 30


def test_cli_overrides_win_over_env(monkeypatch) -> None:
    monkeypatch.setenv("CHUNKHOUND_WEBSEARCH_BACKENDS", "duckduckgo,bing")
    ns = argparse.Namespace(
        websearch_backends="bing",
        websearch_selection="random_healthy",
        websearch_cooldown_seconds=120,
    )
    cfg = Config(args=ns)
    assert cfg.websearch.backends == ["bing"]
    assert cfg.websearch.selection == "random_healthy"
    assert cfg.websearch.cooldown_seconds == 120


def test_extract_cli_overrides() -> None:
    ns = argparse.Namespace(
        websearch_backends="bing,duckduckgo",
        websearch_selection="random_healthy",
        websearch_cooldown_seconds=60,
    )
    overrides = WebSearchConfig.extract_cli_overrides(ns)
    assert overrides["backends"] == ["bing", "duckduckgo"]
    assert overrides["selection"] == "random_healthy"
    assert overrides["cooldown_seconds"] == 60


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown websearch backend"):
        WebSearchConfig(backends=["google"])


def test_empty_backends_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one engine"):
        WebSearchConfig(backends=[])


def test_backends_dedupe_preserves_order() -> None:
    cfg = WebSearchConfig(backends=["bing", "duckduckgo", "bing"])
    assert cfg.backends == ["bing", "duckduckgo"]


def test_invalid_selection_rejected() -> None:
    with pytest.raises(ValidationError):
        WebSearchConfig(selection="round_robin")


def test_file_overrides_env_backends(tmp_path, monkeypatch) -> None:
    (tmp_path / ".chunkhound.json").write_text(
        '{"websearch":{"backends":["bing"],"selection":"random_healthy"}}'
    )
    monkeypatch.setenv("CHUNKHOUND_WEBSEARCH_BACKENDS", "duckduckgo,bing")
    cfg = Config(target_dir=tmp_path)
    assert cfg.websearch.backends == ["bing"]
    # env fills only the fields the file does not set; file values survive
    assert cfg.websearch.selection == "random_healthy"


def test_file_overrides_defaults(tmp_path) -> None:
    (tmp_path / ".chunkhound.json").write_text(
        '{"websearch":{"backends":["bing"],"selection":"random_healthy"}}'
    )
    cfg = Config(target_dir=tmp_path)
    assert cfg.websearch.backends == ["bing"]
    assert cfg.websearch.selection == "random_healthy"


def test_env_invalid_cooldown_defeated_by_cli_override(monkeypatch) -> None:
    """Invalid env cooldown must not defeat a higher-precedence CLI override."""
    monkeypatch.setenv("CHUNKHOUND_WEBSEARCH_COOLDOWN_SECONDS", "oops")
    ns = argparse.Namespace(
        websearch_backends=None,
        websearch_selection=None,
        websearch_cooldown_seconds=120,
    )
    cfg = Config(args=ns)
    assert cfg.websearch.cooldown_seconds == 120


def test_env_invalid_cooldown_fails_without_override(monkeypatch) -> None:
    """Invalid env cooldown fails loudly when no override replaces it."""
    monkeypatch.setenv("CHUNKHOUND_WEBSEARCH_COOLDOWN_SECONDS", "oops")
    with pytest.raises(ValidationError):
        Config()


def _fallback_args(path: str | None = None) -> argparse.Namespace:
    """Arg namespace shared by the create_validated_config fallback tests."""
    return argparse.Namespace(
        command="index",
        config=None,
        path=path,
        debug=False,
        verbose=False,
    )


def test_create_validated_config_fallback_survives_invalid_websearch_env(
    monkeypatch,
) -> None:
    """Config-error recovery must not re-read poisoned websearch env in fallback."""
    monkeypatch.setenv("CHUNKHOUND_WEBSEARCH_BACKENDS", "google")
    args = _fallback_args()
    config, errors = asyncio.run(create_validated_config(args, "index"))
    assert config.websearch.backends == ["duckduckgo", "bing"]
    assert errors


def test_create_validated_config_fallback_survives_invalid_research_env(
    monkeypatch, tmp_path
) -> None:
    """Fallback must not re-read a poisoned sibling research env var."""
    monkeypatch.setenv("CHUNKHOUND_RESEARCH_NUM_EXPANDED_QUERIES", "oops")
    args = _fallback_args(str(tmp_path))
    config, errors = asyncio.run(create_validated_config(args, "index"))
    assert config.research.num_expanded_queries == 2
    assert config.websearch.backends == ["duckduckgo", "bing"]
    assert errors


def test_create_validated_config_fallback_survives_invalid_fetchurl_env(
    monkeypatch, tmp_path
) -> None:
    """Fallback must not re-read a poisoned sibling fetchurl env var."""
    monkeypatch.setenv("CHUNKHOUND_FETCHURL_MAX_RETRIES", "oops")
    args = _fallback_args(str(tmp_path))
    config, errors = asyncio.run(create_validated_config(args, "index"))
    assert config.fetchurl.max_retries == 3
    assert config.websearch.backends == ["duckduckgo", "bing"]
    assert errors
