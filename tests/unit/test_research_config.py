"""Unit tests for research configuration parsing."""

import pytest

from chunkhound.core.config.research_config import ResearchConfig


def test_load_from_env_parses_depth_exploration_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CHUNKHOUND_RESEARCH_DEPTH_EXPLORATION_MAX_COMPLETION_TOKENS",
        "12345",
    )

    config = ResearchConfig.load_from_env()

    assert config["depth_exploration_max_completion_tokens"] == 12345


def test_depth_exploration_token_budget_is_positive() -> None:
    with pytest.raises(ValueError):
        ResearchConfig(depth_exploration_max_completion_tokens=0)
