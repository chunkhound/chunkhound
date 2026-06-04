"""Unit tests for research configuration parsing."""

import pytest

from chunkhound.core.config.research_config import ResearchConfig


def test_load_from_env_parses_exploration_query_generation_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CHUNKHOUND_RESEARCH_EXPLORATION_QUERY_GENERATION_MAX_COMPLETION_TOKENS",
        "12345",
    )

    config = ResearchConfig.load_from_env()

    assert config["exploration_query_generation_max_completion_tokens"] == 12345


def test_exploration_query_generation_token_budget_is_positive() -> None:
    with pytest.raises(ValueError):
        ResearchConfig(exploration_query_generation_max_completion_tokens=0)
