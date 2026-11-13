"""Contract: a blank rerank_model is unset, not an explicit model.

Base config semantics treated ``""`` as unset. Emitting it as an explicit model
suppresses the provider default and fails at query time, so the config boundary
must keep the key absent. Blank means ``None``, ``""``, or whitespace-only.
"""

from __future__ import annotations

import pytest

from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory


@pytest.mark.parametrize("model", ["", "   ", "\t\n"])
def test_blank_rerank_model_uses_provider_default(model: str) -> None:
    config = EmbeddingConfig(
        provider="voyageai",
        model="voyage-3.5",
        rerank_model=model,
        api_key="test-key",
    )

    assert config.rerank_model is None
    assert "rerank_model" not in config.get_provider_config()

    provider = EmbeddingProviderFactory.create_provider(config)

    assert provider.supports_reranking()


def test_none_rerank_model_uses_provider_default() -> None:
    config = EmbeddingConfig(
        provider="voyageai",
        model="voyage-3.5",
        rerank_model=None,
        api_key="test-key",
    )

    assert "rerank_model" not in config.get_provider_config()


def test_surrounding_whitespace_is_trimmed() -> None:
    config = EmbeddingConfig(
        provider="voyageai",
        model="voyage-3.5",
        rerank_model="  rerank-3-lite  ",
        api_key="test-key",
    )

    assert config.rerank_model == "rerank-3-lite"
    assert config.get_provider_config()["rerank_model"] == "rerank-3-lite"


def test_cohere_format_rejects_blank_model() -> None:
    # cohere/voyage need an explicit model; a blank value must fail at load,
    # not later when the factory receives an absent key.
    with pytest.raises(ValueError, match="rerank_model is required"):
        EmbeddingConfig(
            provider="openai",
            rerank_format="cohere",
            rerank_model="   ",
            rerank_url="https://host/rerank",
            api_key="test-key",
        )


def test_unused_malformed_rerank_url_is_ignored_when_model_is_blank() -> None:
    # A blank model is inactive, so an unused malformed URL must not block load.
    config = EmbeddingConfig(
        provider="openai",
        rerank_model="   ",
        rerank_format="auto",
        rerank_url="ftp://unused/rerank",
        api_key="test-key",
    )

    assert "rerank_model" not in config.get_provider_config()


def test_tei_blank_model_uses_provider_default() -> None:
    config = EmbeddingConfig(
        provider="openai",
        rerank_format="tei",
        rerank_model="   ",
        rerank_url="https://host/rerank",
        api_key="test-key",
    )

    assert config.rerank_model is None
    assert "rerank_model" not in config.get_provider_config()


def test_blank_rerank_model_from_env_uses_provider_default(
    monkeypatch, clean_environment
) -> None:
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING__RERANK_MODEL", "   ")
    env_config = EmbeddingConfig.load_from_env()
    env_config.update(
        provider="voyageai",
        model="voyage-3.5",
        api_key="test-key",
    )

    config = EmbeddingConfig(**env_config)

    assert config.rerank_model is None
    assert "rerank_model" not in config.get_provider_config()
