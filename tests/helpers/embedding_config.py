"""Declared real provider inputs; deterministic tests use the local fake provider."""

import pytest

from chunkhound.core.config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.embeddings import EmbeddingManager
from tests.fixtures.fake_providers import FakeEmbeddingProvider
from tests.helpers.live import live_inputs


def get_embedding_config_for_tests() -> dict:
    data = live_inputs("embedding")
    if not data.get("provider"):
        raise ValueError("Live embedding input requires an explicit provider")
    config = EmbeddingConfig(**data)
    if not config.is_provider_configured():
        pytest.fail("Live prerequisite missing: embedding provider credentials")
    return data


def create_embedding_manager_for_tests(config_dict: dict | None) -> EmbeddingManager:
    """``None`` deliberately returns ``FakeEmbeddingProvider`` for deterministic
    offline tests; a supplied invalid dict still raises rather than silently
    falling back to offline behavior.
    """
    provider = (
        FakeEmbeddingProvider()
        if config_dict is None
        else EmbeddingProviderFactory.create_provider(EmbeddingConfig(**config_dict))
    )
    manager = EmbeddingManager()
    manager.register_provider(provider, set_default=True)
    return manager
