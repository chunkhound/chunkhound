"""Regression test: the factory must forward performance settings to the OpenAI provider.

The OpenAI factory path used to drop timeout/batch_size/max_retries from the
provider config (the VoyageAI path forwarded them), so OpenAI-compatible
providers always ran with the hardcoded 30s timeout regardless of config.
"""

from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory


def _make(config_overrides):
    config = {
        "api_key": "test-key",
        "base_url": "http://localhost:9999/v1",
        "model": "test-model",
    }
    config.update(config_overrides)
    return EmbeddingProviderFactory._create_openai_provider(config)


def test_openai_factory_forwards_performance_settings():
    provider = _make({"timeout": 120, "batch_size": 16, "max_retries": 7})
    assert provider._timeout == 120
    assert provider._batch_size == 16
    assert provider._retry_attempts == 7


def test_openai_factory_defaults_when_unset():
    provider = _make({})
    assert provider._timeout == 30
    assert provider._batch_size == 100
    assert provider._retry_attempts == 3
