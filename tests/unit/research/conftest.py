"""Shared fakes and fixtures for synthesis engine tests.

Provides the minimal parent/embedding/LLM stubs that let a
``SynthesisEngine`` be instantiated in isolation, plus an LLM manager
whose provider records every ``complete`` call so tests can assert on
the prompt text sent to the model.
"""

import pytest

from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.shared.citation_manager import CitationManager
from tests.fixtures.fake_providers import FakeLLMProvider


class FakeEmbeddingManager:
    def __init__(self, provider):
        self._provider = provider

    def get_provider(self):
        return self._provider


class FakeParent:
    def __init__(self, provider):
        self._embedding_manager = FakeEmbeddingManager(provider)
        self._citation_manager = CitationManager()

    async def _emit_event(self, *args, **kwargs):
        return None


class CapturingFakeLLMProvider(FakeLLMProvider):
    def __init__(self):
        super().__init__()
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "max_completion_tokens": max_completion_tokens,
                "timeout": timeout,
            }
        )
        return await super().complete(
            prompt,
            system=system,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
        )


@pytest.fixture()
def llm_manager(monkeypatch):
    fake_provider = FakeLLMProvider()

    def _fake_create_provider(self, config):
        return fake_provider

    monkeypatch.setattr(LLMManager, "_create_provider", _fake_create_provider)
    utility_config = {"provider": "fake", "model": "fake-gpt"}
    synthesis_config = {"provider": "fake", "model": "fake-gpt"}
    return LLMManager(utility_config, synthesis_config)


@pytest.fixture()
def capturing_llm_manager(monkeypatch):
    fake_provider = CapturingFakeLLMProvider()

    def _fake_create_provider(self, config):
        return fake_provider

    monkeypatch.setattr(LLMManager, "_create_provider", _fake_create_provider)
    utility_config = {"provider": "fake", "model": "fake-gpt"}
    synthesis_config = {"provider": "fake", "model": "fake-gpt"}
    return LLMManager(utility_config, synthesis_config), fake_provider
