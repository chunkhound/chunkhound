from __future__ import annotations

from dataclasses import dataclass

import pytest

from chunkhound.code_mapper.hyde import run_hyde_only_query
from chunkhound.code_mapper.models import HydeConfig


@dataclass
class _FakeResponse:
    content: str | None


class _FakeProvider:
    def __init__(self, content: str | None, raise_error: bool = False) -> None:
        self.content = content
        self.raise_error = raise_error
        self.calls: list[tuple[str, int]] = []

    async def complete(self, *, prompt: str, max_completion_tokens: int) -> _FakeResponse:
        self.calls.append((prompt, max_completion_tokens))
        if self.raise_error:
            raise RuntimeError("boom")
        return _FakeResponse(self.content)


class _FakeLLMManager:
    def __init__(self, provider: _FakeProvider | None) -> None:
        self._provider = provider

    def is_configured(self) -> bool:
        return True

    def get_synthesis_provider(self) -> _FakeProvider | None:
        return self._provider


@pytest.mark.asyncio
async def test_run_hyde_only_query_requires_llm() -> None:
    result, ok = await run_hyde_only_query(llm_manager=None, prompt="prompt")

    assert ok is False
    assert "LLM not configured" in result


@pytest.mark.asyncio
async def test_run_hyde_only_query_handles_missing_provider() -> None:
    manager = _FakeLLMManager(provider=None)

    result, ok = await run_hyde_only_query(llm_manager=manager, prompt="prompt")

    assert ok is False
    assert "Synthesis provider unavailable" in result


@pytest.mark.asyncio
async def test_run_hyde_only_query_returns_content() -> None:
    provider = _FakeProvider("answer")
    hyde_cfg = HydeConfig(
        max_scope_files=1,
        max_snippet_files=0,
        max_snippet_chars=0,
        max_completion_tokens=5,
        max_snippet_tokens=1,
    )

    result, ok = await run_hyde_only_query(
        llm_manager=None,
        prompt="prompt",
        provider_override=provider,
        hyde_cfg=hyde_cfg,
    )

    assert ok is True
    assert result == "answer"
    assert provider.calls == [("prompt", 5)]


@pytest.mark.asyncio
async def test_run_hyde_only_query_handles_provider_error() -> None:
    provider = _FakeProvider("answer", raise_error=True)
    hyde_cfg = HydeConfig(
        max_scope_files=1,
        max_snippet_files=0,
        max_snippet_chars=0,
        max_completion_tokens=5,
        max_snippet_tokens=1,
    )

    result, ok = await run_hyde_only_query(
        llm_manager=None,
        prompt="prompt",
        provider_override=provider,
        hyde_cfg=hyde_cfg,
    )

    assert ok is False
    assert result.startswith("HyDE-only synthesis failed:")
