"""Caller-visible diagnostics for degraded structured research stages."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from chunkhound.core.utils.structured_reasoning_diagnostics import (
    current_empty_failures,
    record_empty_failure,
    structured_reasoning_failure_key,
)
from chunkhound.providers.llm.capability_cache import LLMCapabilityStore
from chunkhound.providers.llm.openai_compatible_provider import OpenAICompatibleProvider
from chunkhound.services.research.v1.pluggable_research_service import (
    PluggableResearchService,
    apply_structured_reasoning_degradation_note,
)


def test_zero_failed_structured_stages_leave_result_unchanged() -> None:
    result: dict[str, Any] = {"answer": "original", "metadata": {}}

    assert (
        apply_structured_reasoning_degradation_note(
            result, failed_stages=0, provider="openrouter", model="m"
        )
        is result
    )
    assert result == {"answer": "original", "metadata": {}}


def test_failed_structured_stages_prepend_note_and_metadata_warning() -> None:
    result: dict[str, Any] = {"answer": "original answer", "metadata": {}}

    assert (
        apply_structured_reasoning_degradation_note(
            result,
            failed_stages=2,
            provider="openrouter",
            model="openai/gpt-oss-20b",
        )
        is result
    )
    assert result["answer"].startswith("> **Note:**")
    assert "original answer" in result["answer"]
    warnings = result["metadata"]["warnings"]
    assert len(warnings) == 1
    assert "2 structured stage(s) failed" in warnings[0]
    assert "openrouter:openai/gpt-oss-20b" in warnings[0]


def test_service_attaches_note_for_new_failures_only() -> None:
    provider = Mock()
    provider.get_structured_reasoning_health.return_value = {
        "provider": "openrouter",
        "model": "m",
        "capability": "rejected",
        "empty_failures": 3,
        "last_empty_ts": 1.0,
    }
    service = PluggableResearchService.__new__(PluggableResearchService)
    service._llm_manager = SimpleNamespace(get_utility_provider=lambda: provider)

    # Historical health does not attach a note to a fresh request.
    unchanged = {"answer": "a", "metadata": {}}
    assert service._attach_structured_reasoning_note(unchanged) is unchanged
    token = current_empty_failures.set({})
    try:
        for _ in range(3):
            record_empty_failure(provider)
        result = service._attach_structured_reasoning_note(
            {"answer": "a", "metadata": {}}
        )
        assert "> **Note:** 3 structured stage(s) failed" in result["answer"]
    finally:
        current_empty_failures.reset(token)


@pytest.mark.asyncio
async def test_research_failure_restores_request_diagnostics() -> None:
    service = PluggableResearchService.__new__(PluggableResearchService)
    service._emit_event = AsyncMock(side_effect=RuntimeError("start failed"))
    previous: dict[str, int] = {}
    token = current_empty_failures.set(previous)
    try:
        with pytest.raises(RuntimeError, match="start failed"):
            await service.deep_research("broken request")
        assert current_empty_failures.get() is previous
    finally:
        current_empty_failures.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("warning_surface", ["answer", "metadata"])
async def test_overlapping_research_warns_only_for_its_own_failed_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, warning_surface: str
) -> None:
    """Two MCP-style services share a provider, not each other's degradation."""
    monkeypatch.setenv("CHUNKHOUND_LLM_CAPABILITY_CACHE", str(tmp_path / "cache.json"))
    store = LLMCapabilityStore()
    store.set("openrouter", "test-model", "rejected")
    provider = OpenAICompatibleProvider(
        provider_name="openrouter",
        api_key="sk-test",
        model="test-model",
        default_base_url="https://openrouter.ai/api/v1",
        supports_structured_outputs=False,
        structured_reasoning_disable_extra_body={"reasoning": {"enabled": False}},
        capability_store=store,
    )
    healthy_entered = asyncio.Event()
    failed_stage_finished = asyncio.Event()

    async def completion(**kwargs: Any) -> SimpleNamespace:
        prompt = kwargs["messages"][-1]["content"]
        if "affected-request" in prompt:
            # Both deep_research calls have taken their initial health snapshot.
            await healthy_entered.wait()
            content = None
        else:
            healthy_entered.set()
            await failed_stage_finished.wait()
            content = '{"queries": ["first perspective", "second perspective"]}'
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        )

    async def search(**kwargs: Any) -> list[dict[str, Any]]:
        if kwargs["query"] == "affected-request":
            # Real query expansion has caught the real provider's empty response.
            assert kwargs["expanded_queries"] == ["affected-request"]
            failed_stage_finished.set()
        else:
            assert kwargs["expanded_queries"] == [
                "healthy-request", "first perspective", "second perspective"
            ]
        return []

    monkeypatch.setattr(
        provider._client.chat.completions, "create", AsyncMock(side_effect=completion)
    )
    manager = SimpleNamespace(
        get_utility_provider=lambda: provider,
        get_synthesis_provider=lambda: provider,
    )

    def service() -> PluggableResearchService:
        instance = PluggableResearchService.__new__(PluggableResearchService)
        instance._llm_manager = manager
        instance._config = SimpleNamespace(
            query_expansion_enabled=True, num_expanded_queries=2
        )
        instance._path_filter = None
        instance._emit_event = AsyncMock()
        instance._unified_search_helper = SimpleNamespace(unified_search=search)
        instance._exploration_strategy = SimpleNamespace(
            name="empty-index", explore=AsyncMock(return_value=([], {}, {}))
        )
        return instance

    try:
        affected, healthy = await asyncio.wait_for(
            asyncio.gather(
                service().deep_research("affected-request"),
                service().deep_research("healthy-request"),
            ),
            timeout=5,
        )
    finally:
        await provider._client.close()

    # Parameterize the surfaces so one bad assertion cannot hide the other.
    if warning_surface == "answer":
        assert "> **Note:** 1 structured stage(s) failed" in affected["answer"]
        assert "> **Note:**" not in healthy["answer"]
    else:
        warnings = affected["metadata"]["warnings"]
        assert len(warnings) == 1
        assert "1 structured stage(s) failed" in warnings[0]
        assert "openrouter:test-model" in warnings[0]
        assert not healthy["metadata"].get("warnings")


def test_service_without_provider_health_does_not_attach_note() -> None:
    service = PluggableResearchService.__new__(PluggableResearchService)
    provider = object()
    service._llm_manager = SimpleNamespace(get_utility_provider=lambda: provider)
    unchanged = {"answer": "a", "metadata": {}}

    token = current_empty_failures.set({structured_reasoning_failure_key(provider): 1})
    try:
        assert service._attach_structured_reasoning_note(unchanged) is unchanged
        assert unchanged == {"answer": "a", "metadata": {}}
    finally:
        current_empty_failures.reset(token)
