"""Caller-visible diagnostics for degraded structured research stages."""

from types import SimpleNamespace
from typing import Any

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
    health = iter(
        (
            None,
            {
                "provider": "openrouter",
                "model": "m",
                "capability": "rejected",
                "empty_failures": 3,
                "last_empty_ts": 1.0,
            },
        )
    )
    provider = SimpleNamespace(get_structured_reasoning_health=lambda: next(health))
    service = PluggableResearchService.__new__(PluggableResearchService)
    service._llm_manager = SimpleNamespace(get_utility_provider=lambda: provider)

    start_health = service._structured_reasoning_health()
    assert start_health is None
    result = service._attach_structured_reasoning_note(
        {"answer": "a", "metadata": {}}, start_health
    )
    assert "> **Note:** 3 structured stage(s) failed" in result["answer"]

    no_health_provider = SimpleNamespace(get_structured_reasoning_health=lambda: None)
    service._llm_manager = SimpleNamespace(
        get_utility_provider=lambda: no_health_provider
    )
    unchanged = {"answer": "a", "metadata": {}}
    assert service._attach_structured_reasoning_note(unchanged, None) is unchanged
    assert unchanged == {"answer": "a", "metadata": {}}


def test_service_without_provider_health_does_not_attach_note() -> None:
    service = PluggableResearchService.__new__(PluggableResearchService)
    service._llm_manager = SimpleNamespace(get_utility_provider=object)
    unchanged = {"answer": "a", "metadata": {}}

    assert service._attach_structured_reasoning_note(unchanged, None) is unchanged
    assert unchanged == {"answer": "a", "metadata": {}}
