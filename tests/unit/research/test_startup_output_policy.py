"""Research startup output reports the selected synthesis request-limit policy."""

import asyncio
from io import StringIO
from typing import Any
from unittest.mock import patch

import pytest

from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay
from chunkhound.interfaces.llm_provider import (
    OutputLimitCapability,
    OutputLimitMetadata,
    OutputLimitPolicy,
)
from chunkhound.providers.llm.openai_llm_provider import OpenAILLMProvider
from chunkhound.services.research.v1.pluggable_research_service import (
    PluggableResearchService,
)


class _StopAfterStartupError(Exception):
    """Sentinel preventing the contract test from running the research pipeline."""


class _Provider:
    def __init__(self, policy: OutputLimitPolicy) -> None:
        self.synthesis_output_limit_policy = policy


class _Manager:
    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def get_synthesis_provider(self) -> Any:
        return self._provider


async def _stop_after_startup(*_: Any, **__: Any) -> list[dict[str, Any]]:
    raise _StopAfterStartupError


async def _render_startup(manager: _Manager) -> str:
    output = StringIO()
    progress = TreeProgressDisplay(output=output)
    progress.start()

    service = object.__new__(PluggableResearchService)
    service._llm_manager = manager
    service._progress = progress
    service._progress_lock = asyncio.Lock()
    service._unified_search = _stop_after_startup

    try:
        with pytest.raises(_StopAfterStartupError):
            await service.deep_research("policy contract")
    finally:
        progress.stop()

    return output.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (
            OutputLimitPolicy(
                output_limits_enabled=False,
                fallback_tokens=64_000,
                metadata=OutputLimitMetadata(omission=OutputLimitCapability.SUPPORTED),
            ),
            "Max depth: 1; synthesis request limits: provider-managed (cap omitted)",
        ),
        (
            OutputLimitPolicy(
                output_limits_enabled=False,
                fallback_tokens=64_000,
                metadata=OutputLimitMetadata(
                    omission=OutputLimitCapability.UNKNOWN,
                    declared_max_tokens=64_000,
                    declared_max_source="https://provider.example/output-limits",
                ),
            ),
            "Max depth: 1; synthesis request limits: provider-managed "
            "(provider-declared cap: 64,000 tokens)",
        ),
        (
            OutputLimitPolicy(
                output_limits_enabled=False,
                fallback_tokens=64_000,
                metadata=OutputLimitMetadata(omission=OutputLimitCapability.UNKNOWN),
            ),
            "Max depth: 1; synthesis request limits: provider-managed "
            "(fallback cap: 64,000 tokens)",
        ),
        (
            OutputLimitPolicy(
                output_limits_enabled=True,
                fallback_tokens=64_000,
                metadata=OutputLimitMetadata(),
            ),
            "Max depth: 1; synthesis request limits: legacy numeric "
            "(30,000-token single/reduce cap; computed per-map caps)",
        ),
    ],
    ids=["omission", "declaration", "fallback", "legacy"],
)
async def test_deep_research_renders_resolved_startup_output_policy(
    policy: OutputLimitPolicy,
    expected: str,
) -> None:
    rendered = await _render_startup(_Manager(_Provider(policy)))

    info_lines = [line for line in rendered.splitlines() if "Max depth:" in line]
    assert len(info_lines) == 1
    assert info_lines[0].endswith(expected)
    assert "output budget" not in rendered

    all_variants = {
        "Max depth: 1; synthesis request limits: provider-managed (cap omitted)",
        "Max depth: 1; synthesis request limits: provider-managed "
        "(provider-declared cap: 64,000 tokens)",
        "Max depth: 1; synthesis request limits: provider-managed "
        "(fallback cap: 64,000 tokens)",
        "Max depth: 1; synthesis request limits: legacy numeric "
        "(30,000-token single/reduce cap; computed per-map caps)",
    }
    for other in all_variants - {expected}:
        assert other not in rendered


@pytest.mark.asyncio
async def test_custom_openai_environment_reports_fallback_policy(
    monkeypatch: pytest.MonkeyPatch,
    clean_environment,
) -> None:
    """Actual startup reporting uses the real custom-route provider policy."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    with patch(
        "chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI"
    ):
        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-5")
    provider.configure_synthesis_output_limit_policy(
        output_limits_enabled=False,
        fallback_tokens=64_000,
    )

    rendered = await _render_startup(_Manager(provider))

    assert (
        "Max depth: 1; synthesis request limits: provider-managed "
        "(fallback cap: 64,000 tokens)" in rendered
    )
    assert "provider-managed (cap omitted)" not in rendered
