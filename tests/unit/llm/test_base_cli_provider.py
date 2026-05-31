"""Tests for BaseCLIProvider double-wrap guard."""

import pytest

from chunkhound.providers.llm.base_cli_provider import BaseCLIProvider


class _StubCLIProvider(BaseCLIProvider):
    async def _run_cli_command(
        self, prompt: str, system=None, max_completion_tokens=None, timeout=None
    ) -> str:
        return ""  # empty → triggers RuntimeError in complete()

    def _get_provider_name(self) -> str:
        return "stub"


@pytest.mark.asyncio
async def test_internal_runtime_error_not_double_wrapped_complete():
    """RuntimeError from empty-response check must pass through unwrapped in complete()."""
    provider = _StubCLIProvider()

    with pytest.raises(RuntimeError) as exc:
        await provider.complete("test")

    msg = str(exc.value)
    assert "LLM returned empty response" in msg
    assert "LLM completion failed" not in msg


@pytest.mark.asyncio
async def test_internal_runtime_error_not_double_wrapped_complete_structured():
    """RuntimeError from empty-response check must pass through unwrapped in complete_structured()."""
    provider = _StubCLIProvider()

    with pytest.raises(RuntimeError) as exc:
        await provider.complete_structured("test", json_schema={"type": "object"})

    msg = str(exc.value)
    assert "LLM structured completion returned empty response" in msg
    assert "LLM structured completion failed" not in msg
