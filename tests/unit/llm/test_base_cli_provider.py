"""Tests for BaseCLIProvider double-wrap guard."""

import pytest

from chunkhound.interfaces.llm_provider import (
    PROVIDER_MANAGED_OUTPUT,
    OutputLimitCapability,
    OutputLimitMetadata,
)
from chunkhound.providers.llm.base_cli_provider import BaseCLIProvider


class _CapturingCLIProvider(BaseCLIProvider):
    def __init__(self, metadata: OutputLimitMetadata = OutputLimitMetadata()):
        super().__init__()
        self._metadata = metadata
        self.received_limits: list[int | None] = []

    @property
    def output_limit_metadata(self) -> OutputLimitMetadata:
        return self._metadata

    async def _run_cli_command(
        self, prompt: str, system=None, max_completion_tokens=None, timeout=None
    ) -> str:
        self.received_limits.append(max_completion_tokens)
        return '{"ok": true}'

    def _get_provider_name(self) -> str:
        return "capturing-stub"


class _StubCLIProvider(BaseCLIProvider):
    async def _run_cli_command(
        self, prompt: str, system=None, max_completion_tokens=None, timeout=None
    ) -> str:
        return ""  # empty → triggers RuntimeError in complete()

    def _get_provider_name(self) -> str:
        return "stub"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "expected_limit"),
    [
        (OutputLimitMetadata(), 8192),
        (
            OutputLimitMetadata(omission=OutputLimitCapability.SUPPORTED),
            None,
        ),
        (
            OutputLimitMetadata(
                declared_max_tokens=32768,
                declared_max_source="provider declaration",
            ),
            32768,
        ),
    ],
)
async def test_provider_managed_output_is_normalized_before_cli_boundary(
    metadata: OutputLimitMetadata, expected_limit: int | None
) -> None:
    provider = _CapturingCLIProvider(metadata)
    provider.configure_synthesis_output_limit_policy(
        output_limits_enabled=False, fallback_tokens=8192
    )

    response = await provider.complete(
        "test", max_completion_tokens=PROVIDER_MANAGED_OUTPUT
    )

    assert provider.received_limits == [expected_limit]
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_explicit_and_default_output_limits_remain_numeric() -> None:
    provider = _CapturingCLIProvider(
        OutputLimitMetadata(omission=OutputLimitCapability.SUPPORTED)
    )
    provider.configure_synthesis_output_limit_policy(
        output_limits_enabled=False, fallback_tokens=8192
    )

    await provider.complete("default")
    await provider.complete("explicit", max_completion_tokens=123)

    assert provider.received_limits == [4096, 123]


@pytest.mark.asyncio
async def test_structured_provider_managed_output_is_normalized() -> None:
    provider = _CapturingCLIProvider(
        OutputLimitMetadata(omission=OutputLimitCapability.SUPPORTED)
    )
    provider.configure_synthesis_output_limit_policy(
        output_limits_enabled=False, fallback_tokens=8192
    )

    result = await provider.complete_structured(
        "test",
        json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        max_completion_tokens=PROVIDER_MANAGED_OUTPUT,
    )

    assert result == {"ok": True}
    assert provider.received_limits == [None]


@pytest.mark.asyncio
async def test_internal_runtime_error_not_double_wrapped_complete():
    """Pass through an empty-response RuntimeError from complete unwrapped."""
    provider = _StubCLIProvider()

    with pytest.raises(RuntimeError) as exc:
        await provider.complete("test")

    msg = str(exc.value)
    assert "LLM returned empty response" in msg
    assert "LLM completion failed" not in msg


@pytest.mark.asyncio
async def test_internal_runtime_error_not_double_wrapped_complete_structured():
    """Pass through an empty-response RuntimeError from structured completion."""
    provider = _StubCLIProvider()

    with pytest.raises(RuntimeError) as exc:
        await provider.complete_structured("test", json_schema={"type": "object"})

    msg = str(exc.value)
    assert "LLM structured completion returned empty response" in msg
    assert "LLM structured completion failed" not in msg
