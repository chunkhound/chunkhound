"""OpenAI-compatible provider-managed output-cap wire contracts."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from openai import AsyncOpenAI

from chunkhound.interfaces.llm_provider import (
    PROVIDER_MANAGED_OUTPUT,
    OutputLimitCapability,
)
from chunkhound.llm_manager import LLMManager
from chunkhound.providers.llm.openai_compatible_provider import OpenAICompatibleProvider
from tests.fixtures.openai_compatible_server import (
    ChatCompletionScript,
    OpenAICompatibleTestServer,
)


@contextmanager
def _server(marker: str) -> Iterator[OpenAICompatibleTestServer]:
    script = ChatCompletionScript(name=marker, marker=marker, content="complete answer")
    with OpenAICompatibleTestServer([script]) as server:
        yield server
        server.assert_all_scripts_consumed()


def _manager_provider(
    name: str, *, base_url: str | None = None
) -> OpenAICompatibleProvider:
    manager = object.__new__(LLMManager)
    config = {"model": "contract-test-model", "api_key": "sk-test"}
    if base_url is not None:
        config["base_url"] = base_url
    return manager._create_openai_compatible_provider(name, config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "cap_field"),
    [("deepseek", "max_tokens"), ("grok", "max_completion_tokens")],
)
async def test_canonical_builtin_provider_managed_request_omits_cap(
    name: str,
    cap_field: str,
) -> None:
    marker = f"canonical-{name}-omit"
    with _server(marker) as server:
        provider = _manager_provider(name)
        assert (
            provider.output_limit_metadata.omission is OutputLimitCapability.SUPPORTED
        )

        await provider._client.close()
        provider._client = AsyncOpenAI(
            api_key="sk-test",
            base_url=server.base_url,
            max_retries=0,
        )
        provider.configure_synthesis_output_limit_policy(
            output_limits_enabled=False,
            fallback_tokens=64_000,
        )
        try:
            await provider.complete(
                marker,
                max_completion_tokens=PROVIDER_MANAGED_OUTPUT,
            )
        finally:
            await provider._client.close()

        body = server.requests[0]["json"]
        assert cap_field not in body
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "cap_field"),
    [("deepseek", "max_tokens"), ("grok", "max_completion_tokens")],
)
async def test_custom_builtin_endpoint_uses_fallback_cap(
    name: str,
    cap_field: str,
) -> None:
    marker = f"custom-{name}-fallback"
    with _server(marker) as server:
        provider = _manager_provider(name, base_url=server.base_url)
        assert provider.output_limit_metadata.omission is OutputLimitCapability.UNKNOWN
        provider.configure_synthesis_output_limit_policy(
            output_limits_enabled=False,
            fallback_tokens=64_003,
        )
        try:
            await provider.complete(
                marker,
                max_completion_tokens=PROVIDER_MANAGED_OUTPUT,
            )
        finally:
            await provider._client.close()

        body = server.requests[0]["json"]
        assert body[cap_field] == 64_003


@pytest.mark.asyncio
async def test_generic_compatible_endpoint_uses_fallback_cap() -> None:
    marker = "generic-compatible-fallback"
    with _server(marker) as server:
        provider = OpenAICompatibleProvider(
            api_key="sk-test",
            model="contract-test-model",
            base_url=server.base_url,
            max_retries=0,
        )
        assert provider.output_limit_metadata.omission is OutputLimitCapability.UNKNOWN
        provider.configure_synthesis_output_limit_policy(
            output_limits_enabled=False,
            fallback_tokens=64_007,
        )
        try:
            await provider.complete(
                marker,
                max_completion_tokens=PROVIDER_MANAGED_OUTPUT,
            )
        finally:
            await provider._client.close()

        body = server.requests[0]["json"]
        assert body["max_completion_tokens"] == 64_007
