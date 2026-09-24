"""Contract tests: the LLM providers' per-call chokepoints must record
exactly one analytics provider-call per invocation, success or failure,
reusing the SDK response's own token-usage fields. Uses the real
chunkhound_native.AnalyticsRecorder (no mocking) with a mocked SDK client.
"""

import glob
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.core.analytics.recorder import (
    build_recorder,
    end_command,
    start_command,
)
from chunkhound.core.config.analytics_config import AnalyticsConfig
from chunkhound.providers.llm.anthropic_llm_provider import AnthropicLLMProvider
from chunkhound.providers.llm.openai_compatible_provider import OpenAICompatibleProvider
from chunkhound.providers.llm.openai_llm_provider import OpenAILLMProvider


def _read_events(buffer_dir: Path) -> list[dict]:
    events = []
    for path in glob.glob(str(buffer_dir / "buffer-*.jsonl")):
        for line in Path(path).read_text().splitlines():
            events.append(json.loads(line))
    return events


@pytest.fixture
def open_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Opens a command (setting the analytics ContextVar). Yields
    (recorder, handle, buffer_dir) -- the test calls end_command itself
    once it's made whatever provider calls it wants to observe, since the
    event isn't serialized to the buffer until end_command runs."""
    buffer_dir = tmp_path / "analytics"
    monkeypatch.setattr("chunkhound.core.analytics.recorder._ANALYTICS_DIR", buffer_dir)
    config = AnalyticsConfig(
        enabled=True, flush_interval_seconds=999999, flush_batch_size=999999
    )
    recorder = build_recorder(config, tmp_path)
    handle = start_command(recorder, "research", "mcp", {})
    yield recorder, handle, buffer_dir


@pytest.mark.asyncio
async def test_anthropic_success_records_a_provider_call(open_command) -> None:
    recorder, handle, buffer_dir = open_command
    provider = AnthropicLLMProvider(api_key="test-key")
    response = MagicMock()
    response.content = [MagicMock(type="text", text="ok")]
    response.stop_reason = "end_turn"
    response.usage = MagicMock(input_tokens=100, output_tokens=20)
    provider._client = MagicMock()
    provider._client.messages.create = AsyncMock(return_value=response)

    await provider.complete("hello")
    end_command(recorder, handle, True)

    llm = _read_events(buffer_dir)[-1]["providers"]["llm"][0]
    assert llm["provider"] == "anthropic"
    assert llm["calls"] == 1
    assert llm["fails"] == 0
    assert llm["input_tokens"] == 100
    assert llm["output_tokens"] == 20


@pytest.mark.asyncio
async def test_anthropic_failure_records_a_failed_provider_call(open_command) -> None:
    recorder, handle, buffer_dir = open_command
    provider = AnthropicLLMProvider(api_key="test-key")
    provider._client = MagicMock()
    provider._client.messages.create = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(Exception):
        await provider.complete("hello")
    end_command(recorder, handle, False)

    llm = _read_events(buffer_dir)[-1]["providers"]["llm"][0]
    assert llm["calls"] == 1
    assert llm["fails"] == 1
    assert llm["error_types"] == {"RuntimeError": 1}


@pytest.mark.asyncio
async def test_openai_compatible_success_records_a_provider_call(open_command) -> None:
    recorder, handle, buffer_dir = open_command
    provider = OpenAICompatibleProvider(
        api_key="test-key", model="gpt-4", provider_name="openai"
    )
    response = MagicMock()
    response.choices = [
        MagicMock(message=MagicMock(content="ok"), finish_reason="stop")
    ]
    response.usage = MagicMock(prompt_tokens=50, completion_tokens=10, total_tokens=60)
    provider._client = MagicMock()
    provider._client.chat.completions.create = AsyncMock(return_value=response)

    await provider.complete("hello")
    end_command(recorder, handle, True)

    llm = _read_events(buffer_dir)[-1]["providers"]["llm"][0]
    assert llm["provider"] == "openai"
    assert llm["calls"] == 1
    assert llm["fails"] == 0
    assert llm["input_tokens"] == 50
    assert llm["output_tokens"] == 10


@pytest.mark.asyncio
async def test_openai_compatible_failure_records_a_failed_provider_call(
    open_command,
) -> None:
    recorder, handle, buffer_dir = open_command
    provider = OpenAICompatibleProvider(api_key="test-key", model="gpt-4")
    provider._client = MagicMock()
    provider._client.chat.completions.create = AsyncMock(side_effect=ValueError("boom"))

    with pytest.raises(Exception):
        await provider.complete("hello")
    end_command(recorder, handle, False)

    llm = _read_events(buffer_dir)[-1]["providers"]["llm"][0]
    assert llm["calls"] == 1
    assert llm["fails"] == 1
    assert llm["error_types"] == {"ValueError": 1}


@pytest.mark.asyncio
async def test_openai_responses_api_success_records_a_provider_call(
    open_command,
) -> None:
    """Reasoning models (o1-pro, o3-pro, gpt-5.1, ...) route through the
    Responses API instead of Chat Completions -- must go through the same
    instrumented chokepoint as the Chat Completions path."""
    recorder, handle, buffer_dir = open_command
    provider = OpenAILLMProvider(api_key="test-key", model="o1-pro")
    response = MagicMock()
    response.output = [
        MagicMock(
            type="message",
            content=[MagicMock(type="output_text", text="ok")],
        )
    ]
    response.status = "completed"
    response.usage = MagicMock(input_tokens=70, output_tokens=15, total_tokens=85)
    provider._client = MagicMock()
    provider._client.responses.create = AsyncMock(return_value=response)

    await provider.complete("hello")
    end_command(recorder, handle, True)

    llm = _read_events(buffer_dir)[-1]["providers"]["llm"][0]
    assert llm["provider"] == "openai"
    assert llm["calls"] == 1
    assert llm["fails"] == 0
    assert llm["input_tokens"] == 70
    assert llm["output_tokens"] == 15


@pytest.mark.asyncio
async def test_openai_responses_api_failure_records_a_failed_provider_call(
    open_command,
) -> None:
    recorder, handle, buffer_dir = open_command
    provider = OpenAILLMProvider(api_key="test-key", model="o1-pro")
    provider._client = MagicMock()
    provider._client.responses.create = AsyncMock(side_effect=ValueError("boom"))

    with pytest.raises(Exception):
        await provider.complete("hello")
    end_command(recorder, handle, False)

    llm = _read_events(buffer_dir)[-1]["providers"]["llm"][0]
    assert llm["calls"] == 1
    assert llm["fails"] == 1
    assert llm["error_types"] == {"ValueError": 1}
