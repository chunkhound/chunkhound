import asyncio
import os
import subprocess
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chunkhound.interfaces.llm_provider import LLMResponse
from chunkhound.providers.llm.antigravity_llm_provider import AntigravityLLMProvider
from chunkhound.providers.llm.antigravity_cli_provider import AntigravityCLIProvider


@pytest.fixture
def mock_antigravity_agent():
    """Mock the Agent class inside the provider module."""
    with patch("chunkhound.providers.llm.antigravity_llm_provider.Agent") as mock:
        yield mock


@pytest.fixture
def mock_subprocess():
    """Mock subprocess exec calls for CLI provider."""
    with patch("asyncio.create_subprocess_exec") as mock:
        yield mock


def _make_agent_mock(mock_agent_class: MagicMock) -> MagicMock:
    """Return mock agent instance entered via async context manager."""
    return mock_agent_class.return_value.__aenter__.return_value


def _make_sdk_response(
    text: str,
    thoughts: str = "",
    prompt_tokens: int = 10,
    candidates_tokens: int = 20,
    thoughts_tokens: int = 5,
) -> tuple[MagicMock, MagicMock]:
    """Helper to create nested Agent responses with usage metadata."""
    response = MagicMock()
    # Mock text() as async function
    response.text = AsyncMock(return_value=text)
    # Mock thoughts property
    response.thoughts = thoughts

    # Mock structured_output as async function
    response.structured_output = AsyncMock(return_value={"key": "val"})

    # Nest usage metadata
    total_usage = MagicMock()
    total_usage.total_token_count = prompt_tokens + candidates_tokens + thoughts_tokens
    total_usage.prompt_token_count = prompt_tokens
    total_usage.candidates_token_count = candidates_tokens
    total_usage.thoughts_token_count = thoughts_tokens

    conversation = MagicMock()
    conversation.total_usage = total_usage

    return response, conversation


# --- SDK Provider Tests ---

@pytest.mark.asyncio
async def test_sdk_complete_success(mock_antigravity_agent):
    provider = AntigravityLLMProvider(api_key="test-api-key", model="gemini-3.5-flash")
    
    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response("Hello from Antigravity!", thoughts="Let me think...")
    
    # Mock complete/chat
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    result = await provider.complete("Test prompt", system="Test system")

    assert isinstance(result, LLMResponse)
    assert result.content == "Hello from Antigravity!"
    assert result.tokens_used == 35  # 10 + 20 + 5
    assert result.model == "gemini-3.5-flash"
    
    # Verify mock call details
    mock_antigravity_agent.assert_called_once()
    config_passed = mock_antigravity_agent.call_args[1].get("config")
    assert config_passed.model == "gemini-3.5-flash"
    assert config_passed.system_instructions == "Test system"


@pytest.mark.asyncio
async def test_sdk_complete_error_propagation(mock_antigravity_agent):
    provider = AntigravityLLMProvider(api_key="test-api-key", model="gemini-3.5-flash")
    
    agent_mock = _make_agent_mock(mock_antigravity_agent)
    # Simulate API connection failure
    agent_mock.chat = AsyncMock(side_effect=Exception("API connection refused"))

    with pytest.raises(RuntimeError, match="API connection refused"):
        await provider.complete("hello")


@pytest.mark.asyncio
async def test_sdk_structured_success(mock_antigravity_agent):
    provider = AntigravityLLMProvider(api_key="test-api-key", model="gemini-3.5-flash")
    
    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response("{}", thoughts="Structured output...")
    
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    # Simple JSON schema
    schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string"}
        },
        "required": ["key"]
    }

    result = await provider.complete_structured("Structured prompt", json_schema=schema)

    assert isinstance(result, dict)
    assert result == {"key": "val"}
    
    # Verify mock call details
    mock_antigravity_agent.assert_called_once()
    config_passed = mock_antigravity_agent.call_args[1].get("config")
    
    # Verify pydantic class or dict is passed as response_schema
    assert config_passed.response_schema is not None

