import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.interfaces.llm_provider import LLMResponse
from chunkhound.providers.llm.antigravity_cli_provider import AntigravityCLIProvider
from chunkhound.providers.llm.antigravity_llm_provider import AntigravityLLMProvider


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
    resp_mock, conv_mock = _make_sdk_response(
        "Hello from Antigravity!", thoughts="Let me think..."
    )

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
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    }

    result = await provider.complete_structured("Structured prompt", json_schema=schema)

    assert isinstance(result, dict)
    assert result == {"key": "val"}

    # Verify mock call details
    mock_antigravity_agent.assert_called_once()
    config_passed = mock_antigravity_agent.call_args[1].get("config")

    # Verify pydantic class is compiled and passed as response_schema
    assert isinstance(config_passed.response_schema, str)
    assert "DynamicResponseModel" in config_passed.response_schema


@pytest.mark.asyncio
async def test_sdk_complete_token_estimation_includes_thoughts(mock_antigravity_agent):
    provider = AntigravityLLMProvider(api_key="test-api-key", model="gemini-3.5-flash")

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response(
        "Hello from Antigravity!", thoughts="Thinking deeply..."
    )

    # Simulate failed token usage retrieval by setting conversation to None
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = None

    result = await provider.complete("Test prompt", system="Test system")

    # Estimated tokens: len(prompt) // 4 + len(system) // 4
    # + len(content + thoughts) // 4
    # prompt = "Test prompt" (11 chars) -> 2 tokens
    # system = "Test system" (11 chars) -> 2 tokens
    # content = "Hello from Antigravity!" (23 chars)
    # thoughts = "Thinking deeply..." (18 chars)
    # content + thoughts = 41 chars -> 10 tokens
    # Total estimated = 2 + 2 + 10 = 14 tokens.
    assert result.tokens_used == 14


# --- CLI Provider Tests ---


@pytest.mark.asyncio
async def test_cli_complete_success(mock_subprocess):
    provider = AntigravityCLIProvider(model="gemini-3.5-flash")

    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"Hello from CLI!", b"")
    mock_subprocess.return_value = mock_process

    # Set some test env vars to verify scrubbing
    with patch.dict(
        os.environ,
        {
            "CHUNKHOUND_TEST": "1",
            "GOOGLE_APPLICATION_CREDENTIALS": "abc",
            "PATH": "/usr/bin",
        },
    ):
        result = await provider.complete("CLI prompt")

    assert isinstance(result, LLMResponse)
    assert result.content == "Hello from CLI!"
    assert result.model == "gemini-3.5-flash"
    assert result.tokens_used > 0

    # Assert subprocess call arguments
    mock_subprocess.assert_called_once()
    cmd_args = mock_subprocess.call_args.args
    assert "agy" in cmd_args or "antigravity" in cmd_args
    assert "--print" in cmd_args
    assert "--sandbox" in cmd_args
    assert "--model" in cmd_args
    assert cmd_args[cmd_args.index("--model") + 1] == "gemini-3.5-flash"

    # Assert subprocess run CWD and env isolation
    call_kwargs = mock_subprocess.call_args.kwargs
    assert call_kwargs.get("cwd") == tempfile.gettempdir()

    env_passed = call_kwargs.get("env")
    assert env_passed is not None
    assert "CHUNKHOUND_TEST" not in env_passed
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env_passed
    assert env_passed.get("PATH") == "/usr/bin"


@pytest.mark.asyncio
async def test_cli_complete_failure(mock_subprocess):
    provider = AntigravityCLIProvider(model="gemini-3.5-flash")

    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate.return_value = (
        b"",
        b"Command not found or permission denied",
    )
    mock_subprocess.return_value = mock_process

    with pytest.raises(RuntimeError, match="Command not found or permission denied"):
        await provider.complete("CLI prompt")


@pytest.mark.asyncio
async def test_cli_binary_fallback(mock_subprocess):
    with patch("shutil.which") as mock_which:

        def side_effect(cmd):
            if cmd == "agy":
                return None
            if cmd == "antigravity":
                return "/usr/local/bin/antigravity"
            return None

        mock_which.side_effect = side_effect

        provider = AntigravityCLIProvider(model="gemini-3.5-flash")
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Hello from fallback CLI!", b"")
        mock_subprocess.return_value = mock_process

        result = await provider.complete("CLI prompt")

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from fallback CLI!"
        mock_subprocess.assert_called_once()
        cmd_args = mock_subprocess.call_args.args
        assert cmd_args[0] == "antigravity"


@pytest.mark.asyncio
async def test_cli_timeout_cleanup(mock_subprocess):
    provider = AntigravityCLIProvider(model="gemini-3.5-flash", timeout=2)

    mock_process = AsyncMock()
    mock_process.kill = MagicMock()
    mock_process.wait = AsyncMock()
    mock_subprocess.return_value = mock_process

    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
        with pytest.raises(RuntimeError, match="timed out after 2s"):
            await provider.complete("CLI prompt")

    mock_process.kill.assert_called_once()
    mock_process.wait.assert_called_once()


@pytest.mark.asyncio
async def test_cli_contract_probe():
    """Verify that if the real binary (agy or antigravity) is on PATH, it supports --print and --sandbox."""
    import shutil
    binary = "agy" if shutil.which("agy") else ("antigravity" if shutil.which("antigravity") else None)
    if not binary:
        pytest.skip("Neither 'agy' nor 'antigravity' binary found in PATH, skipping contract probe.")

    process = await asyncio.create_subprocess_exec(
        binary,
        "--help",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    help_output = (stdout + stderr).decode("utf-8", errors="ignore")
    assert "--print" in help_output or "-p" in help_output, f"Binary '{binary}' help does not document '--print'"
    assert "--sandbox" in help_output, f"Binary '{binary}' help does not document '--sandbox'"


@pytest.mark.asyncio
async def test_sdk_timeout(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key", model="gemini-3.5-flash", timeout=10
    )

    _make_agent_mock(mock_antigravity_agent)

    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()) as mock_wait_for:
        with pytest.raises(RuntimeError, match="Antigravity SDK call failed:"):
            await provider.complete("Test prompt", timeout=5)

        mock_wait_for.assert_called_once()
        assert mock_wait_for.call_args.kwargs["timeout"] == 5


@pytest.mark.asyncio
async def test_sdk_security_constraints(mock_antigravity_agent):
    provider = AntigravityLLMProvider(api_key="test-api-key", model="gemini-3.5-flash")

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response("Hello!")

    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    await provider.complete("Test prompt")

    mock_antigravity_agent.assert_called_once()
    config_passed = mock_antigravity_agent.call_args[1].get("config")

    assert config_passed.capabilities is not None
    assert not config_passed.capabilities.enable_subagents
    enabled_tools = config_passed.capabilities.enabled_tools
    assert enabled_tools is not None
    from google.antigravity.types import BuiltinTools

    assert BuiltinTools.LIST_DIR in enabled_tools
    assert BuiltinTools.CREATE_FILE not in enabled_tools

    assert config_passed.workspaces == [os.getcwd()]
    assert len(config_passed.policies) > 0


@pytest.mark.asyncio
async def test_sdk_structured_validation(mock_antigravity_agent):
    provider = AntigravityLLMProvider(api_key="test-api-key", model="gemini-3.5-flash")

    agent_mock = _make_agent_mock(mock_antigravity_agent)

    response = MagicMock()
    response.structured_output = AsyncMock(return_value={"confidence": 2.0})
    agent_mock.chat = AsyncMock(return_value=response)
    agent_mock.conversation = MagicMock()

    schema = {
        "type": "object",
        "properties": {
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["confidence"],
    }

    with pytest.raises(
        RuntimeError,
        match="validation failed|exceeds|maximum|greater than|is greater than",
    ):
        await provider.complete_structured("Structured prompt", json_schema=schema)


def test_cli_synthesis_concurrency():
    provider = AntigravityCLIProvider(model="gemini-3.5-flash")
    assert provider.get_synthesis_concurrency() == 1


@pytest.mark.asyncio
async def test_sdk_structured_fallback_validation_failure(mock_antigravity_agent):
    provider = AntigravityLLMProvider(api_key="test-api-key", model="gemini-3.5-flash")

    agent_mock = _make_agent_mock(mock_antigravity_agent)

    response = MagicMock()
    # Remove structured_output attribute to trigger fallback path
    if hasattr(response, "structured_output"):
        delattr(response, "structured_output")

    # Mock text() returning schema-invalid JSON
    response.text = AsyncMock(return_value='{"confidence": 2.0}')
    response.thoughts = ""
    agent_mock.chat = AsyncMock(return_value=response)
    agent_mock.conversation = MagicMock()

    schema = {
        "type": "object",
        "properties": {
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["confidence"],
    }

    # Verify that the schema validation error propagates out as a RuntimeError
    with pytest.raises(
        RuntimeError,
        match="validation failed|exceeds|maximum|greater than|is greater than",
    ):
        await provider.complete_structured("Structured prompt", json_schema=schema)


@pytest.mark.asyncio
async def test_sdk_usage_stats_tracking(mock_antigravity_agent):
    provider = AntigravityLLMProvider(api_key="test-api-key", model="gemini-3.5-flash")

    # Check initial stats
    initial_stats = provider.get_usage_stats()
    assert initial_stats == {
        "requests_made": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response(
        "Hello!", prompt_tokens=50, candidates_tokens=100, thoughts_tokens=25
    )
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    # Complete request
    await provider.complete("Prompt")

    # Check updated stats
    updated_stats = provider.get_usage_stats()
    assert updated_stats == {
        "requests_made": 1,
        "total_tokens": 175,
        "prompt_tokens": 50,
        "completion_tokens": 125,  # candidates (100) + thoughts (25)
    }


@pytest.mark.asyncio
async def test_sdk_health_check_healthy(mock_antigravity_agent):
    provider = AntigravityLLMProvider(api_key="test-api-key", model="gemini-3.5-flash")

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response("pong")
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    health = await provider.health_check()
    assert health == {
        "status": "healthy",
        "provider": "antigravity-sdk",
        "model": "gemini-3.5-flash",
        "test_response": "pong",
    }


@pytest.mark.asyncio
async def test_sdk_health_check_unhealthy(mock_antigravity_agent):
    provider = AntigravityLLMProvider(api_key="test-api-key", model="gemini-3.5-flash")

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    agent_mock.chat = AsyncMock(side_effect=Exception("Connection failed"))

    health = await provider.health_check()
    assert health["status"] == "unhealthy"
    assert health["provider"] == "antigravity-sdk"
    assert "Connection failed" in health["error"]


@pytest.mark.asyncio
async def test_sdk_max_completion_tokens_warning(mock_antigravity_agent):
    provider = AntigravityLLMProvider(api_key="test-api-key", model="gemini-3.5-flash")

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response("Hello!")
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    with patch(
        "chunkhound.providers.llm.antigravity_llm_provider.logger.warning"
    ) as mock_warn:
        await provider.complete("Prompt", max_completion_tokens=100)
        mock_warn.assert_called_once()
        assert "max_completion_tokens" in mock_warn.call_args[0][0]


def test_sdk_compile_schema_to_pydantic_constraints():
    provider = AntigravityLLMProvider(api_key="test-api-key", model="gemini-3.5-flash")
    schema = {
        "type": "object",
        "properties": {
            "num_val": {
                "type": "number",
                "minimum": 1.0,
                "maximum": 5.0,
                "exclusiveMinimum": 0.5,
                "exclusiveMaximum": 5.5,
            },
            "str_val": {
                "type": "string",
                "minLength": 2,
                "maxLength": 10,
                "pattern": "^[a-z]+$",
            },
            "arr_val": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 3,
            },
            "enum_val": {"type": "string", "enum": ["red", "blue"]},
            "enum_arr": {
                "type": "array",
                "items": {"type": "string", "enum": ["apple", "banana"]},
            },
        },
        "required": ["num_val", "enum_val"],
        "additionalProperties": False,
    }

    model_cls = provider._compile_schema_to_pydantic(schema)

    # Assert model constraints by trying to validate valid and invalid inputs
    from pydantic import ValidationError

    # Valid model instance
    valid_instance = model_cls(
        num_val=3.0,
        str_val="abc",
        arr_val=[1, 2],
        enum_val="red",
        enum_arr=["apple", "banana"],
    )
    assert valid_instance.num_val == 3.0
    assert valid_instance.str_val == "abc"
    assert valid_instance.arr_val == [1, 2]
    assert valid_instance.enum_val == "red"
    assert valid_instance.enum_arr == ["apple", "banana"]

    # Invalid numeric bounds (minimum / exclusiveMinimum / maximum / exclusiveMaximum)
    with pytest.raises(ValidationError):
        model_cls(num_val=0.5, enum_val="red")  # fails exclusiveMinimum
    with pytest.raises(ValidationError):
        model_cls(num_val=5.5, enum_val="red")  # fails exclusiveMaximum
    with pytest.raises(ValidationError):
        model_cls(num_val=6.0, enum_val="red")  # fails maximum

    # Invalid string length
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, str_val="a", enum_val="red")
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, str_val="abcdefghiklmn", enum_val="red")

    # Invalid string pattern
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, str_val="abc1", enum_val="red")

    # Invalid array items size
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, arr_val=[], enum_val="red")
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, arr_val=[1, 2, 3, 4], enum_val="red")

    # Invalid enum value
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, enum_val="green")

    # Invalid array enum value
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, enum_val="red", enum_arr=["orange"])

    # Forbidden extra properties (additionalProperties=False)
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, enum_val="red", extra_field="forbidden")
