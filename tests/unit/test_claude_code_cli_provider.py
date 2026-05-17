"""Tests for Claude Code CLI LLM provider."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.claude_model_resolution import (
    CLAUDE_HAIKU_SENTINEL,
    CLAUDE_OPUS_SENTINEL,
    CLAUDE_SONNET_SENTINEL,
)
from chunkhound.interfaces.llm_provider import LLMResponse
from chunkhound.providers.llm.claude_code_cli_provider import ClaudeCodeCLIProvider


@pytest.fixture
def provider():
    """Create a ClaudeCodeCLIProvider instance for testing."""
    return ClaudeCodeCLIProvider(
        model="claude-sonnet-4-5-20250929",
        timeout=60,
        max_retries=3,
    )


@pytest.fixture
def mock_subprocess():
    """Mock subprocess calls to avoid calling actual CLI."""
    with patch("asyncio.create_subprocess_exec") as mock:
        yield mock


class TestClaudeCodeCLIProvider:
    """Test suite for ClaudeCodeCLIProvider."""

    def test_provider_name(self, provider):
        """Test that provider name is correct."""
        assert provider.name == "claude-code-cli"

    def test_provider_model(self, provider):
        """Test that model name is stored correctly."""
        assert provider.model == "claude-sonnet-4-5-20250929"

    def test_default_model_is_haiku(self, monkeypatch):
        """Test that Claude CLI defaults to the shared Haiku sentinel."""
        monkeypatch.delenv("CHUNKHOUND_CLAUDE_DEFAULT_HAIKU_MODEL", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        provider = ClaudeCodeCLIProvider()
        assert provider.model == CLAUDE_HAIKU_SENTINEL

    def test_default_model_stores_sentinel_when_no_chunkhound_override(
        self, monkeypatch
    ):
        """Without a ChunkHound override, the CLI sentinel stays unresolved."""
        monkeypatch.delenv("CHUNKHOUND_CLAUDE_DEFAULT_HAIKU_MODEL", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setattr(
            "chunkhound.core.config.claude_model_resolution."
            "get_latest_available_haiku_model",
            lambda api_key=None: pytest.fail("discovery should not run"),
        )

        provider = ClaudeCodeCLIProvider()
        assert provider.model == CLAUDE_HAIKU_SENTINEL

    def test_chunkhound_env_override_resolves_before_cli(self, monkeypatch):
        """ChunkHound Claude overrides still pin the CLI model."""
        monkeypatch.setenv(
            "CHUNKHOUND_CLAUDE_DEFAULT_SONNET_MODEL",
            "claude-sonnet-4-6-20260217",
        )

        provider = ClaudeCodeCLIProvider(model=CLAUDE_SONNET_SENTINEL)

        assert provider.model == "claude-sonnet-4-6-20260217"

    @pytest.mark.asyncio
    async def test_complete_uses_cli_alias_for_sentinel(
        self, mock_subprocess
    ):
        """Sentinels reach the subprocess as bare Claude CLI aliases."""
        provider = ClaudeCodeCLIProvider(model=CLAUDE_SONNET_SENTINEL)
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Test response", b"")
        mock_subprocess.return_value = mock_process

        await provider.complete("Test prompt")

        cmd = mock_subprocess.call_args.args
        assert cmd[cmd.index("--model") + 1] == "sonnet"

    @pytest.mark.asyncio
    async def test_complete_uses_chunkhound_env_override_model(
        self, mock_subprocess, monkeypatch
    ):
        """ChunkHound Claude overrides must reach the CLI unchanged."""
        monkeypatch.setenv(
            "CHUNKHOUND_CLAUDE_DEFAULT_OPUS_MODEL",
            "claude-opus-4-7-20260416",
        )
        provider = ClaudeCodeCLIProvider(model=CLAUDE_OPUS_SENTINEL)
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Test response", b"")
        mock_subprocess.return_value = mock_process

        await provider.complete("Test prompt")

        cmd = mock_subprocess.call_args.args
        assert cmd[cmd.index("--model") + 1] == "claude-opus-4-7-20260416"

    @pytest.mark.asyncio
    async def test_complete_uses_explicit_full_model_name(
        self, mock_subprocess
    ):
        """Explicit Claude model names must reach the CLI unchanged."""
        provider = ClaudeCodeCLIProvider(model="claude-sonnet-4-5-20250929")
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Test response", b"")
        mock_subprocess.return_value = mock_process

        await provider.complete("Test prompt")

        cmd = mock_subprocess.call_args.args
        assert cmd[cmd.index("--model") + 1] == "claude-sonnet-4-5-20250929"

    @pytest.mark.asyncio
    async def test_complete_success(self, provider, mock_subprocess):
        """Test successful completion."""
        # Mock subprocess to return success
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Test response", b"")
        mock_subprocess.return_value = mock_process

        response = await provider.complete("Test prompt")

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.model == "claude-sonnet-4-5-20250929"
        assert response.tokens_used > 0
        assert response.finish_reason == "stop"

        # Verify usage tracking
        assert provider._requests_made == 1
        assert provider._estimated_tokens_used > 0

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self, provider, mock_subprocess):
        """Test completion with system prompt."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Response with system", b"")
        mock_subprocess.return_value = mock_process

        response = await provider.complete("User prompt", system="System instructions")

        assert response.content == "Response with system"
        # Verify that CLI was called with --append-system-prompt
        # (we'd need to inspect mock_subprocess.call_args for this)

    @pytest.mark.asyncio
    async def test_complete_timeout(self, provider, mock_subprocess):
        """Test that timeout is handled correctly and process is killed."""
        # Mock process that will timeout
        mock_process = AsyncMock()
        mock_process.returncode = None  # Process still running
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = MagicMock()  # Use MagicMock since kill() is not async
        mock_process.wait = AsyncMock()
        mock_subprocess.return_value = mock_process

        with pytest.raises(RuntimeError, match="CLI command timed out"):
            await provider.complete("Test prompt", timeout=1)

        # Verify process was killed and waited for (3 times due to retries)
        assert mock_process.kill.call_count == 3  # max_retries = 3
        assert mock_process.wait.call_count == 3

    @pytest.mark.asyncio
    async def test_complete_cli_error(self, provider, mock_subprocess):
        """Test handling of CLI errors."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"CLI error message")
        mock_subprocess.return_value = mock_process

        with pytest.raises(RuntimeError, match="CLI command failed"):
            await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_structured_success(self, provider, mock_subprocess):
        """Test successful structured completion with JSON."""
        json_response = {"result": "success", "data": [1, 2, 3]}
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            json.dumps(json_response).encode("utf-8"),
            b"",
        )
        mock_subprocess.return_value = mock_process

        schema = {
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "data": {"type": "array"},
            },
            "required": ["result", "data"],
        }

        response = await provider.complete_structured("Test prompt", schema)

        assert isinstance(response, dict)
        assert response["result"] == "success"
        assert response["data"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_complete_structured_with_markdown_code_block(
        self, provider, mock_subprocess
    ):
        """Test structured completion with JSON in markdown code block."""
        json_response = {"result": "success"}
        markdown_response = f"```json\n{json.dumps(json_response)}\n```"
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (markdown_response.encode("utf-8"), b"")
        mock_subprocess.return_value = mock_process

        schema = {"type": "object", "properties": {"result": {"type": "string"}}}

        response = await provider.complete_structured("Test prompt", schema)

        assert isinstance(response, dict)
        assert response["result"] == "success"

    @pytest.mark.asyncio
    async def test_complete_structured_invalid_json(self, provider, mock_subprocess):
        """Test handling of invalid JSON in structured completion."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Not valid JSON!", b"")
        mock_subprocess.return_value = mock_process

        schema = {"type": "object"}

        with pytest.raises(RuntimeError, match="Invalid JSON"):
            await provider.complete_structured("Test prompt", schema)

    @pytest.mark.asyncio
    async def test_complete_structured_missing_required_field(
        self, provider, mock_subprocess
    ):
        """Test validation of required fields in structured output."""
        json_response = {"data": [1, 2, 3]}  # Missing 'result'
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            json.dumps(json_response).encode("utf-8"),
            b"",
        )
        mock_subprocess.return_value = mock_process

        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}, "data": {"type": "array"}},
            "required": ["result", "data"],
        }

        with pytest.raises(RuntimeError, match="schema validation failed"):
            await provider.complete_structured("Test prompt", schema)

    @pytest.mark.asyncio
    async def test_complete_structured_rejects_additional_properties(
        self, provider, mock_subprocess
    ):
        """Test full JSON Schema validation in CLI fallback mode."""
        json_response = {"result": "success", "extra": True}
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            json.dumps(json_response).encode("utf-8"),
            b"",
        )
        mock_subprocess.return_value = mock_process

        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
            "additionalProperties": False,
        }

        with pytest.raises(RuntimeError, match="schema validation failed"):
            await provider.complete_structured("Test prompt", schema)

    @pytest.mark.asyncio
    async def test_complete_structured_rejects_nested_schema_violation(
        self, provider, mock_subprocess
    ):
        """Nested array/object constraints must be enforced."""
        json_response = {"items": [{"label": 7}]}
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            json.dumps(json_response).encode("utf-8"),
            b"",
        )
        mock_subprocess.return_value = mock_process

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}},
                        "required": ["label"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }

        with pytest.raises(RuntimeError, match="schema validation failed"):
            await provider.complete_structured("Test prompt", schema)

    @pytest.mark.asyncio
    async def test_batch_complete(self, provider, mock_subprocess):
        """Test batch completion (sequential calls)."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        # Return different responses for each call
        responses = [b"Response 1", b"Response 2", b"Response 3"]
        mock_process.communicate.side_effect = [(resp, b"") for resp in responses]
        mock_subprocess.return_value = mock_process

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = await provider.batch_complete(prompts)

        assert len(results) == 3
        assert all(isinstance(r, LLMResponse) for r in results)
        # Note: Due to mocking, all responses will be the same
        # In a real scenario with proper mocking, we'd verify each response

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, provider, mock_subprocess):
        """Test health check when CLI is available and working."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"OK", b"")
        mock_subprocess.return_value = mock_process

        result = await provider.health_check()

        assert result["status"] == "healthy"
        assert result["provider"] == "claude-code-cli"
        assert result["model"] == "claude-sonnet-4-5-20250929"
        assert "test_response" in result

    @pytest.mark.asyncio
    async def test_health_check_cli_not_found(self, provider, mock_subprocess):
        """Test health check when CLI is not available."""
        mock_subprocess.side_effect = FileNotFoundError("claude: command not found")

        result = await provider.health_check()

        assert result["status"] == "unhealthy"
        assert result["provider"] == "claude-code-cli"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_health_check_cli_fails(self, provider, mock_subprocess):
        """Test health check when CLI call fails."""
        mock_subprocess.side_effect = RuntimeError("CLI failed")

        result = await provider.health_check()

        assert result["status"] == "unhealthy"
        assert "error" in result

    def test_get_usage_stats(self, provider):
        """Test usage statistics retrieval."""
        # Initially zero
        stats = provider.get_usage_stats()
        assert stats["requests_made"] == 0
        assert stats["total_tokens_estimated"] == 0

        # Manually increment (normally done by complete methods)
        provider._requests_made = 5
        provider._estimated_tokens_used = 1000
        provider._estimated_prompt_tokens = 600
        provider._estimated_completion_tokens = 400

        stats = provider.get_usage_stats()
        assert stats["requests_made"] == 5
        assert stats["total_tokens_estimated"] == 1000
        assert stats["prompt_tokens_estimated"] == 600
        assert stats["completion_tokens_estimated"] == 400

    @pytest.mark.asyncio
    async def test_retry_logic(self, provider, mock_subprocess):
        """Test that retries work correctly on failure."""
        # First two attempts fail, third succeeds
        mock_process_fail = AsyncMock()
        mock_process_fail.returncode = 1
        mock_process_fail.communicate.return_value = (b"", b"Error")

        mock_process_success = AsyncMock()
        mock_process_success.returncode = 0
        mock_process_success.communicate.return_value = (b"Success", b"")

        mock_subprocess.side_effect = [
            mock_process_fail,
            mock_process_fail,
            mock_process_success,
        ]

        response = await provider.complete("Test prompt")

        assert response.content == "Success"
        # Should have tried 3 times
        assert mock_subprocess.call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, provider, mock_subprocess):
        """Test that error is raised after max retries."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"Persistent error")
        mock_subprocess.return_value = mock_process

        with pytest.raises(RuntimeError, match="CLI command failed"):
            await provider.complete("Test prompt")

        # Should have tried max_retries (3) times
        assert mock_subprocess.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_with_already_terminated_process(
        self,
        provider,
        mock_subprocess,
    ):
        """Test that process is not killed if already terminated."""
        # Mock process that terminates before we can kill it
        mock_process = AsyncMock()
        mock_process.returncode = 0  # Process already terminated
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()
        mock_subprocess.return_value = mock_process

        with pytest.raises(RuntimeError, match="CLI command timed out"):
            await provider.complete("Test prompt", timeout=1)

        # Verify process was NOT killed (since returncode was set)
        # With retries, this will be called 3 times but each time returncode is 0
        mock_process.kill.assert_not_called()
        mock_process.wait.assert_not_called()
