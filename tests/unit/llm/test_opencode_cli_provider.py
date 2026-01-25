"""Tests for OpenCode CLI LLM provider."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.interfaces.llm_provider import LLMResponse
from chunkhound.providers.llm.opencode_cli_provider import OpenCodeCLIProvider


class TestOpenCodeCLIProvider:
    """Test cases for OpenCode CLI provider."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch.object(
            OpenCodeCLIProvider, "_opencode_available", return_value=True
        ):
            self.provider = OpenCodeCLIProvider(
                model="openai/gpt-5-nano",
                timeout=30,
                max_retries=2,
            )

    def test_provider_name(self):
        """Test provider name property."""
        assert self.provider.name == "opencode-cli"

    def test_model_property(self):
        """Test model property."""
        assert self.provider.model == "openai/gpt-5-nano"

    def test_validate_model_format_valid(self):
        """Test valid model format validation."""
        # Should not raise exception
        self.provider._validate_model_format("openai/gpt-5")
        self.provider._validate_model_format("anthropic/claude-sonnet-4-5-20250929")
        self.provider._validate_model_format("groq/llama-3.1-8b-instant")

    def test_validate_model_format_invalid(self):
        """Test invalid model format validation."""
        with pytest.raises(
            ValueError, match="Model must be in 'provider/model' format"
        ):
            self.provider._validate_model_format("gpt-5")

        with pytest.raises(ValueError, match="Provider cannot be empty"):
            self.provider._validate_model_format("/gpt-5")

    @pytest.mark.asyncio
    async def test_complete_success(self):
        """Test successful completion."""
        mock_response = "Test response"

        with patch.object(
            self.provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_response

            response = await self.provider.complete("Test prompt")

            assert isinstance(response, LLMResponse)
            assert response.content == "Test response"
            assert response.model == "openai/gpt-5-nano"
            assert response.finish_reason == "stop"
            assert response.tokens_used > 0

    @pytest.mark.asyncio
    async def test_complete_with_system(self):
        """Test completion with system prompt."""
        mock_response = "Response with system"

        with patch.object(
            self.provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_response

            await self.provider.complete("Test prompt", system="System message")

            mock_run.assert_called_once_with(
                "Test prompt",
                "System message",
                4096,  # max_completion_tokens
                None,  # timeout
            )

    @pytest.mark.asyncio
    async def test_complete_empty_response(self):
        """Test completion with empty response."""
        with patch.object(
            self.provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = ""

            with pytest.raises(RuntimeError, match="LLM returned empty response"):
                await self.provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_structured_success(self):
        """Test successful structured completion."""
        mock_response = '{"name": "test", "value": 42}'

        with patch.object(
            self.provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_response

            schema = {"type": "object", "properties": {"name": {"type": "string"}}}
            result = await self.provider.complete_structured("Test prompt", schema)

            assert result == {"name": "test", "value": 42}

    @pytest.mark.asyncio
    async def test_complete_structured_with_required_fields(self):
        """Test structured completion with required field validation."""
        mock_response = '{"name": "test", "value": 42}'

        with patch.object(
            self.provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_response

            schema = {
                "type": "object",
                "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                "required": ["name", "value"],
            }
            result = await self.provider.complete_structured("Test prompt", schema)

            assert result == {"name": "test", "value": 42}

    @pytest.mark.asyncio
    async def test_complete_structured_missing_required_fields(self):
        """Test structured completion with missing required fields."""
        mock_response = '{"name": "test"}'

        with patch.object(
            self.provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_response

            schema = {
                "type": "object",
                "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                "required": ["name", "value"],
            }

            with pytest.raises(
                RuntimeError,
                match="LLM structured completion failed.*Missing required fields: \\['value'\\]",
            ):
                await self.provider.complete_structured("Test prompt", schema)

    @pytest.mark.asyncio
    async def test_complete_structured_invalid_json(self):
        """Test structured completion with invalid JSON."""
        mock_response = "not valid json"

        with patch.object(
            self.provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_response

            with pytest.raises(RuntimeError, match="Invalid JSON in structured output"):
                await self.provider.complete_structured("Test prompt", {})

    @pytest.mark.asyncio
    async def test_batch_complete(self):
        """Test batch completion."""
        with patch.object(
            self.provider, "complete", new_callable=AsyncMock
        ) as mock_complete:
            mock_complete.return_value = LLMResponse(
                content="Response", tokens_used=10, model="openai/gpt-5-nano"
            )

            prompts = ["prompt1", "prompt2", "prompt3"]
            results = await self.provider.batch_complete(prompts)

            assert len(results) == 3
            assert all(isinstance(r, LLMResponse) for r in results)
            assert mock_complete.call_count == 3

    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "This is a test string for token estimation"
        estimated = self.provider.estimate_tokens(text)
        assert estimated == len(text) // self.provider.TOKEN_CHARS_RATIO

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        with patch.object(
            self.provider, "complete", new_callable=AsyncMock
        ) as mock_complete:
            mock_complete.return_value = LLMResponse(
                content="OK", tokens_used=2, model="openai/gpt-5-nano"
            )

            result = await self.provider.health_check()

            assert result["status"] == "healthy"
            assert result["provider"] == "opencode-cli"
            assert result["model"] == "openai/gpt-5-nano"
            assert result["test_response"] == "OK"

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure."""
        with patch.object(
            self.provider, "complete", new_callable=AsyncMock
        ) as mock_complete:
            mock_complete.side_effect = RuntimeError("CLI not found")

            result = await self.provider.health_check()

            assert result["status"] == "unhealthy"
            assert result["provider"] == "opencode-cli"
            assert "CLI not found" in result["error"]

    def test_get_usage_stats(self):
        """Test usage statistics."""
        stats = self.provider.get_usage_stats()

        assert "requests_made" in stats
        assert "total_tokens_estimated" in stats
        assert "prompt_tokens_estimated" in stats
        assert "completion_tokens_estimated" in stats
        assert stats["requests_made"] == 0  # Initially zero

    def test_get_synthesis_concurrency(self):
        """Test synthesis concurrency."""
        concurrency = self.provider.get_synthesis_concurrency()
        assert concurrency == 3

    @pytest.mark.asyncio
    async def test_run_cli_command_success(self):
        """Test successful CLI command execution."""
        mock_response = "Test response"

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (mock_response.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await self.provider._run_cli_command("Test prompt")

            assert result == mock_response.strip()
            mock_subprocess.assert_called_once()

            # Verify command structure - args are unpacked when calling create_subprocess_exec
            call_args = mock_subprocess.call_args[0]
            # Since *cmd is used, the cmd list elements are passed as separate arguments
            # So call_args[0] should contain the unpacked cmd elements
            assert "opencode" in call_args
            assert "run" in call_args
            assert "--model" in call_args
            assert "openai/gpt-5-nano" in call_args
            # Prompt should be passed via stdin, not as a command argument
            assert "Test prompt" not in call_args
            # Should NOT contain --format since we use default text format
            assert "--format" not in call_args

            # Verify prompt was passed via stdin
            communicate_call = mock_process.communicate.call_args
            assert communicate_call[1]["input"] == b"Test prompt"

    @pytest.mark.asyncio
    async def test_run_cli_command_with_system(self):
        """Test CLI command with system prompt."""
        mock_response = "Test response"

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (mock_response.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            await self.provider._run_cli_command("Test prompt", system="System message")

            # Verify system prompt is combined with user prompt and passed via stdin
            call_args = mock_subprocess.call_args[0]
            # Should NOT contain --system flag
            assert "--system" not in call_args
            # Prompt should be passed via stdin, not as a command argument
            assert "System message\nTest prompt" not in call_args
            # Verify basic command structure
            assert "opencode" in call_args
            assert "run" in call_args
            assert "--model" in call_args
            assert "openai/gpt-5-nano" in call_args

            # Verify combined prompt was passed via stdin
            communicate_call = mock_process.communicate.call_args
            assert communicate_call[1]["input"] == b"System message\nTest prompt"

    @pytest.mark.asyncio
    async def test_run_cli_command_without_system(self):
        """Test CLI command without system prompt."""
        mock_response = "Test response"

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (mock_response.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            await self.provider._run_cli_command("Test prompt")

            # Verify only user prompt is passed via stdin
            call_args = mock_subprocess.call_args[0]
            # Prompt should be passed via stdin, not as a command argument
            assert "Test prompt" not in call_args
            # Should NOT contain system prompt
            assert "System message" not in call_args
            # Should NOT contain --system flag
            assert "--system" not in call_args

            # Verify prompt was passed via stdin
            communicate_call = mock_process.communicate.call_args
            assert communicate_call[1]["input"] == b"Test prompt"

    @pytest.mark.asyncio
    async def test_run_cli_command_retry_on_failure(self):
        """Test CLI command retry on failure."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process_fail = AsyncMock()
            mock_process_fail.communicate.return_value = (b"", b"CLI error")
            mock_process_fail.returncode = 1

            mock_process_success = AsyncMock()
            mock_json_response = json.dumps({"message": {"content": "Success"}})
            mock_process_success.communicate.return_value = (
                mock_json_response.encode(),
                b"",
            )
            mock_process_success.returncode = 0

            # First call fails, second succeeds
            mock_subprocess.side_effect = [mock_process_fail, mock_process_success]

            result = await self.provider._run_cli_command("Test prompt")

            assert result == mock_json_response.strip()
            assert mock_subprocess.call_count == 2

    @pytest.mark.asyncio
    async def test_run_cli_command_max_retries_exceeded(self):
        """Test CLI command max retries exceeded."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"Persistent CLI error")
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="OpenCode CLI command failed"):
                await self.provider._run_cli_command("Test prompt")

            # Should be called max_retries times
            assert mock_subprocess.call_count == self.provider._max_retries

    @pytest.mark.asyncio
    async def test_run_cli_command_timeout(self):
        """Test CLI command timeout."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = MagicMock()
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            mock_process.returncode = None
            mock_process.wait = AsyncMock()  # wait is async, kill is sync
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="timed out"):
                await self.provider._run_cli_command("Test prompt")

            # Process should be killed on timeout
            mock_process.kill.assert_called()
            mock_process.wait.assert_called()

    @pytest.mark.asyncio
    async def test_run_cli_command_invalid_model_format(self):
        """Test CLI command with invalid model format."""
        provider = OpenCodeCLIProvider(model="invalid-model-format")

        with pytest.raises(
            ValueError, match="Model must be in 'provider/model' format"
        ):
            await provider._run_cli_command("Test prompt")
