"""Tests for OpenCode CLI LLM provider."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.interfaces.llm_provider import LLMResponse
from chunkhound.providers.llm.opencode_cli_provider import OpenCodeCLIProvider


class TestOpenCodeCLIProvider:
    """Test cases for OpenCode CLI provider."""

    @pytest.fixture
    def provider(self):
        """Create a provider with a dummy model."""
        with patch.object(
            OpenCodeCLIProvider, "_opencode_available", return_value=True
        ):
            yield OpenCodeCLIProvider(
                model="test-provider/test-model",
                timeout=30,
                max_retries=2,
            )

    def test_provider_name(self, provider):
        """Test provider name property."""
        assert provider.name == "opencode-cli"

    def test_model_property(self, provider):
        """Test model property."""
        assert "/" in provider.model

    def test_validate_model_format_valid(self, provider):
        """Test valid model format validation."""
        provider._validate_model_format("openai/gpt-5")
        provider._validate_model_format("anthropic/claude-sonnet-4-5-20250929")
        provider._validate_model_format("groq/llama-3.1-8b-instant")

    def test_validate_model_format_invalid(self, provider):
        """Test invalid model format validation."""
        with pytest.raises(
            ValueError, match="Model must be in 'provider/model' format"
        ):
            provider._validate_model_format("gpt-5")

        with pytest.raises(ValueError, match="Provider cannot be empty"):
            provider._validate_model_format("/gpt-5")

        with pytest.raises(ValueError, match="opencode-cli requires a model"):
            provider._validate_model_format("")

    def test_validate_reasoning_effort_invalid(self, provider):
        """Test invalid reasoning effort raises ValueError."""
        with pytest.raises(ValueError, match="Invalid reasoning_effort"):
            provider._validate_reasoning_effort("invalid")

    def test_validate_reasoning_effort_case_normalization(self, provider):
        """Test reasoning effort is case-normalized."""
        assert provider._validate_reasoning_effort("HIGH") == "high"
        assert provider._validate_reasoning_effort("Minimal") == "minimal"

    @pytest.mark.asyncio
    async def test_complete_success(self, provider):
        """Test successful completion."""
        mock_response = "Test response"

        with patch.object(
            provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_response

            response = await provider.complete("Test prompt")

            assert isinstance(response, LLMResponse)
            assert response.content == "Test response"
            assert response.model == provider.model
            assert response.finish_reason == "stop"
            assert response.tokens_used > 0

    @pytest.mark.asyncio
    async def test_complete_with_system(self, provider):
        """Test completion passes through system prompt to CLI layer."""
        mock_response = "Response with system"

        with patch.object(
            provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_response

            await provider.complete("Test prompt", system="System message")

            mock_run.assert_called_once_with(
                "Test prompt", "System message", 4096, None
            )

    @pytest.mark.asyncio
    async def test_complete_empty_response(self, provider):
        """Test completion with empty response."""
        with patch.object(
            provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = ""

            with pytest.raises(RuntimeError, match="LLM returned empty response"):
                await provider.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_complete_structured_success(self, provider):
        """Test successful structured completion."""
        mock_response = '{"name": "test", "value": 42}'

        with patch.object(
            provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_response

            schema = {"type": "object", "properties": {"name": {"type": "string"}}}
            result = await provider.complete_structured("Test prompt", schema)

            assert result == {"name": "test", "value": 42}

    @pytest.mark.asyncio
    async def test_complete_structured_with_required_fields(self, provider):
        """Test structured completion with required field validation."""
        mock_response = '{"name": "test", "value": 42}'

        with patch.object(
            provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_response

            schema = {
                "type": "object",
                "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                "required": ["name", "value"],
            }
            result = await provider.complete_structured("Test prompt", schema)

            assert result == {"name": "test", "value": 42}

    @pytest.mark.asyncio
    async def test_complete_structured_missing_required_fields(self, provider):
        """Test structured completion with missing required fields."""
        mock_response = '{"name": "test"}'

        with patch.object(
            provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_response

            schema = {
                "type": "object",
                "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                "required": ["name", "value"],
            }

            with pytest.raises(
                RuntimeError,
                match=(
                    "LLM structured completion failed.*"
                    r"Missing required fields: \['value'\]"
                ),
            ):
                await provider.complete_structured("Test prompt", schema)

    @pytest.mark.asyncio
    async def test_complete_structured_invalid_json(self, provider):
        """Test structured completion with invalid JSON."""
        mock_response = "not valid json"

        with patch.object(
            provider, "_run_cli_command", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_response

            with pytest.raises(RuntimeError, match="Invalid JSON in structured output"):
                await provider.complete_structured("Test prompt", {})

    @pytest.mark.asyncio
    async def test_batch_complete(self, provider):
        """Test batch completion."""
        with patch.object(
            provider, "complete", new_callable=AsyncMock
        ) as mock_complete:
            mock_complete.return_value = LLMResponse(
                content="Response", tokens_used=10, model=provider.model
            )

            prompts = ["prompt1", "prompt2", "prompt3"]
            results = await provider.batch_complete(prompts)

            assert len(results) == 3
            assert all(isinstance(r, LLMResponse) for r in results)
            assert mock_complete.call_count == 3

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test successful health check."""
        with patch.object(
            provider, "complete", new_callable=AsyncMock
        ) as mock_complete:
            mock_complete.return_value = LLMResponse(
                content="OK", tokens_used=2, model=provider.model
            )

            result = await provider.health_check()

            assert result["status"] == "healthy"
            assert result["provider"] == "opencode-cli"
            assert result["model"] == provider.model
            assert result["test_response"] == "OK"

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider):
        """Test health check failure."""
        with patch.object(
            provider, "complete", new_callable=AsyncMock
        ) as mock_complete:
            mock_complete.side_effect = RuntimeError("CLI not found")

            result = await provider.health_check()

            assert result["status"] == "unhealthy"
            assert result["provider"] == "opencode-cli"
            assert "CLI not found" in result["error"]

    def test_get_usage_stats(self, provider):
        """Test usage statistics."""
        stats = provider.get_usage_stats()

        assert "requests_made" in stats
        assert "total_tokens_estimated" in stats
        assert "prompt_tokens_estimated" in stats
        assert "completion_tokens_estimated" in stats
        assert stats["requests_made"] == 0

    def test_get_synthesis_concurrency(self, provider):
        """Test synthesis concurrency."""
        concurrency = provider.get_synthesis_concurrency()
        assert concurrency == 3

    @pytest.mark.asyncio
    async def test_run_cli_command_max_retries_exceeded(self, provider):
        """Test CLI command max retries exceeded."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"Persistent CLI error")
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="OpenCode CLI command failed"):
                await provider._run_cli_command("Test prompt")

            assert mock_subprocess.call_count == provider._max_retries

    @pytest.mark.asyncio
    async def test_run_cli_command_timeout(self, provider):
        """Test CLI command timeout."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = MagicMock()
            mock_process.communicate = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )
            mock_process.wait = AsyncMock()
            mock_process.returncode = None
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="timed out"):
                await provider._run_cli_command("Test prompt")

            mock_process.kill.assert_called()
            mock_process.wait.assert_awaited()

    def test_init_invalid_model_format(self, provider):
        """Test provider creation with invalid model format fails fast."""
        with patch.object(
            OpenCodeCLIProvider, "_opencode_available", return_value=True
        ):
            with pytest.raises(
                ValueError, match="Model must be in 'provider/model' format"
            ):
                OpenCodeCLIProvider(model="invalid-model-format")

    @pytest.mark.asyncio
    async def test_run_cli_command_ndjson_error_event(self, provider):
        """Test NDJSON error detection when CLI exits 0 but returns error event."""
        error_event = json.dumps({
            "type": "error",
            "error": {"data": {"message": "model not available"}},
        })
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (error_event.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="model not available"):
                await provider._run_cli_command("Test prompt")

    @pytest.mark.asyncio
    async def test_run_cli_command_error_event_null_data(self, provider):
        """Test NDJSON error event with null data field does not crash."""
        error_event = json.dumps({
            "type": "error",
            "error": {"data": None, "message": "backend error"},
        })
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (error_event.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="backend error"):
                await provider._run_cli_command("Test prompt")

    @pytest.mark.asyncio
    async def test_run_cli_command_retry_on_ndjson_error_then_success(self, provider):
        """Test retry on NDJSON error event, then success."""
        error_event = json.dumps({
            "type": "error",
            "error": {"message": "temporary failure"},
        })
        text_event = json.dumps({
            "type": "text",
            "part": {"text": "Success response"},
        })
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process_fail = AsyncMock()
            mock_process_fail.communicate.return_value = (error_event.encode(), b"")
            mock_process_fail.returncode = 0

            mock_process_ok = AsyncMock()
            mock_process_ok.communicate.return_value = (text_event.encode(), b"")
            mock_process_ok.returncode = 0

            mock_subprocess.side_effect = [mock_process_fail, mock_process_ok]

            result = await provider._run_cli_command("Test prompt")

            assert result == "Success response"
            assert mock_subprocess.call_count == 2

    @pytest.mark.asyncio
    async def test_run_cli_command_no_text_content(self, provider):
        """Test error when NDJSON has no text events."""
        status_event = json.dumps({"type": "status", "data": "running"})
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (status_event.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="no text content"):
                await provider._run_cli_command("Test prompt")

    @pytest.mark.asyncio
    async def test_run_cli_command_multiple_text_events(self, provider):
        """Test that multiple text events are concatenated."""
        events = "\n".join([
            json.dumps({"type": "text", "part": {"text": "Hello "}}),
            json.dumps({"type": "text", "part": {"text": "World"}}),
        ])
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (events.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await provider._run_cli_command("Test prompt")

            assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_run_cli_command_text_then_error(self, provider):
        """Test that error event after text events raises and discards partial text."""
        events = "\n".join([
            json.dumps({"type": "text", "part": {"text": "Partial"}}),
            json.dumps({"type": "error", "error": {"message": "Failed"}}),
        ])
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (events.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="Failed"):
                await provider._run_cli_command("Test prompt")

    @pytest.mark.asyncio
    async def test_run_cli_command_string_error_event(self, provider):
        """Test NDJSON error detection when error field is a plain string."""
        error_event = json.dumps({"type": "error", "error": "rate limited"})
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (error_event.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="rate limited"):
                await provider._run_cli_command("Test prompt")

    @pytest.mark.asyncio
    async def test_run_cli_command_non_dict_json_event(self, provider):
        """Test that non-dict JSON lines are skipped."""
        events = "\n".join([
            json.dumps([1, 2, 3]),
            json.dumps("just a string"),
            json.dumps({"type": "text", "part": {"text": "OK"}}),
        ])
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (events.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await provider._run_cli_command("Test prompt")

            assert result == "OK"

    @pytest.mark.asyncio
    async def test_run_cli_command_malformed_json_interleaved(self, provider):
        """Test that malformed JSON lines are skipped and valid lines are processed."""
        events = "\n".join([
            json.dumps({"type": "text", "part": {"text": "Before"}}),
            '{"type": "text", "part": {", "text": "bad json"',
            json.dumps({"type": "text", "part": {"text": "After"}}),
        ])
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (events.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await provider._run_cli_command("Test prompt")

            assert result == "BeforeAfter"

    @pytest.mark.asyncio
    async def test_run_cli_command_null_part_text(self, provider):
        """Test that null or missing part.text is handled gracefully."""
        events = "\n".join([
            json.dumps({"type": "text", "part": {}}),
            json.dumps({"type": "text", "part": {"text": None}}),
            json.dumps({"type": "text", "part": {"text": "OK"}}),
        ])
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (events.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await provider._run_cli_command("Test prompt")

            assert result == "OK"

    @pytest.mark.asyncio
    async def test_run_cli_command_error_event_null_error(self, provider):
        """Test error event with null/missing error field falls back to Unknown error."""
        events = "\n".join([
            json.dumps({"type": "error", "error": None}),
        ])
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (events.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="Unknown error"):
                await provider._run_cli_command("Test prompt")

    @pytest.mark.asyncio
    async def test_run_cli_command_error_event_missing_error_key(self, provider):
        """Test error event without error key falls back to Unknown error."""
        events = "\n".join([
            json.dumps({"type": "error"}),
        ])
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (events.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="Unknown error"):
                await provider._run_cli_command("Test prompt")

    @pytest.mark.asyncio
    async def test_run_cli_command_nonzero_exit_with_text(self, provider):
        """Test that non-zero exit code with text content still returns text."""
        events = "\n".join([
            json.dumps({"type": "text", "part": {"text": "Partial output"}}),
        ])
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (events.encode(), b"Some stderr")
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            result = await provider._run_cli_command("Test prompt")

            assert result == "Partial output"

    @pytest.mark.asyncio
    async def test_run_cli_command_invalid_utf8_in_stdout(self, provider):
        """Test that invalid UTF-8 bytes in stdout are replaced and parsing continues."""
        valid_event = json.dumps({"type": "text", "part": {"text": "OK"}})
        # Place invalid bytes on their own line so the NDJSON parser skips
        # the unparseable line and still finds the valid event.
        stdout = b"\xff\xfe\n" + valid_event.encode("utf-8")
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (stdout, b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await provider._run_cli_command("Test prompt")

            assert result == "OK"

    @pytest.mark.asyncio
    async def test_run_cli_command_includes_variant_flag(self, provider):
        """Test that reasoning_effort maps to --variant in the CLI command."""
        with patch.object(
            OpenCodeCLIProvider, "_opencode_available", return_value=True
        ):
            provider = OpenCodeCLIProvider(
                model="test/provider",
                reasoning_effort="high",
            )
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError):
                await provider._run_cli_command("Test prompt")

            call_args = mock_subprocess.call_args[0]
            assert "--variant" in call_args
            variant_idx = call_args.index("--variant")
            assert call_args[variant_idx + 1] == "high"

    @pytest.mark.asyncio
    async def test_run_cli_command_no_variant_when_effort_none(self, provider):
        """Test that --variant is NOT in cmd when reasoning_effort is None."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError):
                await provider._run_cli_command("Test prompt")

            assert "--variant" not in mock_subprocess.call_args[0]

    @pytest.mark.asyncio
    async def test_run_cli_command_error_event_string_data(self, provider):
        """Test error event where error.data is a plain string, not a dict."""
        error_event = json.dumps({
            "type": "error",
            "error": {"data": "rate limited"},
        })
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (error_event.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="rate limited"):
                await provider._run_cli_command("Test prompt")

    @pytest.mark.asyncio
    async def test_run_cli_command_text_event_non_dict_part(self, provider):
        """Test that non-dict part in text event is safely skipped."""
        events = "\n".join([
            json.dumps({"type": "text", "part": "plain string, not a dict"}),
            json.dumps({"type": "text", "part": {"text": "OK"}}),
        ])
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (events.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await provider._run_cli_command("Test prompt")

            assert result == "OK"

    @pytest.mark.asyncio
    async def test_run_cli_command_system_prompt_format(self, provider):
        """Test that system prompt is concatenated with user prompt."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError):
                await provider._run_cli_command(
                    "Test prompt", system="System message"
                )

            call_args = mock_subprocess.call_args[0]
            # The last positional arg is the combined message
            message_arg = call_args[-1]
            assert message_arg == (
                "System Instructions:\nSystem message\n\nUser Request:\nTest prompt"
            )



    def test_format_json_flag_unsupported_markers(self, provider):
        """Test predicate recognizes all unsupported-flag marker strings."""
        markers = [
            "error: unexpected argument '--format'",
            "unknown option: --format",
            "unrecognized option --format json",
            "no such option: --format",
            "invalid option --format",
            "unknown flag --format",
        ]
        for err in markers:
            assert provider._format_json_flag_unsupported(err), (
                f"Should detect '{err}' as unsupported"
            )
        assert not provider._format_json_flag_unsupported("legitimate error")
        assert not provider._format_json_flag_unsupported("")

    @pytest.mark.asyncio
    async def test_run_cli_command_fallback_to_plain_text(self, provider):
        """Test fallback to plain text when --format json is unsupported."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Attempt 1: --format json rejected
            # Attempt 2: plain text succeeds
            mock_process_1 = AsyncMock()
            mock_process_1.communicate.return_value = (
                b"", b"error: unrecognized option --format json"
            )
            mock_process_1.returncode = 2

            mock_process_2 = AsyncMock()
            mock_process_2.communicate.return_value = (b"Hello", b"")
            mock_process_2.returncode = 0

            mock_subprocess.side_effect = [mock_process_1, mock_process_2]

            result = await provider._run_cli_command("Test prompt")
            assert result == "Hello"

            # Second call should NOT include --format json
            call_args = mock_subprocess.call_args[0]
            assert "--format" not in call_args

    @pytest.mark.asyncio
    async def test_run_cli_command_env_disable_json(self, monkeypatch, provider):
        """Test CHUNKHOUND_OPENCODE_JSON=0 skips --format json entirely."""
        monkeypatch.setenv("CHUNKHOUND_OPENCODE_JSON", "0")
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"Plain output", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await provider._run_cli_command("Test prompt")
            assert result == "Plain output"
            assert "--format" not in mock_subprocess.call_args[0]

    @pytest.mark.asyncio
    async def test_run_cli_command_fallback_both_modes_fail(self, provider):
        """Test that error is raised when both json and plain text modes fail."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process_1 = AsyncMock()
            mock_process_1.communicate.return_value = (
                b"", b"error: unrecognized option --format json"
            )
            mock_process_1.returncode = 2

            mock_process_2 = AsyncMock()
            mock_process_2.communicate.return_value = (b"", b"real failure")
            mock_process_2.returncode = 1

            mock_subprocess.side_effect = [mock_process_1, mock_process_2]

            with pytest.raises(RuntimeError, match="real failure"):
                await provider._run_cli_command("Test prompt")

    @pytest.mark.asyncio
    async def test_run_cli_command_fallback_model_on_timeout(self):
        """Test that fallback model is tried once when primary exhausts retries with timeout."""
        fallback_text_event = json.dumps(
            {"type": "text", "part": {"text": "Fallback response"}}
        )
        with patch.object(OpenCodeCLIProvider, "_opencode_available", return_value=True):
            provider = OpenCodeCLIProvider(
                model="test-provider/primary-model",
                fallback_model="test-provider/fallback-model",
                timeout=30,
                max_retries=1,
            )

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process_timeout = MagicMock()
            mock_process_timeout.communicate = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )
            mock_process_timeout.wait = AsyncMock()
            mock_process_timeout.returncode = None

            mock_process_fallback = AsyncMock()
            mock_process_fallback.communicate.return_value = (
                fallback_text_event.encode(), b""
            )
            mock_process_fallback.returncode = 0

            mock_subprocess.side_effect = [mock_process_timeout, mock_process_fallback]

            result = await provider._run_cli_command("Test prompt")

            assert result == "Fallback response"
            assert mock_subprocess.call_count == 2
            # Second call must use the fallback model slug
            second_call_args = mock_subprocess.call_args_list[1][0]
            model_idx = list(second_call_args).index("--model")
            assert second_call_args[model_idx + 1] == "test-provider/fallback-model"

    @pytest.mark.asyncio
    async def test_run_cli_command_fallback_not_tried_on_non_timeout_error(self):
        """Test that fallback model is NOT invoked on non-timeout failures."""
        with patch.object(OpenCodeCLIProvider, "_opencode_available", return_value=True):
            provider = OpenCodeCLIProvider(
                model="test-provider/primary-model",
                fallback_model="test-provider/fallback-model",
                timeout=30,
                max_retries=1,
            )

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"model not found")
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="model not found"):
                await provider._run_cli_command("Test prompt")

            assert mock_subprocess.call_count == 1
