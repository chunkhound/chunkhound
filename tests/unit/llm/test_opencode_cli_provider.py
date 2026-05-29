"""Tests for OpenCode CLI LLM provider."""

import asyncio
import json
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.interfaces.llm_provider import LLMResponse
from chunkhound.providers.llm.opencode_cli_provider import OpenCodeCLIProvider


def test_default_timeout():
    """Default timeout resolves to DEFAULT_LLM_TIMEOUT."""
    p = OpenCodeCLIProvider(model="openai/gpt-5-nano")
    assert p.timeout == DEFAULT_LLM_TIMEOUT


class TestOpenCodeCLIProvider:
    """Test cases for OpenCode CLI provider."""

    @pytest.fixture(autouse=True)
    def posix_signal_constants(self):
        """Make POSIX-only signal constants available for mocked Unix tests."""
        with patch("signal.SIGKILL", 9, create=True):
            yield

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

        with pytest.raises(ValueError, match="Model cannot be empty"):
            provider._validate_model_format("openai/")

        with pytest.raises(ValueError, match="Model cannot be empty"):
            provider._validate_model_format("openai/   ")

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
                match=("LLM structured completion failed.*schema validation failed"),
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
    async def test_run_single_attempt_timeout_uses_killpg(self, provider):
        """Timeout cleanup kills the captured process group, not just the parent."""
        with (
            patch("sys.platform", "linux"),
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("os.killpg", create=True) as mock_killpg,
        ):
            mock_process = AsyncMock()
            mock_process.pid = 4321
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            mock_process.wait = AsyncMock()
            mock_process.kill = MagicMock()
            mock_process.returncode = None
            mock_subprocess.return_value = mock_process

            result = await provider._run_single_attempt(
                "Test prompt",
                1,
                model="test-provider/test-model",
                use_json=False,
            )

            assert result.action == "timeout"
            assert mock_subprocess.call_args.kwargs["start_new_session"] is True
            killpg_calls = [
                (args[0], args[1]) for args, _kwargs in mock_killpg.call_args_list
            ]
            assert (4321, signal.SIGTERM) in killpg_calls
            assert (4321, signal.SIGKILL) in killpg_calls
            mock_process.kill.assert_not_called()
            mock_process.wait.assert_awaited()

    @pytest.mark.asyncio
    async def test_run_single_attempt_group_escalation_uses_captured_pgid(
        self, provider
    ):
        """Cleanup uses the PGID captured at spawn rather than a later PID lookup."""
        with (
            patch("sys.platform", "linux"),
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("os.killpg", create=True) as mock_killpg,
            patch(
                "os.getpgid",
                side_effect=AssertionError("late PGID lookup"),
                create=True,
            ),
        ):
            mock_process = AsyncMock()
            mock_process.pid = 4321

            async def timeout_after_pid_change(*_args, **_kwargs):
                mock_process.pid = 9999
                raise asyncio.TimeoutError()

            mock_process.communicate.side_effect = timeout_after_pid_change
            mock_process.wait = AsyncMock()
            mock_process.kill = MagicMock()
            mock_process.returncode = None
            mock_subprocess.return_value = mock_process

            result = await provider._run_single_attempt(
                "Test prompt",
                1,
                model="test-provider/test-model",
                use_json=False,
            )

            assert result.action == "timeout"
            killpg_calls = [
                (args[0], args[1]) for args, _kwargs in mock_killpg.call_args_list
            ]
            assert (4321, signal.SIGTERM) in killpg_calls
            assert (4321, signal.SIGKILL) in killpg_calls
            assert all(pgid == 4321 for pgid, _sig in killpg_calls)

    @pytest.mark.asyncio
    async def test_run_single_attempt_parent_exit_still_killpg_captured_pgid(
        self, provider
    ):
        """SIGKILL is still attempted for descendants after the parent exits."""
        with (
            patch("sys.platform", "linux"),
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("os.killpg", create=True) as mock_killpg,
        ):
            mock_process = AsyncMock()
            mock_process.pid = 4321
            mock_process.communicate.side_effect = asyncio.TimeoutError()

            async def wait_and_mark_parent_exit():
                mock_process.returncode = 0
                return 0

            mock_process.wait = AsyncMock(side_effect=wait_and_mark_parent_exit)
            mock_process.kill = MagicMock()
            mock_process.returncode = None
            mock_subprocess.return_value = mock_process

            result = await provider._run_single_attempt(
                "Test prompt",
                1,
                model="test-provider/test-model",
                use_json=False,
            )

            assert result.action == "timeout"
            killpg_calls = [
                (args[0], args[1]) for args, _kwargs in mock_killpg.call_args_list
            ]
            assert (4321, signal.SIGTERM) in killpg_calls
            assert (4321, signal.SIGKILL) in killpg_calls

    @pytest.mark.asyncio
    async def test_run_single_attempt_cancelled_cleans_up(self, provider):
        """Cancelled tasks propagate only after process-group cleanup runs."""
        with (
            patch("sys.platform", "linux"),
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("os.killpg", create=True) as mock_killpg,
        ):
            mock_process = AsyncMock()
            mock_process.pid = 4321
            mock_process.communicate.side_effect = asyncio.CancelledError()
            mock_process.wait = AsyncMock()
            mock_process.kill = MagicMock()
            mock_process.returncode = None
            mock_subprocess.return_value = mock_process

            with pytest.raises(asyncio.CancelledError):
                await provider._run_single_attempt(
                    "Test prompt",
                    30,
                    model="test-provider/test-model",
                    use_json=False,
                )

            mock_killpg.assert_any_call(4321, signal.SIGTERM)
            mock_killpg.assert_any_call(4321, signal.SIGKILL)
            mock_process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_single_attempt_exception_cleans_up(self, provider):
        """Unexpected subprocess-body exceptions trigger process-tree cleanup."""
        with (
            patch("sys.platform", "linux"),
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("os.killpg", create=True) as mock_killpg,
        ):
            mock_process = AsyncMock()
            mock_process.pid = 4321
            mock_process.communicate.side_effect = OSError("boom")
            mock_process.wait = AsyncMock()
            mock_process.kill = MagicMock()
            mock_process.returncode = None
            mock_subprocess.return_value = mock_process

            result = await provider._run_single_attempt(
                "Test prompt",
                30,
                model="test-provider/test-model",
                use_json=False,
            )

            assert result.action == "failure"
            assert "boom" in str(result.error)
            mock_killpg.assert_any_call(4321, signal.SIGTERM)
            mock_killpg.assert_any_call(4321, signal.SIGKILL)
            mock_process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_single_attempt_windows_uses_taskkill(self, provider):
        """Windows cleanup routes to taskkill /T instead of Unix killpg."""
        with (
            patch("sys.platform", "win32"),
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("subprocess.run") as mock_run,
            patch("subprocess.CREATE_NEW_PROCESS_GROUP", 0x00000200, create=True),
            patch("os.killpg", create=True) as mock_killpg,
        ):
            mock_process = AsyncMock()
            mock_process.pid = 2468
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            mock_process.wait = AsyncMock()
            mock_process.kill = MagicMock()
            mock_process.returncode = None
            mock_subprocess.return_value = mock_process

            result = await provider._run_single_attempt(
                "Test prompt",
                1,
                model="test-provider/test-model",
                use_json=False,
            )

            assert result.action == "timeout"
            assert "start_new_session" not in mock_subprocess.call_args.kwargs
            assert mock_subprocess.call_args.kwargs["creationflags"] == 0x00000200
            mock_run.assert_called_once_with(
                ["taskkill", "/T", "/PID", "2468", "/F"],
                check=False,
                timeout=10,
            )
            mock_killpg.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cli_command_timeout(self, provider):
        """Test CLI command timeout."""
        with (
            patch("sys.platform", "linux"),
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("os.killpg", create=True) as mock_killpg,
        ):
            mock_process = AsyncMock()
            mock_process.pid = 4321
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            mock_process.wait = AsyncMock()
            mock_process.kill = MagicMock()
            mock_process.returncode = None
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="timed out") as exc:
                await provider._run_cli_command("Test prompt")

            msg = str(exc.value)
            assert "model=test-provider/test-model" in msg
            assert "Test prompt" not in msg
            mock_killpg.assert_any_call(4321, signal.SIGTERM)
            mock_killpg.assert_any_call(4321, signal.SIGKILL)
            mock_process.kill.assert_not_called()
            mock_process.wait.assert_awaited()

    @pytest.mark.asyncio
    async def test_run_cli_command_timeout_ignores_stale_process_lookup_error(
        self, provider
    ):
        """Stale process cleanup must not mask the timeout error."""
        with (
            patch("sys.platform", "linux"),
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("os.killpg", side_effect=ProcessLookupError(), create=True),
        ):
            mock_process = AsyncMock()
            mock_process.pid = 4321
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            mock_process.wait = AsyncMock()
            mock_process.returncode = None
            mock_process.kill = MagicMock()
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="timed out"):
                await provider._run_cli_command("Test prompt")

            mock_process.wait.assert_awaited()
            mock_process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cli_command_generic_failure_ignores_stale_process_lookup_error(
        self, provider
    ):
        """Unexpected failures must survive stale process cleanup."""
        with (
            patch("sys.platform", "linux"),
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("os.killpg", side_effect=ProcessLookupError(), create=True),
        ):
            mock_process = AsyncMock()
            mock_process.pid = 4321
            mock_process.communicate.side_effect = ValueError("boom")
            mock_process.wait = AsyncMock()
            mock_process.returncode = None
            mock_process.kill = MagicMock()
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="OpenCode CLI command failed: boom"):
                await provider._run_cli_command("Test prompt")

            assert mock_process.wait.await_count == provider._max_retries * 2
            mock_process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cli_command_failure_redacts_prompt_from_error(self, provider):
        """Error text must not leak prompt payloads from argv construction."""
        secret_prompt = "SECRET PROMPT PAYLOAD"
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"Persistent CLI error")
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            with pytest.raises(
                RuntimeError, match="OpenCode CLI command failed"
            ) as exc:
                await provider._run_cli_command(secret_prompt)

            msg = str(exc.value)
            assert "command=opencode run" in msg
            assert "model=test-provider/test-model" in msg
            assert secret_prompt not in msg

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
        error_event = json.dumps(
            {
                "type": "error",
                "error": {"data": {"message": "model not available"}},
            }
        )
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
        error_event = json.dumps(
            {
                "type": "error",
                "error": {"data": None, "message": "backend error"},
            }
        )
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
        error_event = json.dumps(
            {
                "type": "error",
                "error": {"message": "temporary failure"},
            }
        )
        text_event = json.dumps(
            {
                "type": "text",
                "part": {"text": "Success response"},
            }
        )
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

    @staticmethod
    def _assert_json_then_plain(mock_subprocess):
        """Verify first call used --format and second did not."""
        first_args = mock_subprocess.call_args_list[0][0]
        second_args = mock_subprocess.call_args_list[1][0]
        assert "--format" in first_args
        assert "--format" not in second_args

    @pytest.mark.asyncio
    async def test_run_cli_command_no_text_in_json_fails(self, provider):
        """When NDJSON has no text events, provider retries in plain-text mode."""
        status_event = json.dumps({"type": "status", "data": "running"})
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # First call (JSON): NDJSON with no text events
            # Second call (plain text): empty stdout
            mock_first = AsyncMock()
            mock_first.communicate.return_value = (status_event.encode(), b"")
            mock_first.returncode = 0
            mock_second = AsyncMock()
            mock_second.communicate.return_value = (b"", b"")
            mock_second.returncode = 0
            mock_subprocess.side_effect = [mock_first, mock_second]

            with pytest.raises(RuntimeError, match="empty output"):
                await provider._run_cli_command("Test prompt")
            # Two calls: JSON mode (no text) → plain text fallback (no text)
            assert mock_subprocess.call_count == provider._max_retries
            self._assert_json_then_plain(mock_subprocess)

    @pytest.mark.asyncio
    async def test_run_cli_command_no_text_in_json_recovers(self, provider):
        """When JSON has no text events, plain text fallback recovers."""
        status_event = json.dumps({"type": "status", "data": "running"})
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # First call (JSON): NDJSON with no text events
            mock_first = AsyncMock()
            mock_first.communicate.return_value = (status_event.encode(), b"")
            mock_first.returncode = 0
            # Second call (plain text): actual output
            mock_second = AsyncMock()
            mock_second.communicate.return_value = (b"Success response", b"")
            mock_second.returncode = 0
            mock_subprocess.side_effect = [mock_first, mock_second]

            result = await provider._run_cli_command("Test prompt")

            assert result == "Success response"
            assert mock_subprocess.call_count == provider._max_retries
            self._assert_json_then_plain(mock_subprocess)

    @pytest.mark.asyncio
    async def test_run_cli_command_json_fallback_respects_single_attempt_budget(self):
        """max_retries=1 leaves no plain-text budget after a JSON probe."""
        status_event = json.dumps({"type": "status", "data": "running"})
        with patch.object(
            OpenCodeCLIProvider, "_opencode_available", return_value=True
        ):
            provider = OpenCodeCLIProvider(
                model="test-provider/test-model",
                timeout=30,
                max_retries=1,
            )

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_first = AsyncMock()
            mock_first.communicate.return_value = (status_event.encode(), b"")
            mock_first.returncode = 0
            mock_subprocess.return_value = mock_first

            with pytest.raises(
                RuntimeError, match="exhausted retry budget before plain-text fallback"
            ):
                await provider._run_cli_command("Test prompt")

            assert mock_subprocess.call_count == 1
            first_args = mock_subprocess.call_args_list[0][0]
            assert "--format" in first_args

    @pytest.mark.asyncio
    async def test_run_cli_command_json_fallback_preserves_remaining_attempt_budget(
        self,
    ):
        """JSON fallback should only spend the primary model's remaining attempts."""
        status_event = json.dumps({"type": "status", "data": "running"})
        with patch.object(
            OpenCodeCLIProvider, "_opencode_available", return_value=True
        ):
            provider = OpenCodeCLIProvider(
                model="test-provider/test-model",
                timeout=30,
                max_retries=3,
            )

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_json = AsyncMock()
            mock_json.communicate.return_value = (status_event.encode(), b"")
            mock_json.returncode = 0

            mock_plain_fail = AsyncMock()
            mock_plain_fail.communicate.return_value = (b"", b"")
            mock_plain_fail.returncode = 0

            mock_plain_success = AsyncMock()
            mock_plain_success.communicate.return_value = (b"Success response", b"")
            mock_plain_success.returncode = 0

            mock_subprocess.side_effect = [
                mock_json,
                mock_plain_fail,
                mock_plain_success,
            ]

            result = await provider._run_cli_command("Test prompt")

            assert result == "Success response"
            assert mock_subprocess.call_count == 3
            first_args = mock_subprocess.call_args_list[0][0]
            second_args = mock_subprocess.call_args_list[1][0]
            third_args = mock_subprocess.call_args_list[2][0]
            assert "--format" in first_args
            assert "--format" not in second_args
            assert "--format" not in third_args

    @pytest.mark.asyncio
    async def test_run_cli_command_multiple_text_events(self, provider):
        """Test that multiple text events are concatenated."""
        events = "\n".join(
            [
                json.dumps({"type": "text", "part": {"text": "Hello "}}),
                json.dumps({"type": "text", "part": {"text": "World"}}),
            ]
        )
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
        events = "\n".join(
            [
                json.dumps({"type": "text", "part": {"text": "Partial"}}),
                json.dumps({"type": "error", "error": {"message": "Failed"}}),
            ]
        )
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
        events = "\n".join(
            [
                json.dumps([1, 2, 3]),
                json.dumps("just a string"),
                json.dumps({"type": "text", "part": {"text": "OK"}}),
            ]
        )
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
        events = "\n".join(
            [
                json.dumps({"type": "text", "part": {"text": "Before"}}),
                '{"type": "text", "part": {", "text": "bad json"',
                json.dumps({"type": "text", "part": {"text": "After"}}),
            ]
        )
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
        events = "\n".join(
            [
                json.dumps({"type": "text", "part": {}}),
                json.dumps({"type": "text", "part": {"text": None}}),
                json.dumps({"type": "text", "part": {"text": "OK"}}),
            ]
        )
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (events.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await provider._run_cli_command("Test prompt")

            assert result == "OK"

    @pytest.mark.asyncio
    async def test_run_cli_command_error_event_null_error(self, provider):
        """Test null/missing error field falls back to Unknown error."""
        events = "\n".join(
            [
                json.dumps({"type": "error", "error": None}),
            ]
        )
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
        events = "\n".join(
            [
                json.dumps({"type": "error"}),
            ]
        )
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (events.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="Unknown error"):
                await provider._run_cli_command("Test prompt")

    @pytest.mark.asyncio
    async def test_run_cli_command_nonzero_exit_with_text(self, provider):
        """Test that non-zero exit code with text content still raises."""
        events = "\n".join(
            [
                json.dumps({"type": "text", "part": {"text": "Partial output"}}),
            ]
        )
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (events.encode(), b"Some stderr")
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="Some stderr"):
                await provider._run_cli_command("Test prompt")

            assert mock_subprocess.call_count == provider._max_retries

    @pytest.mark.asyncio
    async def test_run_cli_command_nonzero_exit_with_text_logs_discard(
        self,
        provider,
    ):
        """Non-zero exit with streamed text must log the discard count at DEBUG."""
        from loguru import logger as _loguru_logger

        captured: list[str] = []
        sink_id = _loguru_logger.add(
            lambda msg: captured.append(msg),
            level="DEBUG",
            format="{message}",
        )
        try:
            events = "\n".join(
                [
                    json.dumps({"type": "text", "part": {"text": "Partial"}}),
                ]
            )
            with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (events.encode(), b"stderr")
                mock_process.returncode = 1
                mock_subprocess.return_value = mock_process

                with pytest.raises(RuntimeError):
                    await provider._run_cli_command("Test prompt")

            assert any("discarded 1 text parts" in msg for msg in captured), (
                f"Expected discard log not found in: {captured}"
            )
        finally:
            _loguru_logger.remove(sink_id)

    @pytest.mark.asyncio
    async def test_run_cli_command_plain_text_nonzero_exit_with_stdout_raises(
        self, provider, monkeypatch: pytest.MonkeyPatch
    ):
        """Plain-text fallback must not treat failed processes as success."""
        monkeypatch.setenv("CHUNKHOUND_OPENCODE_JSON", "0")

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (
                b"Partial output",
                b"Some stderr",
            )
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            with pytest.raises(RuntimeError, match="Some stderr"):
                await provider._run_cli_command("Test prompt")

            assert mock_subprocess.call_count == provider._max_retries

    @pytest.mark.asyncio
    async def test_run_cli_command_invalid_utf8_in_stdout(self, provider):
        """Test invalid UTF-8 bytes are replaced and parsing continues."""
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
    async def test_run_cli_command_passes_prompt_via_stdin(
        self, provider, monkeypatch: pytest.MonkeyPatch
    ):
        """OpenCode CLI reads the prompt from stdin to avoid ARG_MAX limits."""
        monkeypatch.setenv("CHUNKHOUND_OPENCODE_JSON", "0")
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"Plain output", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await provider._run_cli_command("Test prompt")

            assert result == "Plain output"
            mock_process.communicate.assert_called_once_with(input=b"Test prompt")

    @pytest.mark.asyncio
    async def test_run_cli_command_pipes_stdin(
        self, provider, monkeypatch: pytest.MonkeyPatch
    ):
        """Non-interactive runs pipe the prompt via stdin instead of argv."""
        monkeypatch.setenv("CHUNKHOUND_OPENCODE_JSON", "0")
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"Plain output", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await provider._run_cli_command("Test prompt")

            assert result == "Plain output"
            assert mock_subprocess.call_args.kwargs["stdin"] is asyncio.subprocess.PIPE
            assert "Test prompt" not in mock_subprocess.call_args.args
            mock_process.communicate.assert_called_once_with(input=b"Test prompt")

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
        error_event = json.dumps(
            {
                "type": "error",
                "error": {"data": "rate limited"},
            }
        )
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
        events = "\n".join(
            [
                json.dumps({"type": "text", "part": "plain string, not a dict"}),
                json.dumps({"type": "text", "part": {"text": "OK"}}),
            ]
        )
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (events.encode(), b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await provider._run_cli_command("Test prompt")

            assert result == "OK"

    @pytest.mark.asyncio
    async def test_run_cli_command_system_prompt_format(
        self, provider, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that system and user prompts are concatenated and sent via stdin."""
        monkeypatch.setenv("CHUNKHOUND_OPENCODE_JSON", "0")
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"OK", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            await provider._run_cli_command("Test prompt", system="System message")

            expected_prompt = (
                b"System Instructions:\nSystem message\n\nUser Request:\nTest prompt"
            )
            mock_process.communicate.assert_called_once_with(input=expected_prompt)

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
                b"",
                b"error: unrecognized option --format json",
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
                b"",
                b"error: unrecognized option --format json",
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
        """Test fallback model after primary retries exhaust with timeout."""
        fallback_text_event = json.dumps(
            {"type": "text", "part": {"text": "Fallback response"}}
        )
        with patch.object(
            OpenCodeCLIProvider, "_opencode_available", return_value=True
        ):
            provider = OpenCodeCLIProvider(
                model="test-provider/primary-model",
                fallback_model="test-provider/fallback-model",
                timeout=30,
                max_retries=1,
            )

        with (
            patch("sys.platform", "linux"),
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("os.killpg", create=True),
        ):
            mock_process_timeout = AsyncMock()
            mock_process_timeout.pid = 4321
            mock_process_timeout.communicate.side_effect = asyncio.TimeoutError()
            mock_process_timeout.wait = AsyncMock()
            mock_process_timeout.kill = MagicMock()
            mock_process_timeout.returncode = None

            mock_process_fallback = AsyncMock()
            mock_process_fallback.communicate.return_value = (
                fallback_text_event.encode(),
                b"",
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
        with patch.object(
            OpenCodeCLIProvider, "_opencode_available", return_value=True
        ):
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

    @pytest.mark.asyncio
    async def test_run_cli_command_fallback_does_not_mutate_provider_model(self):
        """Fallback attempts must not rewrite the provider's primary model."""
        fallback_text_event = json.dumps(
            {"type": "text", "part": {"text": "Fallback response"}}
        )
        with patch.object(
            OpenCodeCLIProvider, "_opencode_available", return_value=True
        ):
            provider = OpenCodeCLIProvider(
                model="test-provider/primary-model",
                fallback_model="test-provider/fallback-model",
                timeout=30,
                max_retries=1,
            )

        with (
            patch("sys.platform", "linux"),
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("os.killpg", create=True),
        ):
            mock_process_timeout = AsyncMock()
            mock_process_timeout.pid = 4321
            mock_process_timeout.communicate.side_effect = asyncio.TimeoutError()
            mock_process_timeout.wait = AsyncMock()
            mock_process_timeout.kill = MagicMock()
            mock_process_timeout.returncode = None

            mock_process_fallback = AsyncMock()
            mock_process_fallback.communicate.return_value = (
                fallback_text_event.encode(),
                b"",
            )
            mock_process_fallback.returncode = 0

            mock_subprocess.side_effect = [mock_process_timeout, mock_process_fallback]

            assert await provider._run_cli_command("Test prompt") == "Fallback response"
            assert provider.model == "test-provider/primary-model"

    @pytest.mark.asyncio
    async def test_run_cli_command_fallback_error_reports_fallback_model(self):
        """Fallback failures must report the fallback model in error context."""
        with patch.object(
            OpenCodeCLIProvider, "_opencode_available", return_value=True
        ):
            provider = OpenCodeCLIProvider(
                model="test-provider/primary-model",
                fallback_model="test-provider/fallback-model",
                timeout=30,
                max_retries=1,
            )

        with (
            patch("sys.platform", "linux"),
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("os.killpg", create=True),
        ):
            mock_process_timeout = AsyncMock()
            mock_process_timeout.pid = 4321
            mock_process_timeout.communicate.side_effect = asyncio.TimeoutError()
            mock_process_timeout.wait = AsyncMock()
            mock_process_timeout.kill = MagicMock()
            mock_process_timeout.returncode = None

            mock_process_fallback = AsyncMock()
            mock_process_fallback.communicate.return_value = (b"", b"fallback exploded")
            mock_process_fallback.returncode = 1

            mock_subprocess.side_effect = [mock_process_timeout, mock_process_fallback]

            with pytest.raises(RuntimeError, match="fallback exploded") as exc:
                await provider._run_cli_command("Test prompt")

            msg = str(exc.value)
            assert "model=test-provider/fallback-model" in msg
            assert "model=test-provider/primary-model" in msg
            assert "timed out after 30s" in msg

    @pytest.mark.asyncio
    async def test_run_cli_command_fallback_timeout_does_not_recurse(self):
        """Fallback that also times out must raise RuntimeError (no infinite loop).

        The ``allow_fallback=False`` guard on the recursive call to
        ``_run_with_model`` prevents infinite recursion when the fallback
        model itself exhausts all retries with timeouts.
        """
        with patch.object(
            OpenCodeCLIProvider, "_opencode_available", return_value=True
        ):
            provider = OpenCodeCLIProvider(
                model="test-provider/primary-model",
                fallback_model="test-provider/fallback-model",
                timeout=30,
                max_retries=1,
            )

        with (
            patch("sys.platform", "linux"),
            patch("asyncio.create_subprocess_exec") as mock_subprocess,
            patch("os.killpg", create=True),
        ):
            mock_process_timeout = AsyncMock()
            mock_process_timeout.pid = 4321
            mock_process_timeout.communicate.side_effect = asyncio.TimeoutError()
            mock_process_timeout.wait = AsyncMock()
            mock_process_timeout.kill = MagicMock()
            mock_process_timeout.returncode = None

            mock_process_fallback = AsyncMock()
            mock_process_fallback.pid = 8765
            mock_process_fallback.communicate.side_effect = asyncio.TimeoutError()
            mock_process_fallback.wait = AsyncMock()
            mock_process_fallback.kill = MagicMock()
            mock_process_fallback.returncode = None

            # Primary exhausts retries (max_retries=1), triggers fallback.
            # Fallback also exhausts retries (max_retries=1), should raise.
            mock_subprocess.side_effect = [mock_process_timeout, mock_process_fallback]

            with pytest.raises(RuntimeError, match="timed out after 30s") as exc:
                await provider._run_cli_command("Test prompt")

            msg = str(exc.value)
            assert "fallback-model" in msg, (
                f"Fallback model name should appear in compound error: {msg}"
            )
            assert "primary-model" in msg, (
                f"Primary model name should appear in compound error: {msg}"
            )
            # Exactly 2 subprocess invocations — no recursion
            assert mock_subprocess.call_count == 2
