"""OpenCode CLI LLM provider implementation for ChunkHound deep research.

This provider wraps the OpenCode CLI (opencode run) to enable deep research
using the user's existing OpenCode configuration and access to 75+ LLM providers.

Note: This provider is configured for vanilla LLM behavior:
- Uses --format json for structured NDJSON output (error detection)
- Runs in non-interactive mode via opencode run
- Leverages existing opencode auth login credentials
- Supports all providers/models available via "opencode models"
- Supports --variant flag for reasoning effort control
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Literal

from loguru import logger

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.providers.llm.base_cli_provider import BaseCLIProvider

VALID_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}


@dataclass(frozen=True)
class _JSONParseResult:
    """Outcome of parsing NDJSON mode output."""

    action: Literal["success", "retry_plain", "failure"]
    text: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class _PhaseResult:
    """Outcome of running one provider phase."""

    action: Literal["success", "retry_plain", "timeout", "failure"]
    output: str | None = None
    error: RuntimeError | None = None
    attempts_used: int = 1


class OpenCodeCLIProvider(BaseCLIProvider):
    """OpenCode CLI provider using subprocess calls to opencode run."""

    _reasoning_effort: str | None = None

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "",
        base_url: str | None = None,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_retries: int = 3,
        reasoning_effort: str | None = None,
        fallback_model: str | None = None,
    ):
        """Initialize OpenCode CLI provider.

        Args:
            api_key: Not used (credentials managed by opencode auth)
            model: Model in provider/model format (e.g., "opencode/gpt-5-nano").
                Must be specified — no default since models depend on user config.
            base_url: Not used (CLI uses default endpoints)
            timeout: Request timeout in seconds
            max_retries: Total subprocess attempts allowed per model run.
                JSON-mode probes and plain-text fallback share this single
                budget; fallback_model gets one separate timeout-only run.
            reasoning_effort: Effort level mapped to --variant
                (minimal, low, medium, high, xhigh)
            fallback_model: Model to try once if primary model exhausts all retries
                with timeout errors (different provider queue avoids the same hotspot).
        """
        super().__init__(api_key, model, base_url, timeout, max_retries)
        self._reasoning_effort = self._validate_reasoning_effort(reasoning_effort)

        # Validate model format eagerly so invalid configs fail fast
        if self._model:
            self._validate_model_format(self._model)

        if fallback_model:
            self._validate_model_format(fallback_model)
        self._fallback_model: str | None = fallback_model

        # Check CLI availability
        if not self._opencode_available():
            logger.warning("OpenCode CLI not found in PATH")

    def _validate_reasoning_effort(self, effort: str | None) -> str | None:
        """Validate reasoning effort against allowed values.

        Raises:
            ValueError: If effort is not one of the allowed values.
        """
        if effort is None:
            return None
        normalized = effort.strip().lower()
        if normalized not in VALID_REASONING_EFFORTS:
            raise ValueError(
                f"Invalid reasoning_effort '{effort}', "
                f"must be one of {sorted(VALID_REASONING_EFFORTS)}"
            )
        return normalized

    def _format_json_flag_unsupported(self, err: str) -> bool:
        """Check if opencode CLI stderr indicates --format json is unsupported."""
        lowered = err.lower()
        if "--format" not in lowered and "json" not in lowered:
            return False
        return any(marker in lowered for marker in self.UNSUPPORTED_FLAG_MARKERS)

    def _get_provider_name(self) -> str:
        """Get the provider name."""
        return "opencode-cli"

    def _opencode_available(self) -> bool:
        """Check if opencode CLI is available in PATH."""
        try:
            result = subprocess.run(
                ["opencode", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _validate_model_format(self, model: str) -> None:
        """Validate that model follows provider/model format.

        Args:
            model: Model string to validate

        Raises:
            ValueError: If model format is invalid or empty
        """
        if not model:
            raise ValueError(
                "opencode-cli requires a model in provider/model format "
                "(e.g., opencode/nematron-3-super-free). "
                "Run 'opencode models' to see available models."
            )
        if "/" not in model:
            raise ValueError(
                f"Model must be in 'provider/model' format, got: {model}. "
                f"Run 'opencode models' to see available models."
            )

        provider, model_name = model.split("/", 1)
        provider = provider.strip()
        model_name = model_name.strip()
        if not provider:
            raise ValueError(f"Provider cannot be empty in model: {model}")
        if not model_name:
            raise ValueError(f"Model cannot be empty in model: {model}")

    def _describe_command_context(self, *, model: str, use_json: bool) -> str:
        """Build sanitized CLI context for errors without leaking prompt text."""
        parts = ["command=opencode run", f"model={model}"]
        if use_json:
            parts.append("format=json")
        if self._reasoning_effort:
            parts.append(f"variant={self._reasoning_effort}")
        return ", ".join(parts)

    async def _run_cli_command(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        """Run opencode CLI command and return output.

        Uses --format json to get structured NDJSON output for:
        - Error detection (opencode exits 0 even on errors)
        - Text extraction from streaming events

        Args:
            prompt: User prompt
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout override

        Returns:
            CLI output text

        Raises:
            RuntimeError: If CLI command fails or returns no content
        """

        return await self._run_with_model(
            prompt,
            system,
            max_completion_tokens,
            timeout,
            model=self._model,
            max_retries=self._max_retries,
            allow_fallback=True,
        )

    async def _run_with_model(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        timeout: int | None = None,
        *,
        model: str,
        max_retries: int,
        allow_fallback: bool = True,
    ) -> str:
        """Run opencode with explicit model and retry config (no instance mutation).

        Separated from _run_cli_command so the fallback path (C1, M3 fix) can
        call this with a different model without mutating self._model.
        """
        message = self._merge_prompts(prompt, system)
        request_timeout = timeout if timeout is not None else self._timeout

        # Phase 1: Try JSON mode within the model's single total attempt budget.
        json_phase = await self._try_phase(
            message,
            request_timeout,
            model=model,
            max_retries=max_retries,
            use_json=os.getenv("CHUNKHOUND_OPENCODE_JSON", "1") != "0",
        )
        if json_phase.action == "success":
            return json_phase.output or ""

        final_phase = json_phase

        # Phase 2: Retry in plain text mode only within the remaining budget.
        # ``max_retries`` means total attempts per model run, so a JSON probe
        # consumes attempts that plain-text fallback cannot reclaim.
        if json_phase.action == "retry_plain":
            remaining_attempts = max_retries - json_phase.attempts_used
            if remaining_attempts > 0:
                plain_phase = await self._try_phase(
                    message,
                    request_timeout,
                    model=model,
                    max_retries=remaining_attempts,
                    use_json=False,
                )
                if plain_phase.action == "success":
                    return plain_phase.output or ""
                final_phase = plain_phase
            else:
                cmd_ctx = self._describe_command_context(model=model, use_json=True)
                final_phase = _PhaseResult(
                    action="failure",
                    error=RuntimeError(
                        "OpenCode CLI exhausted retry budget before plain-text "
                        f"fallback ({cmd_ctx})"
                    ),
                    attempts_used=json_phase.attempts_used,
                )

        # Phase 3: Fallback model (timeout only)
        if (
            allow_fallback
            and self._fallback_model is not None
            and final_phase.action == "timeout"
            and final_phase.error is not None
        ):
            logger.info(
                f"Primary model {model!r} timed out; "
                f"trying fallback {self._fallback_model!r} once"
            )
            try:
                return await self._run_with_model(
                    prompt,
                    system,
                    max_completion_tokens,
                    timeout,
                    model=self._fallback_model,
                    max_retries=1,
                    allow_fallback=False,  # prevent recursion
                )
            except RuntimeError as fallback_error:
                raise RuntimeError(
                    f"{final_phase.error} Fallback model {self._fallback_model!r} "
                    f"also failed: {fallback_error}"
                ) from fallback_error

        raise final_phase.error or RuntimeError(
            "OpenCode CLI command failed after retries"
        )

    def _build_cmd(self, model: str, use_json: bool) -> list[str]:
        """Build the opencode run command list without the prompt.

        The prompt is sent via stdin to avoid ARG_MAX limits on large inputs.
        """
        cmd = ["opencode", "run", "--model", model]
        if use_json:
            cmd.extend(["--format", "json"])
        if self._reasoning_effort:
            cmd.extend(["--variant", self._reasoning_effort])
        return cmd

    def _ndjson_parse_stdout(self, stdout: bytes) -> tuple[list[str], str | None]:
        """Parse NDJSON output from opencode --format json.

        Returns (text_parts, error_message).
        error_message is set when an error event is encountered.
        """
        text_parts: list[str] = []
        error_message: str | None = None

        for line in stdout.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(event, dict):
                continue

            event_type = event.get("type")

            if event_type == "error":
                error_data = event.get("error", {})
                if isinstance(error_data, dict):
                    resolved_data = error_data.get("data")
                    if isinstance(resolved_data, dict):
                        error_message = resolved_data.get(
                            "message",
                            error_data.get("message", "Unknown error"),
                        )
                    elif resolved_data:
                        error_message = str(resolved_data)
                    else:
                        error_message = error_data.get("message", "Unknown error")
                else:
                    error_message = str(error_data) if error_data else "Unknown error"
                break

            if event_type == "text":
                part = event.get("part")
                if isinstance(part, dict):
                    part_text = part.get("text", "")
                    if part_text:
                        text_parts.append(part_text)

        return text_parts, error_message

    def _parse_json_output(
        self,
        stdout: bytes,
        stderr_msg: str,
        returncode: int,
        *,
        model: str,
    ) -> _JSONParseResult:
        """Parse NDJSON output into an explicit action.

        JSON mode only falls back to plain text when the CLI lacks the flag or
        when the stream contains no text parts to extract.
        """
        text_parts, error_message = self._ndjson_parse_stdout(stdout)

        if error_message:
            logger.debug(
                f"OpenCode CLI error event: {error_message}; "
                f"discarded {len(text_parts)} preceding text parts"
            )
            return _JSONParseResult(
                action="failure",
                error_message=f"OpenCode CLI error: {error_message}",
            )

        # If --format json is unsupported, retry in plain text mode.
        if returncode != 0 and self._format_json_flag_unsupported(stderr_msg):
            logger.info(
                "OpenCode CLI does not support --format json; "
                "retrying in plain text mode"
            )
            return _JSONParseResult(action="retry_plain")

        if returncode != 0:
            logger.debug(
                f"OpenCode CLI: discarded {len(text_parts)} text parts "
                f"from failed process (exit {returncode})"
            )
            cmd_ctx = self._describe_command_context(model=model, use_json=True)
            return _JSONParseResult(
                action="failure",
                error_message=(
                    f"OpenCode CLI command failed (exit {returncode}): "
                    f"{stderr_msg} ({cmd_ctx})"
                ),
            )

        # No text in JSON mode → retry plain text.
        if not text_parts:
            logger.info(
                "OpenCode CLI returned no text in JSON mode; "
                "retrying in plain text mode"
            )
            return _JSONParseResult(action="retry_plain")

        return _JSONParseResult(action="success", text="".join(text_parts).strip())

    async def _kill_process_tree(
        self, process: asyncio.subprocess.Process, *, pgid: int | None = None
    ) -> None:
        """Terminate an opencode subprocess and its descendants."""
        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["taskkill", "/T", "/PID", str(process.pid), "/F"],
                    check=False,
                    timeout=10,
                )
            except (FileNotFoundError, subprocess.SubprocessError, OSError):
                logger.debug("Windows taskkill failed during OpenCode cleanup")
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass
            return

        process_group_id = pgid if pgid is not None else process.pid
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass

        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass

        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass

    async def _run_single_attempt(
        self,
        message: str,
        request_timeout: int,
        *,
        model: str,
        use_json: bool,
    ) -> _PhaseResult:
        """Run one opencode subprocess attempt and return a typed result.

        Separated from ``_try_phase`` so the retry-loop orchestration stays
        readable while the per-attempt subprocess/parse/exception handling
        lives in one focused method.
        """
        cmd = self._build_cmd(model, use_json)
        process = None
        process_pgid: int | None = None
        cleanup_required = False
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
                cwd=tempfile.gettempdir(),
                start_new_session=True,
            )
            cleanup_required = True
            if sys.platform != "win32":
                process_pgid = process.pid

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=message.encode("utf-8")),
                timeout=request_timeout,
            )
            stderr_msg = (
                stderr.decode("utf-8", errors="replace").strip()
                if stderr
                else "Unknown error"
            )

            # communicate() completed — returncode is guaranteed set
            if process.returncode is None:
                raise RuntimeError(
                    "Subprocess communicate() completed but returncode is None"
                )
            cleanup_required = False

            if use_json:
                json_result = self._parse_json_output(
                    stdout, stderr_msg, process.returncode, model=model
                )
                if json_result.action == "retry_plain":
                    return _PhaseResult(action="retry_plain", attempts_used=1)
                if json_result.action == "failure":
                    # Defensive guard — should never happen.
                    if json_result.error_message is None:
                        raise RuntimeError(
                            "_parse_json_output returned failure without error_message"
                        )
                    return _PhaseResult(
                        action="failure",
                        error=RuntimeError(json_result.error_message),
                        attempts_used=1,
                    )

                output = json_result.text or ""
                if not output.strip():
                    return _PhaseResult(
                        action="failure",
                        error=RuntimeError(
                            "OpenCode CLI returned no text content. "
                            "The model may have produced no output "
                            "or only tool calls."
                        ),
                        attempts_used=1,
                    )
                return _PhaseResult(action="success", output=output, attempts_used=1)

            if process.returncode != 0:
                cmd_ctx = self._describe_command_context(
                    model=model,
                    use_json=False,
                )
                return _PhaseResult(
                    action="failure",
                    error=RuntimeError(
                        f"OpenCode CLI command failed "
                        f"(exit {process.returncode}): "
                        f"{stderr_msg} ({cmd_ctx})"
                    ),
                    attempts_used=1,
                )

            output = stdout.decode("utf-8", errors="replace").strip()
            if not output:
                cmd_ctx = self._describe_command_context(
                    model=model,
                    use_json=False,
                )
                return _PhaseResult(
                    action="failure",
                    error=RuntimeError(
                        f"OpenCode CLI returned empty output ({cmd_ctx})"
                    ),
                    attempts_used=1,
                )
            return _PhaseResult(action="success", output=output, attempts_used=1)

        except asyncio.TimeoutError:
            cmd_ctx = self._describe_command_context(model=model, use_json=use_json)
            error = RuntimeError(
                f"OpenCode CLI command timed out after {request_timeout}s "
                f"({cmd_ctx})"
            )
            return _PhaseResult(action="timeout", error=error, attempts_used=1)

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise

            cmd_ctx = self._describe_command_context(model=model, use_json=use_json)
            error = RuntimeError(f"OpenCode CLI command failed: {e} ({cmd_ctx})")
            error.__cause__ = e
            return _PhaseResult(action="failure", error=error, attempts_used=1)

        finally:
            if cleanup_required and process is not None:
                await self._kill_process_tree(process, pgid=process_pgid)

    async def _try_phase(
        self,
        message: str,
        request_timeout: int,
        *,
        model: str,
        max_retries: int,
        use_json: bool,
    ) -> _PhaseResult:
        """Try a phase (JSON or plain) with up to max_retries attempts.

        Calls ``_run_single_attempt`` up to *max_retries* times and reports
        how much of the caller's total per-model attempt budget was consumed.
        When a JSON attempt signals ``retry_plain`` the method returns
        immediately so the caller can spend only the remaining budget in plain
        text mode.
        """
        attempts_used = 0
        for attempt in range(max_retries):
            result = await self._run_single_attempt(
                message, request_timeout, model=model, use_json=use_json
            )
            attempts_used += result.attempts_used
            if result.action == "retry_plain":
                return _PhaseResult(
                    action="retry_plain",
                    attempts_used=attempts_used,
                )
            if result.action == "success":
                return _PhaseResult(
                    action="success",
                    output=result.output,
                    attempts_used=attempts_used,
                )

            # failure or timeout — log and either retry or return
            if result.error is None:  # defensive — should never happen
                raise RuntimeError(
                    "_run_single_attempt returned failure/timeout without setting error"
                )
            if attempt < max_retries - 1:
                tag = use_json and "json" or "plain"
                if result.action == "timeout":
                    logger.warning(
                        f"OpenCode CLI attempt {attempt + 1} timed out "
                        f"(model={model!r}, format={tag}), retrying"
                    )
                else:
                    logger.warning(
                        f"OpenCode CLI attempt {attempt + 1} failed "
                        f"(model={model!r}, format={tag}), retrying: {result.error}"
                    )
                continue
            return _PhaseResult(
                action=result.action,
                error=result.error,
                attempts_used=attempts_used,
            )

        return _PhaseResult(
            action="failure",
            error=RuntimeError("OpenCode CLI command failed after retries"),
            attempts_used=attempts_used,
        )
