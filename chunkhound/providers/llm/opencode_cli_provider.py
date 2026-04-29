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
import subprocess
import tempfile

from loguru import logger

from chunkhound.providers.llm.base_cli_provider import BaseCLIProvider

VALID_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}


class OpenCodeCLIProvider(BaseCLIProvider):
    """OpenCode CLI provider using subprocess calls to opencode run."""

    _reasoning_effort: str | None = None

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "",
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        reasoning_effort: str | None = None,
    ):
        """Initialize OpenCode CLI provider.

        Args:
            api_key: Not used (credentials managed by opencode auth)
            model: Model in provider/model format (e.g., "opencode/gpt-5-nano").
                Must be specified — no default since models depend on user config.
            base_url: Not used (CLI uses default endpoints)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
            reasoning_effort: Effort level mapped to --variant
                (minimal, low, medium, high, xhigh)
        """
        super().__init__(api_key, model, base_url, timeout, max_retries)
        self._reasoning_effort = self._validate_reasoning_effort(reasoning_effort)

        # Validate model format eagerly so invalid configs fail fast
        if self._model:
            self._validate_model_format(self._model)

        # Check CLI availability
        if not self._opencode_available():
            logger.warning("OpenCode CLI not found in PATH")

    def _validate_reasoning_effort(
        self, effort: str | None
    ) -> str | None:
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

        provider, _ = model.split("/", 1)
        if not provider:
            raise ValueError(f"Provider cannot be empty in model: {model}")

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
        if max_completion_tokens is not None:
            logger.warning(
                "max_completion_tokens not supported by opencode-cli, ignoring"
            )

        # Honor env override to disable JSON format
        use_json = os.getenv("CHUNKHOUND_OPENCODE_JSON", "1") != "0"

        # Build message with optional system prompt
        message = self._merge_prompts(prompt, system)

        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self._timeout

        # Run command with retry logic
        last_error = None
        for attempt in range(self._max_retries):
            # Build CLI command (rebuilt each attempt to support flag negotiation)
            cmd = [
                "opencode",
                "run",
                "--model",
                self._model,
            ]
            if use_json:
                cmd.extend(["--format", "json"])

            # Add reasoning effort as --variant flag
            if self._reasoning_effort:
                cmd.extend(["--variant", self._reasoning_effort])

            cmd.append(message)
            process = None
            try:
                # Create subprocess with neutral CWD to prevent workspace scanning
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=os.environ.copy(),
                    cwd=tempfile.gettempdir(),
                )

                # Wrap communicate() with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=request_timeout,
                )

                if use_json:
                    # Parse NDJSON output
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
                                    error_message = error_data.get(
                                        "message", "Unknown error"
                                    )
                            else:
                                error_message = (
                                    str(error_data) if error_data else "Unknown error"
                                )
                            break

                        if event_type == "text":
                            part = event.get("part")
                            if isinstance(part, dict):
                                part_text = part.get("text", "")
                                if part_text:
                                    text_parts.append(part_text)

                    # If we got an error event, raise it
                    if error_message:
                        last_error = RuntimeError(
                            f"OpenCode CLI error: {error_message}"
                        )
                        if attempt < self._max_retries - 1:
                            logger.warning(
                                f"OpenCode CLI attempt {attempt + 1} failed, "
                                f"retrying: {error_message}"
                            )
                            continue
                        raise last_error

                    # If exit code is non-zero and we have no text, report stderr
                    if process.returncode != 0 and not text_parts:
                        stderr_msg = (
                            stderr.decode("utf-8", errors="replace").strip()
                            if stderr
                            else "Unknown error"
                        )
                        # If --format json is unsupported, retry in plain text mode
                        if self._format_json_flag_unsupported(stderr_msg):
                            use_json = False
                            logger.info(
                                "OpenCode CLI does not support --format json; "
                                "retrying in plain text mode"
                            )
                            continue
                        last_error = RuntimeError(
                            f"OpenCode CLI command failed (exit {process.returncode}): "
                            f"{stderr_msg} (command: {' '.join(cmd)})"
                        )
                        if attempt < self._max_retries - 1:
                            logger.warning(
                                f"OpenCode CLI attempt {attempt + 1} failed, "
                                f"retrying: {stderr_msg}"
                            )
                            continue
                        raise last_error

                    # If no text was extracted, report failure immediately.
                    # Retrying with identical parameters is futile — the model
                    # will produce the same (empty) output every time.
                    if not text_parts:
                        raise RuntimeError(
                            "OpenCode CLI returned no text content. "
                            "The model may have produced no output or only tool calls."
                        )

                    return "".join(text_parts)
                else:
                    # Plain text mode
                    output = stdout.decode("utf-8", errors="replace").strip()
                    if not output:
                        stderr_msg = (
                            stderr.decode("utf-8", errors="replace").strip()
                            if stderr
                            else "Unknown error"
                        )
                        if process.returncode != 0:
                            last_error = RuntimeError(
                                f"OpenCode CLI command failed "
                                f"(exit {process.returncode}): "
                                f"{stderr_msg} (command: {' '.join(cmd)})"
                            )
                        else:
                            last_error = RuntimeError(
                                "OpenCode CLI returned empty output"
                            )
                        if attempt < self._max_retries - 1:
                            logger.warning(
                                f"OpenCode CLI attempt {attempt + 1} "
                                f"returned empty output, retrying"
                            )
                            continue
                        raise last_error
                    return output

            except asyncio.TimeoutError as e:
                # Kill the subprocess if it's still running
                if process and process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()

                last_error = RuntimeError(
                    f"OpenCode CLI command timed out after {request_timeout}s"
                )
                if attempt < self._max_retries - 1:
                    logger.warning(
                        f"OpenCode CLI attempt {attempt + 1} timed out, retrying"
                    )
                    continue
                raise last_error from e

            except Exception as e:
                # Re-raise RuntimeError directly to avoid double-wrapping
                if isinstance(e, RuntimeError):
                    raise

                # Kill the subprocess if it's still running on unexpected errors
                if process and process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()

                last_error = RuntimeError(
                    f"OpenCode CLI command failed: {e} (command: {' '.join(cmd)})"
                )
                if attempt < self._max_retries - 1:
                    logger.warning(f"OpenCode CLI attempt {attempt + 1} failed: {e}")
                    continue
                raise last_error from e

        # Should not reach here, but just in case
        raise last_error or RuntimeError("OpenCode CLI command failed after retries")
