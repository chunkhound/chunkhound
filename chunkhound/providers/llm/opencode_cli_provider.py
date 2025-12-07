"""OpenCode CLI LLM provider implementation for ChunkHound deep research.

This provider wraps the OpenCode CLI (opencode run) to enable deep research
using the user's existing OpenCode configuration and access to 75+ LLM providers.

Note: This provider is configured for vanilla LLM behavior:
- Uses default text format for simple, reliable output
- Runs in non-interactive mode via opencode run
- Leverages existing opencode auth login credentials
- Supports all providers/models available via "opencode models"
"""

import asyncio
import json
import os
import subprocess
import tempfile
from typing import Any

from loguru import logger

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse
from chunkhound.utils.json_extraction import extract_json_from_response


class OpenCodeCLIProvider(LLMProvider):
    """OpenCode CLI provider using subprocess calls to opencode run."""

    # Constants for timeouts and estimation
    TOKEN_CHARS_RATIO = 4  # Approximate characters per token for most models
    HEALTH_CHECK_TIMEOUT = 30  # Seconds to wait for health check completion

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "opencode/grok-code",
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize OpenCode CLI provider.

        Args:
            api_key: Not used (credentials managed by opencode auth)
            model: Model name to use in provider/model format
                (e.g., "opencode/grok-code")
            base_url: Not used (CLI uses default endpoints)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
        """
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

        # Usage tracking
        self._requests_made = 0
        self._estimated_tokens_used = 0
        self._estimated_prompt_tokens = 0
        self._estimated_completion_tokens = 0

    def _validate_model_format(self, model: str) -> None:
        """Validate that model follows provider/model format.

        Args:
            model: Model string to validate

        Raises:
            ValueError: If model format is invalid
        """
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

        Args:
            prompt: User prompt
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout override

        Returns:
            CLI output text

        Raises:
            RuntimeError: If CLI command fails
        """
        # Validate model format
        self._validate_model_format(self._model)

        # Build CLI command
        cmd = [
            "opencode",
            "run",
            "--model",
            self._model,
        ]

        # Add system prompt if provided
        if system:
            cmd.append(system + "\n" + prompt)
        else:
            cmd.append(prompt)

        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self._timeout

        # Run command with retry logic
        last_error = None
        for attempt in range(self._max_retries):
            process = None
            try:
                # Create subprocess with neutral CWD to prevent workspace scanning
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=subprocess.DEVNULL,  # Prevent stdin inheritance
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=os.environ,  # Use existing environment (includes opencode auth)
                    cwd=tempfile.gettempdir(),  # Cross-platform temp directory
                )

                # Wrap communicate() with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=request_timeout,
                )

                if process.returncode != 0:
                    error_msg = stderr.decode("utf-8") if stderr else "Unknown error"
                    last_error = RuntimeError(
                        f"OpenCode CLI command failed (exit {process.returncode}): "
                        f"{error_msg}"
                    )
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            f"OpenCode CLI attempt {attempt + 1} failed, "
                            f"retrying: {error_msg}"
                        )
                        continue
                    raise last_error

                return stdout.decode("utf-8").strip()

            except asyncio.TimeoutError as e:
                # Kill the subprocess if it's still running
                if process and process.returncode is None:
                    process.kill()
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
                # Kill the subprocess if it's still running on unexpected errors
                if process and process.returncode is None:
                    process.kill()
                    await process.wait()

                last_error = RuntimeError(
                    f"OpenCode CLI command failed: {e}, with command {cmd}"
                )
                if attempt < self._max_retries - 1:
                    logger.warning(f"OpenCode CLI attempt {attempt + 1} failed: {e}")
                    continue
                raise last_error from e

        # Should not reach here, but just in case
        raise last_error or RuntimeError("OpenCode CLI command failed after retries")

    @property
    def name(self) -> str:
        """Provider name."""
        return "opencode-cli"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            LLMResponse with content and estimated token usage
        """
        try:
            content = await self._run_cli_command(
                prompt, system, max_completion_tokens, timeout
            )
            logger.warning(content)

            # Validate content is not empty
            if not content or not content.strip():
                logger.error(
                    "OpenCode CLI returned empty content "
                    f"(model={self._model}, prompt_length={len(prompt)})"
                )
                raise RuntimeError(
                    "LLM returned empty response from OpenCode CLI. This may "
                    "indicate a CLI error, authentication issue, or model refusal."
                )

            # Track usage (estimates since CLI doesn't return token counts)
            self._requests_made += 1
            prompt_tokens = self.estimate_tokens(prompt)
            if system:
                prompt_tokens += self.estimate_tokens(system)
            completion_tokens = self.estimate_tokens(content)
            total_tokens = prompt_tokens + completion_tokens

            self._estimated_prompt_tokens += prompt_tokens
            self._estimated_completion_tokens += completion_tokens
            self._estimated_tokens_used += total_tokens

            return LLMResponse(
                content=content,
                tokens_used=total_tokens,
                model=self._model,
                finish_reason="stop",  # CLI doesn't provide this
            )

        except Exception as e:
            logger.error(f"OpenCode CLI completion failed: {e}")
            raise RuntimeError(f"LLM completion failed: {e}") from e

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON completion conforming to the given schema.

        Since OpenCode CLI doesn't support native JSON schema validation,
        we include the schema in the prompt and request JSON output.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            Parsed JSON object

        Raises:
            RuntimeError: If output is not valid JSON or doesn't match schema
        """
        # Build structured prompt with schema
        structured_prompt = f"""Please respond with ONLY valid JSON that conforms
to this schema:

{json.dumps(json_schema, indent=2)}

User request: {prompt}

Respond with JSON only, no additional text."""

        try:
            content = await self._run_cli_command(
                structured_prompt, system, max_completion_tokens, timeout
            )
            logger.warning(content)
            # Validate content is not empty
            if not content or not content.strip():
                logger.error(
                    "OpenCode CLI structured completion returned empty content"
                )
                raise RuntimeError(
                    "LLM structured completion returned empty response from "
                    "OpenCode CLI"
                )

            # Track usage
            self._requests_made += 1
            prompt_tokens = self.estimate_tokens(structured_prompt)
            if system:
                prompt_tokens += self.estimate_tokens(system)
            completion_tokens = self.estimate_tokens(content)
            total_tokens = prompt_tokens + completion_tokens

            self._estimated_prompt_tokens += prompt_tokens
            self._estimated_completion_tokens += completion_tokens
            self._estimated_tokens_used += total_tokens

            # Extract JSON from response (handle markdown code blocks)
            json_content = extract_json_from_response(content)

            # Parse JSON
            parsed = json.loads(json_content)

            # Ensure parsed is a dict
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

            # Basic schema validation (check required fields if specified)
            if "required" in json_schema:
                missing = [
                    field for field in json_schema["required"] if field not in parsed
                ]
                if missing:
                    raise ValueError(f"Missing required fields: {missing}")

            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output as JSON: {e}")
            logger.debug(f"Raw output: {locals().get('content', 'N/A')}")
            raise RuntimeError(f"Invalid JSON in structured output: {e}") from e
        except Exception as e:
            logger.error(f"OpenCode CLI structured completion failed: {e}")
            raise RuntimeError(f"LLM structured completion failed: {e}") from e

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        """Generate completions for multiple prompts concurrently.

        Note: CLI doesn't support true batch API, so we run sequentially
        to avoid overwhelming the CLI or rate limits.
        """
        results = []
        for prompt in prompts:
            result = await self.complete(prompt, system, max_completion_tokens)
            results.append(result)
        return results

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses rough approximation since we don't have direct tokenizer access.
        Most models use ~4 characters per token.
        """
        return len(text) // self.TOKEN_CHARS_RATIO

    async def health_check(self) -> dict[str, Any]:
        """Perform health check by attempting a simple completion.

        This will naturally detect if the CLI is missing or incompatible.
        """
        try:
            response = await self.complete(
                "Say 'OK'", max_completion_tokens=10, timeout=self.HEALTH_CHECK_TIMEOUT
            )
            return {
                "status": "healthy",
                "provider": "opencode-cli",
                "model": self._model,
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "opencode-cli",
                "error": str(e),
            }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics (estimates since CLI doesn't return actual counts)."""
        return {
            "requests_made": self._requests_made,
            "total_tokens_estimated": self._estimated_tokens_used,
            "prompt_tokens_estimated": self._estimated_prompt_tokens,
            "completion_tokens_estimated": self._estimated_completion_tokens,
        }

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations.

        Returns:
            3 for OpenCode CLI (conservative default matching other CLI providers)
        """
        return 3
