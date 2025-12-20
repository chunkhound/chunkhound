"""Base CLI provider for LLM providers that use command-line interfaces.

This base class contains shared logic for CLI-based providers
(ClaudeCode, Codex, OpenCode) to avoid code duplication and ensure
consistent behavior.
"""

import json
from abc import abstractmethod
from typing import Any

from loguru import logger

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse
from chunkhound.utils.json_extraction import extract_json_from_response


class BaseCLIProvider(LLMProvider):
    """Base class for CLI-based LLM providers.

    Subclasses must implement:
    - _run_cli_command(): Execute the actual CLI command
    - _get_provider_name(): Return the provider name string
    """

    # Constants for token estimation and timeouts
    TOKEN_CHARS_RATIO = 4  # Approximate characters per token
    HEALTH_CHECK_TIMEOUT = 30  # Seconds to wait for health check

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "default",
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize base CLI provider.

        Args:
            api_key: API key (may not be used by CLI providers)
            model: Model name to use
            base_url: Base URL (may not be used by CLI providers)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
        """
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

        # Usage tracking (estimates since CLIs don't return token counts)
        self._requests_made = 0
        self._estimated_tokens_used = 0
        self._estimated_prompt_tokens = 0
        self._estimated_completion_tokens = 0

    @abstractmethod
    async def _run_cli_command(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        """Run CLI command and return output.

        This method must be implemented by subclasses to execute their
        specific CLI command.

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
        ...

    @abstractmethod
    def _get_provider_name(self) -> str:
        """Get the provider name for this CLI provider.

        Returns:
            Provider name (e.g., "claude-code-cli", "codex-cli", "opencode-cli")
        """
        ...

    @property
    def name(self) -> str:
        """Provider name."""
        return self._get_provider_name()

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

            # Validate content is not empty
            if not content or not content.strip():
                logger.error(
                    f"{self.name} returned empty content "
                    f"(model={self._model}, prompt_length={len(prompt)})"
                )
                raise RuntimeError(
                    f"LLM returned empty response from {self.name}. This may "
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
            logger.error(f"{self.name} completion failed: {e}")
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

        Since CLI providers don't support native JSON schema validation,
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
        structured_prompt = f"""Please respond with ONLY valid JSON that conforms to this schema:

{json.dumps(json_schema, indent=2)}

User request: {prompt}

Respond with JSON only, no additional text."""

        try:
            content = await self._run_cli_command(
                structured_prompt, system, max_completion_tokens, timeout
            )

            # Validate content is not empty
            if not content or not content.strip():
                logger.error(
                    f"{self.name} structured completion returned empty content"
                )
                raise RuntimeError(
                    f"LLM structured completion returned empty response from "
                    f"{self.name}"
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
            logger.debug(f"Raw output: {content if 'content' in locals() else 'N/A'}")
            raise RuntimeError(f"Invalid JSON in structured output: {e}") from e
        except Exception as e:
            logger.error(f"{self.name} structured completion failed: {e}")
            raise RuntimeError(f"LLM structured completion failed: {e}") from e

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        """Generate completions for multiple prompts.

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
                "Say 'OK'",
                max_completion_tokens=10,
                timeout=self.HEALTH_CHECK_TIMEOUT,
            )
            return {
                "status": "healthy",
                "provider": self.name,
                "model": self._model,
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.name,
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
            3 for CLI providers (conservative default)
        """
        return 3
