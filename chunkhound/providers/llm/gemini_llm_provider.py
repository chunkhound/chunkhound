"""Google Gemini LLM provider implementation for ChunkHound deep research."""

import asyncio
import json
from typing import Any

from loguru import logger

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse

try:
    from google import genai
    from google.genai import errors, types

    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    errors = None  # type: ignore
    GENAI_AVAILABLE = False
    logger.warning("google-genai not available - install with: uv add google-genai")


class GeminiLLMProvider(LLMProvider):
    """Google Gemini LLM provider using official Google Gen AI SDK.

    Supported models:
    - Gemini 3: gemini-3-pro-preview (most advanced, reasoning-optimized)
    - Gemini 2.5: gemini-2.5-pro (balanced capability/speed), gemini-2.5-flash (fast)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3-pro-preview",
        thinking_level: str = "high",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """Initialize Gemini LLM provider.

        Args:
            api_key: Google AI API key (get from https://aistudio.google.com/apikey)
            model: Model name to use
                - "gemini-3-pro-preview": Latest, most capable (default)
                - "gemini-2.5-pro": Balanced performance
                - "gemini-2.5-flash": Fast, cost-effective
            thinking_level: Reasoning depth ("low", "high"). Default "high" for deep research.
                Note: Gemini 3 uses thinking_level; 2.5 series uses thinking_budget.
            timeout: Request timeout in seconds (Gemini reasoning can be slow)
            max_retries: Number of retry attempts for failed requests
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai not available - install with: uv add google-genai"
            )

        if not api_key:
            raise ValueError(
                "Gemini API key required. Get one at: https://aistudio.google.com/apikey"
            )

        self._api_key = api_key
        self._model = model
        self._thinking_level = thinking_level
        self._timeout = timeout
        self._max_retries = max_retries

        # Initialize Google Gen AI client with timeout and retry configuration
        # Note: Google SDK applies these at client level, not per-request
        http_options = types.HttpOptions(
            timeout=timeout,
            retry_options=types.HttpRetryOptions(attempts=max_retries)
            if max_retries is not None
            else None,
        )
        self._client = genai.Client(api_key=api_key, http_options=http_options)

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return "gemini"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    def _build_generation_config(
        self,
        max_completion_tokens: int = 4096,
        json_schema: dict[str, Any] | None = None,
        system_instruction: str | None = None,
    ) -> types.GenerateContentConfig:
        """Build generation configuration for Gemini API.

        Args:
            max_completion_tokens: Maximum tokens to generate
            json_schema: Optional JSON schema for structured outputs
            system_instruction: Optional system instruction

        Returns:
            GenerateContentConfig object
        """
        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_completion_tokens,
            "temperature": 1.0,  # Gemini 3 optimized for default 1.0
        }

        # Add system instruction if provided
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        # Add thinking configuration based on model
        if "gemini-3" in self._model:
            # Gemini 3 uses thinking_level
            if self._thinking_level != "high":
                config_kwargs["thinking_level"] = self._thinking_level
        elif "gemini-2.5" in self._model:
            # Gemini 2.5 uses thinking_budget (convert level to budget)
            if self._thinking_level == "low":
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=0
                )
            # Default/high: let model decide thinking budget

        # Add structured output config if schema provided
        if json_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = json_schema

        return types.GenerateContentConfig(**config_kwargs)

    def _handle_api_error(self, e: Any, operation: str) -> RuntimeError:
        """Handle Gemini API errors with detailed messages.

        Args:
            e: The APIError exception
            operation: Description of the operation that failed

        Returns:
            RuntimeError with detailed message
        """
        if hasattr(e, "code") and hasattr(e, "message"):
            code = e.code
            message = e.message

            # Handle specific error codes
            if code == 404:
                return RuntimeError(
                    f"Gemini model '{self._model}' not found. "
                    f"Check model name or API availability."
                )
            elif code == 429:
                return RuntimeError(
                    f"Gemini rate limit exceeded during {operation}. "
                    f"Please retry after a delay or reduce request frequency."
                )
            elif code == 400:
                return RuntimeError(
                    f"Invalid Gemini request during {operation}: {message}"
                )
            elif code in (401, 403):
                return RuntimeError(
                    f"Gemini authentication failed: {message}. "
                    f"Check your API key at https://aistudio.google.com/apikey"
                )
            else:
                return RuntimeError(
                    f"Gemini API error ({code}) during {operation}: {message}"
                )
        else:
            return RuntimeError(f"Gemini {operation} failed: {e}")

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
            system: Optional system prompt (combined with user prompt for Gemini)
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (accepted for interface consistency,
                but Google SDK does not support per-request timeout overrides)

        Returns:
            LLMResponse with content and metadata
        """
        # Build config with system instruction if provided
        config = self._build_generation_config(
            max_completion_tokens=max_completion_tokens,
            system_instruction=system,
        )

        try:
            # Use native async client with proper context management
            async with self._client.aio as aclient:
                response = await aclient.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )

            self._requests_made += 1

            # Extract response content
            content = response.text
            if not content or not content.strip():
                raise RuntimeError(
                    "Gemini returned empty response. "
                    "This may indicate a content filter, API error, or model refusal."
                )

            # Extract usage metadata
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, "prompt_token_count", 0)
                completion_tokens = getattr(usage, "candidates_token_count", 0)
                total_tokens = getattr(
                    usage, "total_token_count", prompt_tokens + completion_tokens
                )

                self._prompt_tokens += prompt_tokens
                self._completion_tokens += completion_tokens
                self._tokens_used += total_tokens
            else:
                total_tokens = 0

            # Extract finish reason
            finish_reason = "STOP"
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason"):
                    finish_reason = str(candidate.finish_reason)

            # Validate finish reason
            if finish_reason in ("MAX_TOKENS", "FINISHREASON_MAX_TOKENS"):
                raise RuntimeError(
                    f"Gemini response truncated - token limit exceeded. "
                    f"The output budget was set to {max_completion_tokens:,} tokens. "
                    f"For reasoning models like Gemini 3, this may indicate extensive internal "
                    f"reasoning that exhausted the output budget. "
                    f"Try breaking your query into smaller, more focused questions."
                )

            if finish_reason in (
                "SAFETY",
                "FINISHREASON_SAFETY",
                "RECITATION",
                "FINISHREASON_RECITATION",
            ):
                raise RuntimeError(
                    f"Gemini response blocked ({finish_reason}). "
                    "Try rephrasing your query or adjusting the prompt."
                )

            return LLMResponse(
                content=content,
                tokens_used=total_tokens,
                model=self._model,
                finish_reason=finish_reason,
            )

        except Exception as e:
            # Handle API errors with detailed messages
            if GENAI_AVAILABLE and isinstance(e, errors.APIError):
                raise self._handle_api_error(e, "completion")

            logger.error(f"Gemini completion failed: {e}")
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

        Uses Gemini's structured outputs with JSON Schema validation.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (accepted for interface consistency,
                but Google SDK does not support per-request timeout overrides)

        Returns:
            Parsed JSON object conforming to schema
        """
        # Build config with system instruction and schema
        config = self._build_generation_config(
            max_completion_tokens=max_completion_tokens,
            json_schema=json_schema,
            system_instruction=system,
        )

        try:
            # Use native async client with proper context management
            async with self._client.aio as aclient:
                response = await aclient.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )

            self._requests_made += 1

            # Extract response content
            content = response.text
            if not content or not content.strip():
                raise RuntimeError(
                    "Gemini structured completion returned empty response"
                )

            # Extract usage metadata
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, "prompt_token_count", 0)
                completion_tokens = getattr(usage, "candidates_token_count", 0)
                total_tokens = getattr(
                    usage, "total_token_count", prompt_tokens + completion_tokens
                )

                self._prompt_tokens += prompt_tokens
                self._completion_tokens += completion_tokens
                self._tokens_used += total_tokens

            # Extract finish reason
            finish_reason = "STOP"
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason"):
                    finish_reason = str(candidate.finish_reason)

            # Validate finish reason
            if finish_reason in ("MAX_TOKENS", "FINISHREASON_MAX_TOKENS"):
                raise RuntimeError(
                    "Gemini structured completion truncated - token limit exceeded. "
                    "This indicates insufficient max_completion_tokens for the structured output. "
                    "Consider increasing the token limit or reducing input context."
                )

            # Parse JSON response
            try:
                parsed: dict[str, Any] = json.loads(content)
                return parsed
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini structured output as JSON: {e}")
                logger.debug(f"Content: {content}")
                raise RuntimeError(f"Invalid JSON in structured output: {e}") from e

        except Exception as e:
            # Handle API errors with detailed messages
            if GENAI_AVAILABLE and isinstance(e, errors.APIError):
                raise self._handle_api_error(e, "structured completion")

            logger.error(f"Gemini structured completion failed: {e}")
            raise RuntimeError(f"LLM structured completion failed: {e}") from e

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        """Generate completions for multiple prompts concurrently."""
        tasks = [
            self.complete(prompt, system, max_completion_tokens) for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation).

        Gemini uses similar tokenization to other models (~4 chars per token).
        """
        return len(text) // 4

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        try:
            response = await self.complete("Say 'OK'", max_completion_tokens=10)
            return {
                "status": "healthy",
                "provider": "gemini",
                "model": self._model,
                "thinking_level": self._thinking_level,
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "gemini",
                "model": self._model,
                "error": str(e),
            }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "requests_made": self._requests_made,
            "total_tokens": self._tokens_used,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations.

        Returns:
            2 for Gemini (conservative due to thinking time and rate limits)
        """
        return 2
