"""OpenAI-compatible LLM provider base class.

Provides common functionality for providers that use OpenAI-compatible APIs
(Chat Completions API with JSON Schema structured outputs).

Subclasses must implement:
- _get_provider_name(): Return the provider name
- _get_default_base_url(): Return the default base URL
"""

import asyncio
import json
from typing import Any

import httpx
from loguru import logger

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.core.config.openai_utils import is_official_openai_endpoint
from chunkhound.core.utils.token_utils import estimate_tokens_llm
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available - install with: uv pip install openai")


class OpenAICompatibleProvider(LLMProvider):
    """Base class for OpenAI-compatible LLM providers.

    Handles Chat Completions API with structured outputs.
    Subclasses can override base URL and provider name.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        base_url: str | None = None,
        ssl_verify: bool = True,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_retries: int = 3,
    ):
        """Initialize OpenAI-compatible provider.

        Args:
            api_key: API key (defaults to environment variable)
            model: Model name
            base_url: Base URL (defaults to subclass implementation)
            ssl_verify: Verify TLS certificates for requests sent via base_url.
                Ignored when base_url is not set.
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI not available - install with: uv pip install openai"
            )

        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._ssl_verify = ssl_verify

        # Use provided base_url or subclass default
        effective_base_url = base_url or self._get_default_base_url()

        # Initialize OpenAI-compatible client
        api_key_value = api_key
        if not is_official_openai_endpoint(effective_base_url) and not api_key_value:
            # The OpenAI SDK still expects a string even for local/custom backends
            api_key_value = "not-required"

        client_kwargs: dict[str, Any] = {
            "api_key": api_key_value,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if effective_base_url:
            client_kwargs["base_url"] = effective_base_url
            if not ssl_verify:
                client_kwargs["http_client"] = httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout=timeout),
                    verify=False,
                )
        self._client = AsyncOpenAI(**client_kwargs)

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def _get_default_base_url(self) -> str | None:
        """Get the default base URL for this provider.

        Subclasses must implement this to provide their API endpoint.
        Returns None to allow AsyncOpenAI to fall back to environment variables.
        """
        raise NotImplementedError("Subclasses must implement _get_default_base_url")

    def _get_provider_name(self) -> str:
        """Get the provider name for this implementation.

        Subclasses must implement this to return their name.
        """
        raise NotImplementedError("Subclasses must implement _get_provider_name")

    @property
    def name(self) -> str:
        """Provider name."""
        return self._get_provider_name()

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    @property
    def timeout(self) -> int:
        """Request timeout in seconds."""
        return self._timeout

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
        """
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                timeout=request_timeout,
            )

            self._requests_made += 1
            if response.usage:
                self._prompt_tokens += response.usage.prompt_tokens
                self._completion_tokens += response.usage.completion_tokens
                self._tokens_used += response.usage.total_tokens

            # Extract response content
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            finish_reason = response.choices[0].finish_reason

            # Validate content
            if content is None or not content.strip():
                logger.error(
                    f"{self.name} returned empty content "
                    f"(finish_reason={finish_reason}, tokens={tokens})"
                )
                raise RuntimeError(
                    f"LLM returned empty response (finish_reason={finish_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
                )

            # Check for truncation
            if finish_reason == "length":
                usage_info = ""
                if response.usage:
                    usage_info = (
                        f" (prompt={response.usage.prompt_tokens:,}, "
                        f"completion={response.usage.completion_tokens:,})"
                    )

                raise RuntimeError(
                    f"LLM response truncated - token limit exceeded{usage_info}. "
                    f"For reasoning models, this indicates the query requires "
                    f"extensive reasoning that exhausted the output budget. "
                    f"Try breaking your query into smaller, more focused questions."
                )

            # Warn on unexpected finish_reason
            if finish_reason not in ("stop",):
                logger.warning(
                    f"Unexpected finish_reason: {finish_reason} "
                    f"(content_length={len(content)})"
                )
                if finish_reason == "content_filter":
                    raise RuntimeError(
                        "LLM response blocked by content filter. "
                        "Try rephrasing your query or adjusting the prompt."
                    )

            return LLMResponse(
                content=content,
                tokens_used=tokens,
                model=self._model,
                finish_reason=finish_reason,
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

        Uses JSON Schema validation for structured outputs.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            Parsed JSON object conforming to schema
        """
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                timeout=request_timeout,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_response",
                        "strict": True,
                        "schema": json_schema,
                    },
                },
            )

            self._requests_made += 1
            if response.usage:
                self._prompt_tokens += response.usage.prompt_tokens
                self._completion_tokens += response.usage.completion_tokens
                self._tokens_used += response.usage.total_tokens

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Validate content
            if content is None or not content.strip():
                logger.error(
                    f"{self.name} structured completion returned empty content "
                    f"(finish_reason={finish_reason})"
                )
                raise RuntimeError(
                    f"LLM structured completion returned empty response "
                    f"(finish_reason={finish_reason})"
                )

            # Check for truncation
            if finish_reason == "length":
                usage_info = ""
                if response.usage:
                    usage_info = (
                        f" (prompt={response.usage.prompt_tokens:,}, "
                        f"completion={response.usage.completion_tokens:,})"
                    )

                raise RuntimeError(
                    f"LLM structured completion truncated - token limit "
                    f"exceeded{usage_info}. "
                    "This indicates insufficient max_completion_tokens for the "
                    "structured output. "
                    "Consider increasing the token limit or reducing input context."
                )

            # Parse JSON
            try:
                parsed = json.loads(content)
                return parsed
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse structured output as JSON: {e}")
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
        """Generate completions for multiple prompts concurrently."""
        tasks = [
            self.complete(prompt, system, max_completion_tokens) for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    def estimate_tokens(self, text: str) -> int:
        return estimate_tokens_llm(text)

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        try:
            response = await self.complete("Say 'OK'", max_completion_tokens=10)
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
            3 for OpenAI-compatible providers (conservative default)
        """
        return 3
