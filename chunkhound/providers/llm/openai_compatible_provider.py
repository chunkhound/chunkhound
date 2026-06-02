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
from chunkhound.core.utils.openai_utils import is_official_openai_endpoint
from chunkhound.core.utils.token_utils import estimate_tokens_llm
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse
from chunkhound.utils.json_extraction import (
    build_schema_system_instruction,
    parse_and_validate_structured_json,
)

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

    Set ``_supports_structured_outputs=False`` (class or instance) for
    providers whose APIs do not support OpenAI's native ``response_format``
    with ``type: "json_schema"`` (e.g. DeepSeek). The fallback injects the
    JSON schema into the system prompt instead.
    """

    _supports_structured_outputs: bool = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        base_url: str | None = None,
        ssl_verify: bool = True,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_retries: int = 3,
        supports_structured_outputs: bool | None = None,
    ):
        """Initialize OpenAI-compatible provider.

        Args:
            api_key: API key (defaults to environment variable)
            model: Model name
            base_url: Base URL (defaults to subclass implementation)
            ssl_verify: Verify TLS certificates for HTTP requests. When False, disables
                TLS verification for the resolved base URL. Ignored when no base URL is
                set (explicit or provider-resolved). Only disable for self-signed / local
                endpoints.
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
            supports_structured_outputs: Override class-level flag.
                When explicitly set, takes precedence over the class default.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI not available - install with: uv pip install openai"
            )

        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        if supports_structured_outputs is not None:
            self._supports_structured_outputs = supports_structured_outputs
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

    def _get_max_completion_tokens_param_name(self) -> str:
        """Get the output-token parameter name for chat completions requests."""
        return "max_completion_tokens"

    def _parse_structured_response(
        self, content: str, json_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse and validate a structured response against the JSON schema."""
        return parse_and_validate_structured_json(content, json_schema)

    def _build_chat_completion_kwargs(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int,
        timeout: int,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build chat-completions kwargs for subclasses to extend safely."""
        max_tokens_param = self._get_max_completion_tokens_param_name()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            max_tokens_param: max_completion_tokens,
            "timeout": timeout,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

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
                **self._build_chat_completion_kwargs(
                    messages,
                    max_completion_tokens,
                    request_timeout,
                )
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

            # Check for truncation FIRST (fix C2) so token-limit diagnostics
            # are not masked by a generic empty-content error.
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

            # Validate content (reached only if answer is complete)
            if content is None or not content.strip():
                logger.error(
                    f"{self.name} returned empty content "
                    f"(finish_reason={finish_reason}, tokens={tokens})"
                )
                raise RuntimeError(
                    f"LLM returned empty response (finish_reason={finish_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
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
        except RuntimeError:
            raise
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

        Uses native ``response_format`` with ``type: "json_schema"`` when the
        provider supports it (``_supports_structured_outputs = True``).
        Otherwise injects the schema into the system prompt and parses the
        free-form text response — ``extract_json_from_response`` handles
        markdown code blocks and raw JSON alike.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            Parsed JSON object conforming to schema
        """
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            if self._supports_structured_outputs:
                # Native structured outputs path
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                response = await self._client.chat.completions.create(
                    **self._build_chat_completion_kwargs(
                        messages,
                        max_completion_tokens,
                        request_timeout,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_response",
                                "strict": True,
                                "schema": json_schema,
                            },
                        },
                    )
                )
            else:
                logger.debug(
                    f"{self.name}: using prompt-based structured output fallback "
                    "(provider does not support native json_schema response_format)"
                )
                # Prompt-based fallback: inject schema into system message
                schema_instruction = build_schema_system_instruction(json_schema)
                if system:
                    effective_system = f"{system}\n\n{schema_instruction}"
                else:
                    effective_system = schema_instruction

                messages = [
                    {"role": "system", "content": effective_system},
                    {"role": "user", "content": prompt},
                ]

                response = await self._client.chat.completions.create(
                    **self._build_chat_completion_kwargs(
                        messages,
                        max_completion_tokens,
                        request_timeout,
                        response_format={"type": "json_object"},
                    )
                )

            self._requests_made += 1
            if response.usage:
                self._prompt_tokens += response.usage.prompt_tokens
                self._completion_tokens += response.usage.completion_tokens
                self._tokens_used += response.usage.total_tokens

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Check for truncation FIRST (more actionable — includes token counts)
            # Must precede the empty-content check: when finish_reason="length" and
            # content is empty/whitespace, the token-count info points at the fix.
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

            # Validate content (only reached if finish_reason != "length")
            if content is None or not content.strip():
                logger.error(
                    f"{self.name} structured completion returned empty content "
                    f"(finish_reason={finish_reason})"
                )
                raise RuntimeError(
                    f"LLM structured completion returned empty response "
                    f"(finish_reason={finish_reason})"
                )

            # Parse JSON
            try:
                return self._parse_structured_response(content, json_schema)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse structured output as JSON: {e}")
                raise RuntimeError(f"Invalid JSON in structured output: {e}") from e

        except RuntimeError:
            raise
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
