"""OpenAI LLM provider implementation for ChunkHound deep research."""

import asyncio
import json
from typing import Any

from loguru import logger

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available - install with: uv pip install openai")


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider using GPT models.

    Supports both Chat Completions API and Responses API:
    - Chat Completions (/v1/chat/completions): Standard models
    - Responses API (/v1/responses): Newer models with agentic capabilities

    Strategy: Prefer Responses API for all compatible models (it's a superset of Chat Completions)
    """

    # Models that ONLY support Responses API (from OpenAI spec: ResponsesOnlyModel)
    # These models will fail if you try to use Chat Completions API
    RESPONSES_ONLY_MODELS = {
        "o1-pro",
        "o3-pro",
        "o3-deep-research",
        "o4-mini-deep-research",
        "computer-use-preview",
        "gpt-5-codex",
        "gpt-5-pro",
    }

    # Models that support both APIs but Responses is preferred (from OpenAI spec + docs)
    # Responses API is a superset with agentic capabilities
    RESPONSES_PREFERRED_MODELS = {
        # GPT-5 series (all support Responses)
        "gpt-5.1",
        "gpt-5.1-codex",  # Also in RESPONSES_ONLY but safe to have both
        "gpt-5.1-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        # GPT-4.1 series
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        # GPT-4o series
        "gpt-4o",
        "gpt-4o-mini",
        # o-series reasoning models
        "o1",
        "o1-preview",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5-nano-mini",
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        reasoning_effort: str | None = None,
    ):
        """Initialize OpenAI LLM provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name to use
            base_url: Base URL for OpenAI API (optional for custom endpoints)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
            reasoning_effort: Reasoning effort for reasoning models (none, minimal, low, medium, high)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available - install with: uv pip install openai")

        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._reasoning_effort = reasoning_effort

        # Initialize client
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    def _should_use_responses_api(self) -> bool:
        """Check if the model should use Responses API instead of Chat Completions.

        Returns:
            True if model should use /v1/responses endpoint
        """
        # Check exact matches against Responses-only models (MUST use Responses)
        if self._model in self.RESPONSES_ONLY_MODELS:
            return True

        # Check exact matches against Responses-preferred models (SHOULD use Responses)
        if self._model in self.RESPONSES_PREFERRED_MODELS:
            return True

        # Check prefixes for dated model snapshots (e.g., "gpt-5.1-2025-11-13")
        all_responses_models = self.RESPONSES_ONLY_MODELS | self.RESPONSES_PREFERRED_MODELS
        for base_model in all_responses_models:
            if self._model.startswith(base_model + "-"):
                return True

        return False

    async def _complete_with_responses_api(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        """Generate a completion using the Responses API for reasoning models.

        Args:
            prompt: The user prompt
            system: Optional system/developer message
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            LLMResponse with content and metadata
        """
        request_timeout = timeout if timeout is not None else self._timeout

        # Build request parameters for Responses API
        request_params: dict[str, Any] = {
            "model": self._model,
            "input": prompt,  # Responses API uses 'input' instead of 'messages'
            "max_output_tokens": max_completion_tokens,  # Different parameter name
            "timeout": request_timeout,
        }

        # Add system instructions if provided
        if system:
            request_params["instructions"] = system

        # Add reasoning configuration if specified
        if self._reasoning_effort:
            request_params["reasoning"] = {"effort": self._reasoning_effort}

        try:
            # Call Responses API
            response = await self._client.responses.create(**request_params)

            self._requests_made += 1
            if response.usage:
                # Responses API uses input_tokens/output_tokens instead of prompt_tokens/completion_tokens
                self._prompt_tokens += response.usage.input_tokens
                self._completion_tokens += response.usage.output_tokens
                self._tokens_used += response.usage.total_tokens

            # Extract response content from output items
            content_parts = []
            for item in response.output:
                if item.type == "message":
                    # Message item contains the actual response text
                    for content_item in item.content:
                        # Responses API uses "output_text" type
                        if content_item.type == "output_text" and hasattr(content_item, 'text'):
                            content_parts.append(content_item.text)

            content = "\n".join(content_parts) if content_parts else None

            tokens = response.usage.total_tokens if response.usage else 0
            finish_reason = response.status  # Responses API uses 'status' instead of 'finish_reason'

            # Validate content is not None or empty
            if content is None:
                logger.error(
                    f"OpenAI Responses API returned None content (status={finish_reason}, "
                    f"tokens={tokens})"
                )
                raise RuntimeError(
                    f"LLM returned empty response (status={finish_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
                )

            if not content.strip():
                logger.warning(
                    f"OpenAI Responses API returned empty content (status={finish_reason}, "
                    f"tokens={tokens})"
                )
                raise RuntimeError(
                    f"LLM returned empty response (status={finish_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
                )

            # Check for incomplete responses
            if finish_reason == "incomplete":
                usage_info = ""
                if response.usage:
                    usage_info = (
                        f" (input={response.usage.input_tokens:,}, "
                        f"output={response.usage.output_tokens:,})"
                    )

                raise RuntimeError(
                    f"LLM response incomplete - token limit exceeded{usage_info}. "
                    f"For reasoning models, this indicates the query requires extensive reasoning "
                    f"that exhausted the output budget. Try breaking your query into smaller, "
                    f"more focused questions."
                )

            # Warn on other unexpected status
            if finish_reason not in ("completed", "complete"):
                logger.warning(
                    f"Unexpected status: {finish_reason} "
                    f"(content_length={len(content)})"
                )

            return LLMResponse(
                content=content,
                tokens_used=tokens,
                model=self._model,
                finish_reason=finish_reason,
            )

        except Exception as e:
            logger.error(f"OpenAI Responses API completion failed: {e}")
            raise RuntimeError(f"LLM completion failed: {e}") from e

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Automatically routes to the appropriate API:
        - Responses API for reasoning models (gpt-5.1, o-series, etc.)
        - Chat Completions API for standard models

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)
        """
        # Route to Responses API for compatible models
        if self._should_use_responses_api():
            logger.debug(f"Using Responses API for model: {self._model}")
            return await self._complete_with_responses_api(
                prompt, system, max_completion_tokens, timeout
            )

        # Use Chat Completions API for standard models
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Use provided timeout or fall back to default
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

            # Extract response metadata
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            finish_reason = response.choices[0].finish_reason

            # Validate content is not None or empty
            if content is None:
                logger.error(
                    f"OpenAI returned None content (finish_reason={finish_reason}, "
                    f"tokens={tokens})"
                )
                raise RuntimeError(
                    f"LLM returned empty response (finish_reason={finish_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
                )

            if not content.strip():
                logger.warning(
                    f"OpenAI returned empty content (finish_reason={finish_reason}, "
                    f"tokens={tokens})"
                )
                raise RuntimeError(
                    f"LLM returned empty response (finish_reason={finish_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
                )

            # Reject truncated responses (finish_reason="length")
            if finish_reason == "length":
                usage_info = ""
                if response.usage:
                    usage_info = (
                        f" (prompt={response.usage.prompt_tokens:,}, "
                        f"completion={response.usage.completion_tokens:,})"
                    )

                raise RuntimeError(
                    f"LLM response truncated - token limit exceeded{usage_info}. "
                    f"For reasoning models (GPT-5, Gemini 2.5), this indicates the query requires "
                    f"extensive reasoning that exhausted the output budget. "
                    f"The output budget is fixed at {max_completion_tokens:,} tokens for all queries "
                    f"to accommodate internal 'thinking' tokens (OUTPUT_TOKENS_WITH_REASONING). "
                    f"Try breaking your query into smaller, more focused questions."
                )

            # Warn on other unexpected finish_reason
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
            logger.error(f"OpenAI completion failed: {e}")
            raise RuntimeError(f"LLM completion failed: {e}") from e

    async def _complete_structured_with_responses_api(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Generate structured JSON using Responses API.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system/developer message
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            Parsed JSON object conforming to schema
        """
        request_timeout = timeout if timeout is not None else self._timeout

        # Build request parameters for Responses API structured output
        request_params: dict[str, Any] = {
            "model": self._model,
            "input": prompt,
            "max_output_tokens": max_completion_tokens,
            "timeout": request_timeout,
            # Responses API uses text.format for structured outputs
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "structured_response",
                    "strict": True,
                    "schema": json_schema,
                }
            },
        }

        # Add system instructions if provided
        if system:
            request_params["instructions"] = system

        # Add reasoning configuration if specified
        if self._reasoning_effort:
            request_params["reasoning"] = {"effort": self._reasoning_effort}

        try:
            response = await self._client.responses.create(**request_params)

            self._requests_made += 1
            if response.usage:
                self._prompt_tokens += response.usage.input_tokens
                self._completion_tokens += response.usage.output_tokens
                self._tokens_used += response.usage.total_tokens

            # Extract JSON content from output items
            content_parts = []
            for item in response.output:
                if item.type == "message":
                    for content_item in item.content:
                        if content_item.type == "output_text" and hasattr(content_item, 'text'):
                            content_parts.append(content_item.text)

            content = "\n".join(content_parts) if content_parts else None
            finish_reason = response.status

            # Validate content
            if content is None or not content.strip():
                logger.error(
                    f"Responses API structured output returned empty content (status={finish_reason})"
                )
                raise RuntimeError(
                    f"LLM structured output returned empty response (status={finish_reason})"
                )

            # Check for incomplete responses
            if finish_reason == "incomplete":
                usage_info = ""
                if response.usage:
                    usage_info = (
                        f" (input={response.usage.input_tokens:,}, "
                        f"output={response.usage.output_tokens:,})"
                    )
                raise RuntimeError(
                    f"LLM structured output incomplete - token limit exceeded{usage_info}"
                )

            # Parse JSON
            parsed = json.loads(content)
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Responses API structured output as JSON: {e}")
            raise RuntimeError(f"Invalid JSON in structured output: {e}") from e
        except Exception as e:
            logger.error(f"Responses API structured completion failed: {e}")
            raise RuntimeError(f"LLM structured completion failed: {e}") from e

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON completion conforming to the given schema.

        Automatically routes to the appropriate API:
        - Responses API for reasoning models (gpt-5.1, o-series, etc.)
        - Chat Completions API for standard models

        Uses OpenAI's structured outputs with strict JSON Schema validation.
        Best practice for GPT-5-Nano: Guarantees valid, parseable JSON output.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            Parsed JSON object conforming to schema
        """
        # Route to Responses API for compatible models
        if self._should_use_responses_api():
            logger.debug(f"Using Responses API for structured output with model: {self._model}")
            return await self._complete_structured_with_responses_api(
                prompt, json_schema, system, max_completion_tokens, timeout
            )

        # Use Chat Completions API for standard models
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Use provided timeout or fall back to default
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

            # Reject truncated responses (finish_reason="length")
            if finish_reason == "length":
                usage_info = ""
                if response.usage:
                    usage_info = (
                        f" (prompt={response.usage.prompt_tokens:,}, "
                        f"completion={response.usage.completion_tokens:,})"
                    )

                raise RuntimeError(
                    f"LLM structured completion truncated - token limit exceeded{usage_info}. "
                    f"This indicates insufficient max_completion_tokens for the structured output. "
                    f"Consider increasing the token limit or reducing input context."
                )

            # Validate content is not None or empty
            if content is None or not content.strip():
                logger.error(
                    f"OpenAI structured completion returned empty content "
                    f"(finish_reason={finish_reason})"
                )
                raise RuntimeError(
                    f"LLM structured completion returned empty response "
                    f"(finish_reason={finish_reason})"
                )

            parsed = json.loads(content)

            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output as JSON: {e}")
            raise RuntimeError(f"Invalid JSON in structured output: {e}") from e
        except Exception as e:
            logger.error(f"OpenAI structured completion failed: {e}")
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
        """Estimate token count for text (rough approximation)."""
        # Rough estimation: ~4 chars per token for GPT models
        return len(text) // 4

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        try:
            response = await self.complete("Say 'OK'", max_completion_tokens=10)
            return {
                "status": "healthy",
                "provider": "openai",
                "model": self._model,
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "openai",
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
            3 for OpenAI (conservative default based on tier limits)
        """
        return 3
