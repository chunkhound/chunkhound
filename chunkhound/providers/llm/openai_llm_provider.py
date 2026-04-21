"""OpenAI LLM provider implementation for ChunkHound deep research."""

import json
from typing import Any

from loguru import logger

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.interfaces.llm_provider import LLMResponse
from chunkhound.providers.llm.openai_compatible_provider import OpenAICompatibleProvider


class OpenAILLMProvider(OpenAICompatibleProvider):
    """OpenAI LLM provider using GPT models.

    Supports both Chat Completions API and Responses API:
    - Chat Completions (/v1/chat/completions): Standard models
    - Responses API (/v1/responses): Newer models with agentic capabilities

    Strategy: Prefer Responses API for all compatible models
    (it's a superset of Chat Completions)
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
        ssl_verify: bool = True,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_retries: int = 3,
        reasoning_effort: str | None = None,
    ):
        """Initialize OpenAI LLM provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name to use
            base_url: Base URL for OpenAI API (optional for custom endpoints)
            ssl_verify: Verify TLS certificates for requests sent via base_url
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
            reasoning_effort: Reasoning effort for reasoning models
                (none, minimal, low, medium, high)
        """
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            ssl_verify=ssl_verify,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._reasoning_effort = reasoning_effort

    def _get_default_base_url(self) -> str | None:
        """Get the default OpenAI API base URL.

        Returns None so AsyncOpenAI falls back to OPENAI_BASE_URL env var
        or its own default.
        """
        return None

    def _get_provider_name(self) -> str:
        """Get the provider name."""
        return "openai"

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
        all_responses_models = (
            self.RESPONSES_ONLY_MODELS | self.RESPONSES_PREFERRED_MODELS
        )
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
                # Responses API uses input_tokens/output_tokens instead of
                # prompt_tokens/completion_tokens
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
                        if content_item.type == "output_text" and hasattr(
                            content_item, "text"
                        ):
                            content_parts.append(content_item.text)

            content = "\n".join(content_parts) if content_parts else None

            tokens = response.usage.total_tokens if response.usage else 0
            finish_reason = (
                response.status
            )  # Responses API uses 'status' instead of 'finish_reason'

            # Validate content is not None or empty
            if content is None:
                logger.error(
                    f"OpenAI Responses API returned None content "
                    f"(status={finish_reason}, tokens={tokens})"
                )
                raise RuntimeError(
                    f"LLM returned empty response (status={finish_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
                )

            if not content.strip():
                logger.warning(
                    f"OpenAI Responses API returned empty content "
                    f"(status={finish_reason}, tokens={tokens})"
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
                    "For reasoning models, this indicates the query requires "
                    "extensive reasoning "
                    "that exhausted the output budget. Try breaking your query into "
                    "smaller, "
                    "more focused questions."
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

        except RuntimeError:
            raise
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

        # Use Chat Completions API for standard models via parent implementation
        return await super().complete(prompt, system, max_completion_tokens, timeout)

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
                        if content_item.type == "output_text" and hasattr(
                            content_item, "text"
                        ):
                            content_parts.append(content_item.text)

            content = "\n".join(content_parts) if content_parts else None
            finish_reason = response.status

            # Validate content
            if content is None or not content.strip():
                logger.error(
                    f"Responses API structured completion returned empty content "
                    f"(status={finish_reason})"
                )
                raise RuntimeError(
                    f"LLM structured completion returned empty response "
                    f"(status={finish_reason})"
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
                    f"LLM structured completion incomplete - token limit "
                    f"exceeded{usage_info}"
                )

            # Parse JSON
            parsed = json.loads(content)
            return parsed

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse Responses API structured output as JSON: {e}"
            )
            raise RuntimeError(f"Invalid JSON in structured output: {e}") from e
        except RuntimeError:
            raise
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
            logger.debug(
                f"Using Responses API for structured output with model: {self._model}"
            )
            return await self._complete_structured_with_responses_api(
                prompt, json_schema, system, max_completion_tokens, timeout
            )

        # Use Chat Completions API for standard models via parent implementation
        return await super().complete_structured(
            prompt, json_schema, system, max_completion_tokens, timeout
        )
