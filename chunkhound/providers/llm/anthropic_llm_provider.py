"""Anthropic LLM provider implementation for ChunkHound deep research.

Supports Claude 4.5 generation models including Opus 4.5 with advanced features:
- Extended thinking with configurable budget
- Interleaved thinking for sophisticated tool use reasoning
- Effort parameter for token usage vs thoroughness tradeoff (Opus 4.5 only)
- Context management for automatic tool result and thinking block clearing
"""

import asyncio
from typing import Any

from loguru import logger

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse

try:
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    AsyncAnthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic package not available")


# Beta headers for various features
BETA_EFFORT = "effort-2025-11-24"
BETA_CONTEXT_MANAGEMENT = "context-management-2025-06-27"
BETA_INTERLEAVED_THINKING = "interleaved-thinking-2025-05-14"

# Models that support effort parameter
EFFORT_SUPPORTED_MODELS = {"claude-opus-4-5-20251101"}


class AnthropicLLMProvider(LLMProvider):
    """Anthropic LLM provider using Claude models.

    Supports extended thinking, tool use, and advanced Opus 4.5 features.

    Extended Thinking:
        - Enables Claude to show reasoning process
        - Supported models: All Claude 4.5 models (Opus 4.5, Sonnet 4.5, Haiku 4.5)
          and legacy models (Opus 4.1, Opus 4, Sonnet 4, Sonnet 3.7)
        - Returns thinking, redacted_thinking, and text content blocks
        - Thinking budget configurable (min 1024 tokens)

    Interleaved Thinking (Claude 4 models):
        - Enables thinking between tool calls
        - More sophisticated reasoning after receiving tool results
        - Requires beta header: interleaved-thinking-2025-05-14

    Effort Parameter (Opus 4.5 only):
        - Controls token usage vs thoroughness tradeoff
        - Levels: high (default), medium, low
        - Affects all tokens: text, tool calls, and thinking
        - Requires beta header: effort-2025-11-24

    Context Management:
        - Automatic clearing of tool results and thinking blocks
        - Helps manage long conversations within context limits
        - Requires beta header: context-management-2025-06-27

    Tool Use:
        - Client tools: User-defined, executed on client side
        - Server tools: Anthropic-hosted (e.g., web search)
        - Automatic tool request/result handling
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        thinking_enabled: bool = False,
        thinking_budget_tokens: int = 10000,
        interleaved_thinking: bool = False,
        effort: str | None = None,
        context_management_enabled: bool = False,
        clear_thinking_keep_turns: int | None = None,
        clear_tool_uses_trigger_tokens: int | None = None,
        clear_tool_uses_keep: int | None = None,
    ):
        """Initialize Anthropic LLM provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name to use
                Latest models (Claude 4.5 generation):
                - claude-opus-4-5-20251101: Most capable model with effort control
                - claude-sonnet-4-5-20250929: Smartest model for complex agents and coding
                - claude-haiku-4-5-20251001: Fastest model with near-frontier intelligence
                Legacy models:
                - claude-opus-4-1-20250805: Exceptional model for specialized reasoning
            base_url: Base URL for Anthropic API (optional for custom endpoints)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
            thinking_enabled: Enable extended thinking (shows reasoning process)
            thinking_budget_tokens: Token budget for thinking (min 1024, recommend 10000)
            interleaved_thinking: Enable thinking between tool calls (Claude 4 only)
            effort: Token usage level - "low", "medium", or "high" (Opus 4.5 only)
            context_management_enabled: Enable automatic context management
            clear_thinking_keep_turns: Number of thinking turns to preserve (None=all)
            clear_tool_uses_trigger_tokens: Token threshold to trigger tool clearing
            clear_tool_uses_keep: Number of tool use pairs to keep after clearing
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not available")

        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._thinking_enabled = thinking_enabled
        self._thinking_budget_tokens = max(1024, thinking_budget_tokens)
        self._interleaved_thinking = interleaved_thinking

        # Effort parameter (Opus 4.5 only)
        self._effort = effort
        if effort and model not in EFFORT_SUPPORTED_MODELS:
            logger.warning(
                f"Effort parameter is only supported on Opus 4.5. "
                f"Model {model} will ignore effort={effort}"
            )

        # Context management settings
        self._context_management_enabled = context_management_enabled
        self._clear_thinking_keep_turns = clear_thinking_keep_turns
        self._clear_tool_uses_trigger_tokens = clear_tool_uses_trigger_tokens
        self._clear_tool_uses_keep = clear_tool_uses_keep

        # Initialize client
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncAnthropic(**client_kwargs)

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._thinking_tokens = 0  # Track full thinking tokens (billed)

    @property
    def name(self) -> str:
        """Provider name."""
        return "anthropic"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    def _extract_text_from_content(self, content_blocks: list[Any]) -> str:
        """Extract text from content blocks.

        Handles multiple content block types:
        - text: Regular text response
        - thinking: Claude's reasoning process
        - redacted_thinking: Encrypted reasoning (safety)
        - tool_use: Tool invocation request

        Args:
            content_blocks: List of content blocks from Anthropic response

        Returns:
            Concatenated text from all text blocks
        """
        text_parts = []

        for block in content_blocks:
            block_type = getattr(block, "type", None)

            if block_type == "text":
                # Standard text response
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            elif block_type == "thinking":
                # Extended thinking block
                if hasattr(block, "thinking"):
                    # Optionally include thinking in output for debugging
                    # For now, we skip including it in the final response
                    # Users can enable this via a flag if needed
                    pass
            elif block_type == "redacted_thinking":
                # Redacted thinking (encrypted for safety)
                # Never include in output, but preserve for multi-turn
                pass
            elif block_type == "tool_use":
                # Tool invocation - not included in text output
                # Should be handled separately for tool use workflows
                pass
            else:
                # Unknown block type - log warning
                logger.warning(f"Unknown content block type: {block_type}")

        return "".join(text_parts)

    def _get_thinking_blocks(self, content_blocks: list[Any]) -> list[dict[str, Any]]:
        """Extract thinking blocks for preservation in multi-turn conversations.

        Critical: Thinking blocks must be preserved unmodified when passing
        back to API in multi-turn conversations to maintain reasoning flow.

        Args:
            content_blocks: List of content blocks from Anthropic response

        Returns:
            List of thinking/redacted_thinking blocks with signatures
        """
        thinking_blocks = []

        for block in content_blocks:
            block_type = getattr(block, "type", None)

            if block_type in ("thinking", "redacted_thinking"):
                # Preserve complete block structure
                block_dict: dict[str, Any] = {"type": block_type}

                if block_type == "thinking":
                    if hasattr(block, "thinking"):
                        block_dict["thinking"] = block.thinking
                    if hasattr(block, "signature"):
                        block_dict["signature"] = block.signature
                elif block_type == "redacted_thinking":
                    if hasattr(block, "data"):
                        block_dict["data"] = block.data

                thinking_blocks.append(block_dict)

        return thinking_blocks

    def _get_beta_headers(self) -> list[str]:
        """Build list of beta headers based on enabled features.

        Returns:
            List of beta header strings to include in API requests
        """
        headers = []

        # Effort parameter (Opus 4.5 only)
        if self._effort and self._model in EFFORT_SUPPORTED_MODELS:
            headers.append(BETA_EFFORT)

        # Context management
        if self._context_management_enabled:
            headers.append(BETA_CONTEXT_MANAGEMENT)

        # Interleaved thinking
        if self._interleaved_thinking and self._thinking_enabled:
            headers.append(BETA_INTERLEAVED_THINKING)

        return headers

    def _build_context_management(
        self, thinking_active: bool | None = None
    ) -> dict[str, Any] | None:
        """Build context management configuration if enabled.

        Args:
            thinking_active: Override for whether thinking is active in this request.
                If None, uses self._thinking_enabled. Pass False for requests
                where thinking is skipped (e.g., complete_structured with forced tool choice).

        Returns:
            Context management dict for API request, or None if disabled
        """
        if not self._context_management_enabled:
            return None

        # Determine if thinking is actually active for this request
        is_thinking_active = (
            thinking_active if thinking_active is not None else self._thinking_enabled
        )

        edits: list[dict[str, Any]] = []

        # Thinking block clearing (must come first per API docs)
        if is_thinking_active:
            thinking_edit: dict[str, Any] = {"type": "clear_thinking_20251015"}
            if self._clear_thinking_keep_turns is not None:
                thinking_edit["keep"] = {
                    "type": "thinking_turns",
                    "value": self._clear_thinking_keep_turns,
                }
            else:
                # Keep all thinking blocks for cache optimization
                thinking_edit["keep"] = "all"
            edits.append(thinking_edit)

        # Tool result clearing
        tool_edit: dict[str, Any] = {"type": "clear_tool_uses_20250919"}
        if self._clear_tool_uses_trigger_tokens is not None:
            tool_edit["trigger"] = {
                "type": "input_tokens",
                "value": self._clear_tool_uses_trigger_tokens,
            }
        if self._clear_tool_uses_keep is not None:
            tool_edit["keep"] = {
                "type": "tool_uses",
                "value": self._clear_tool_uses_keep,
            }
        edits.append(tool_edit)

        return {"edits": edits}

    def _build_output_config(self) -> dict[str, Any] | None:
        """Build output config for effort parameter (Opus 4.5 only).

        Returns:
            Output config dict for API request, or None if not applicable
        """
        if self._effort and self._model in EFFORT_SUPPORTED_MODELS:
            return {"effort": self._effort}
        return None

    async def _create_message(self, request_kwargs: dict[str, Any]) -> Any:
        """Create a message using the appropriate endpoint (beta or standard).

        Uses the beta endpoint when beta features are enabled, otherwise
        uses the standard messages endpoint.

        Args:
            request_kwargs: Request parameters for the API call

        Returns:
            API response
        """
        beta_headers = request_kwargs.pop("betas", None)
        if beta_headers:
            return await self._client.beta.messages.create(
                betas=beta_headers, **request_kwargs
            )
        else:
            return await self._client.messages.create(**request_kwargs)

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
        # Build messages list (Anthropic separates system from messages)
        messages = [{"role": "user", "content": prompt}]

        # Use provided timeout or fall back to default
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            # Build request kwargs
            request_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_completion_tokens,
                "timeout": request_timeout,
            }

            # Add system prompt if provided (Anthropic uses separate system parameter)
            if system:
                request_kwargs["system"] = system

            # Add thinking configuration if enabled
            if self._thinking_enabled:
                # max_tokens must be greater than thinking.budget_tokens
                # Ensure max_tokens is at least budget + 1000 for actual response
                min_max_tokens = self._thinking_budget_tokens + 1000
                if max_completion_tokens < min_max_tokens:
                    logger.warning(
                        f"max_completion_tokens ({max_completion_tokens}) too small for "
                        f"thinking budget ({self._thinking_budget_tokens}). "
                        f"Increasing to {min_max_tokens}"
                    )
                    request_kwargs["max_tokens"] = min_max_tokens

                request_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self._thinking_budget_tokens,
                }

            # Add beta headers for advanced features
            beta_headers = self._get_beta_headers()
            if beta_headers:
                request_kwargs["betas"] = beta_headers

            # Add output config for effort parameter (Opus 4.5 only)
            output_config = self._build_output_config()
            if output_config:
                request_kwargs["output_config"] = output_config

            # Add context management if enabled
            context_management = self._build_context_management()
            if context_management:
                request_kwargs["context_management"] = context_management

            response = await self._create_message(request_kwargs)

            # Update usage statistics
            self._requests_made += 1
            if response.usage:
                self._prompt_tokens += response.usage.input_tokens
                self._completion_tokens += response.usage.output_tokens
                self._tokens_used += (
                    response.usage.input_tokens + response.usage.output_tokens
                )

            # Extract response content from content blocks
            content_blocks = response.content
            if not content_blocks:
                logger.error(
                    f"Anthropic returned no content blocks (stop_reason={response.stop_reason})"
                )
                raise RuntimeError(
                    f"LLM returned empty response (stop_reason={response.stop_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
                )

            # Extract text from content blocks
            content = self._extract_text_from_content(content_blocks)

            if not content.strip():
                logger.warning(
                    f"Anthropic returned empty text content (stop_reason={response.stop_reason})"
                )
                raise RuntimeError(
                    f"LLM returned empty text content (stop_reason={response.stop_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
                )

            # Check for truncated responses
            if response.stop_reason == "max_tokens":
                usage_info = ""
                if response.usage:
                    usage_info = (
                        f" (input={response.usage.input_tokens:,}, "
                        f"output={response.usage.output_tokens:,})"
                    )

                raise RuntimeError(
                    f"LLM response truncated - token limit exceeded{usage_info}. "
                    f"For reasoning models (Claude Opus, Sonnet), this indicates the query requires "
                    f"extensive reasoning that exhausted the output budget. "
                    f"The output budget is fixed at {max_completion_tokens:,} tokens. "
                    f"Try breaking your query into smaller, more focused questions."
                )

            # Warn on unexpected stop reasons
            if response.stop_reason not in ("end_turn", "stop_sequence"):
                logger.warning(
                    f"Unexpected stop_reason: {response.stop_reason} "
                    f"(content_length={len(content)})"
                )

            tokens_used = 0
            if response.usage:
                tokens_used = response.usage.input_tokens + response.usage.output_tokens

            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                model=self._model,
                finish_reason=response.stop_reason,
            )

        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
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

        Uses tool use with schema validation for reliable structured output.
        This is much more reliable than prompt engineering.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            Parsed JSON object conforming to schema
        """
        # Define a tool with the JSON schema as input
        tools = [
            {
                "name": "return_structured_data",
                "description": "Return structured data matching the required schema",
                "input_schema": json_schema,
            }
        ]

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Use provided timeout or fall back to default
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            # Build request kwargs
            request_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_completion_tokens,
                "timeout": request_timeout,
                "tools": tools,
                "tool_choice": {"type": "tool", "name": "return_structured_data"},
            }

            # Add system prompt if provided
            if system:
                request_kwargs["system"] = system

            # Note: Extended thinking is NOT compatible with forced tool choice
            # (tool_choice: {"type": "tool", "name": "..."})
            # We skip thinking config here since we force the tool
            if self._thinking_enabled:
                logger.debug(
                    "Skipping extended thinking for complete_structured - "
                    "forced tool choice is incompatible with thinking"
                )

            # Add beta headers for advanced features (excluding interleaved thinking)
            beta_headers = []
            if self._effort and self._model in EFFORT_SUPPORTED_MODELS:
                beta_headers.append(BETA_EFFORT)
            if self._context_management_enabled:
                beta_headers.append(BETA_CONTEXT_MANAGEMENT)
            if beta_headers:
                request_kwargs["betas"] = beta_headers

            # Add output config for effort parameter (Opus 4.5 only)
            output_config = self._build_output_config()
            if output_config:
                request_kwargs["output_config"] = output_config

            # Add context management if enabled (with thinking_active=False since
            # forced tool choice is incompatible with thinking)
            context_management = self._build_context_management(thinking_active=False)
            if context_management:
                request_kwargs["context_management"] = context_management

            response = await self._create_message(request_kwargs)

            # Update usage statistics
            self._requests_made += 1
            if response.usage:
                self._prompt_tokens += response.usage.input_tokens
                self._completion_tokens += response.usage.output_tokens
                self._tokens_used += (
                    response.usage.input_tokens + response.usage.output_tokens
                )

            # Extract tool use from content blocks
            tool_use_block = None
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    tool_use_block = block
                    break

            if not tool_use_block:
                raise RuntimeError(
                    "Model did not return tool use for structured output. "
                    f"Stop reason: {response.stop_reason}"
                )

            # Return the tool input as structured data
            return tool_use_block.input

        except Exception as e:
            logger.error(f"Anthropic structured completion failed: {e}")
            raise RuntimeError(f"LLM structured completion failed: {e}") from e

    async def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        tool_choice: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> tuple[LLMResponse, list[dict[str, Any]]]:
        """Generate a completion with tool use support.

        Args:
            prompt: The user prompt
            tools: List of tool definitions
                Each tool should have: name, description, input_schema
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            tool_choice: Optional tool choice configuration
                - {"type": "auto"}: Model decides (default)
                - {"type": "any"}: Model must use a tool
                - {"type": "tool", "name": "tool_name"}: Force specific tool
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            Tuple of (LLMResponse with text content, list of tool use requests)
            Tool use requests contain: id, name, input
        """
        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Use provided timeout or fall back to default
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            # Build request kwargs
            request_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_completion_tokens,
                "timeout": request_timeout,
                "tools": tools,
            }

            # Add tool choice if provided (default is auto)
            if tool_choice:
                request_kwargs["tool_choice"] = tool_choice

            # Add system prompt if provided
            if system:
                request_kwargs["system"] = system

            # Check if thinking is compatible with tool_choice
            # Thinking only works with tool_choice: auto (default) or none
            # tool_choice: any or tool are incompatible
            thinking_compatible = True
            if tool_choice:
                choice_type = tool_choice.get("type", "auto")
                if choice_type in ("any", "tool"):
                    thinking_compatible = False
                    if self._thinking_enabled:
                        logger.debug(
                            f"Skipping extended thinking - tool_choice={choice_type} "
                            "is incompatible with thinking"
                        )

            # Add thinking configuration if enabled and compatible
            if self._thinking_enabled and thinking_compatible:
                # max_tokens must be greater than thinking.budget_tokens
                # Ensure max_tokens is at least budget + 1000 for actual response
                min_max_tokens = self._thinking_budget_tokens + 1000
                if max_completion_tokens < min_max_tokens:
                    logger.warning(
                        f"max_completion_tokens ({max_completion_tokens}) too small for "
                        f"thinking budget ({self._thinking_budget_tokens}). "
                        f"Increasing to {min_max_tokens}"
                    )
                    request_kwargs["max_tokens"] = min_max_tokens

                request_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self._thinking_budget_tokens,
                }

            # Add beta headers for advanced features
            beta_headers = []
            if self._effort and self._model in EFFORT_SUPPORTED_MODELS:
                beta_headers.append(BETA_EFFORT)
            if self._context_management_enabled:
                beta_headers.append(BETA_CONTEXT_MANAGEMENT)
            # Only add interleaved thinking header if thinking is enabled and compatible
            if (
                self._interleaved_thinking
                and self._thinking_enabled
                and thinking_compatible
            ):
                beta_headers.append(BETA_INTERLEAVED_THINKING)
            if beta_headers:
                request_kwargs["betas"] = beta_headers

            # Add output config for effort parameter (Opus 4.5 only)
            output_config = self._build_output_config()
            if output_config:
                request_kwargs["output_config"] = output_config

            # Add context management if enabled
            context_management = self._build_context_management()
            if context_management:
                request_kwargs["context_management"] = context_management

            response = await self._create_message(request_kwargs)

            # Update usage statistics
            self._requests_made += 1
            if response.usage:
                self._prompt_tokens += response.usage.input_tokens
                self._completion_tokens += response.usage.output_tokens
                self._tokens_used += (
                    response.usage.input_tokens + response.usage.output_tokens
                )

            # Extract text content and tool uses
            content_blocks = response.content
            text_content = self._extract_text_from_content(content_blocks)

            # Extract tool use blocks
            tool_uses = []
            for block in content_blocks:
                if getattr(block, "type", None) == "tool_use":
                    tool_uses.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            tokens_used = 0
            if response.usage:
                tokens_used = response.usage.input_tokens + response.usage.output_tokens

            llm_response = LLMResponse(
                content=text_content,
                tokens_used=tokens_used,
                model=self._model,
                finish_reason=response.stop_reason,
            )

            return llm_response, tool_uses

        except Exception as e:
            logger.error(f"Anthropic tool use completion failed: {e}")
            raise RuntimeError(f"LLM tool use completion failed: {e}") from e

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

        Note: For accurate token counting, use the Anthropic SDK's
        count_tokens method. This is a rough estimation.
        """
        # Rough estimation: ~3.5 chars per token for Claude models
        return len(text) // 4

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        try:
            response = await self.complete("Say 'OK'", max_completion_tokens=10)
            result = {
                "status": "healthy",
                "provider": "anthropic",
                "model": self._model,
                "test_response": response.content[:50],
                "thinking_enabled": self._thinking_enabled,
                "interleaved_thinking": self._interleaved_thinking,
                "context_management_enabled": self._context_management_enabled,
            }
            # Only include effort if set and supported
            if self._effort and self._model in EFFORT_SUPPORTED_MODELS:
                result["effort"] = self._effort
            return result
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "anthropic",
                "error": str(e),
            }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "requests_made": self._requests_made,
            "total_tokens": self._tokens_used,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "thinking_tokens": self._thinking_tokens,
        }

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations.

        Returns:
            5 for Anthropic (higher tier limits than OpenAI)
        """
        return 5

    def supports_thinking(self) -> bool:
        """Whether provider supports extended thinking."""
        return True

    def supports_tools(self) -> bool:
        """Whether provider supports tool use."""
        return True

    def supports_streaming(self) -> bool:
        """Whether provider supports streaming responses."""
        return True

    async def complete_streaming(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ):
        """Generate a streaming completion for the given prompt.

        Yields text chunks as they arrive. Required for max_tokens > 21,333.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Yields:
            Text chunks as they are generated
        """
        # Build messages list
        messages = [{"role": "user", "content": prompt}]

        # Use provided timeout or fall back to default
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            # Build request kwargs
            request_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_completion_tokens,
                "timeout": request_timeout,
                "stream": True,
            }

            # Add system prompt if provided
            if system:
                request_kwargs["system"] = system

            # Add thinking configuration if enabled
            if self._thinking_enabled:
                # max_tokens must be greater than thinking.budget_tokens
                # Ensure max_tokens is at least budget + 1000 for actual response
                min_max_tokens = self._thinking_budget_tokens + 1000
                if max_completion_tokens < min_max_tokens:
                    logger.warning(
                        f"max_completion_tokens ({max_completion_tokens}) too small for "
                        f"thinking budget ({self._thinking_budget_tokens}). "
                        f"Increasing to {min_max_tokens}"
                    )
                    request_kwargs["max_tokens"] = min_max_tokens

                request_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self._thinking_budget_tokens,
                }

            # Add beta headers for advanced features
            beta_headers = self._get_beta_headers()
            if beta_headers:
                request_kwargs["betas"] = beta_headers

            # Add output config for effort parameter (Opus 4.5 only)
            output_config = self._build_output_config()
            if output_config:
                request_kwargs["output_config"] = output_config

            # Add context management if enabled
            context_management = self._build_context_management()
            if context_management:
                request_kwargs["context_management"] = context_management

            # Create streaming response - use beta endpoint when beta features are enabled
            beta_headers = request_kwargs.pop("betas", None)
            if beta_headers:
                stream = await self._client.beta.messages.create(
                    betas=beta_headers, **request_kwargs
                )
            else:
                stream = await self._client.messages.create(**request_kwargs)

            # Track if we've incremented request count
            request_counted = False

            # Stream events
            async for event in stream:
                # Count request on first event
                if not request_counted:
                    self._requests_made += 1
                    request_counted = True

                # Handle different event types
                event_type = getattr(event, "type", None)

                if event_type == "content_block_delta":
                    # Text delta from content block
                    delta = getattr(event, "delta", None)
                    if delta and hasattr(delta, "text"):
                        yield delta.text

                elif event_type == "message_stop":
                    # Update final usage statistics
                    # Note: usage info comes in message_start event
                    pass

                elif event_type == "message_start":
                    # Track usage from message start
                    message = getattr(event, "message", None)
                    if message and hasattr(message, "usage"):
                        self._prompt_tokens += message.usage.input_tokens
                        self._completion_tokens += message.usage.output_tokens
                        self._tokens_used += (
                            message.usage.input_tokens + message.usage.output_tokens
                        )

        except Exception as e:
            logger.error(f"Anthropic streaming completion failed: {e}")
            raise RuntimeError(f"LLM streaming completion failed: {e}") from e
