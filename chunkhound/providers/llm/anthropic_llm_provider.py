"""Anthropic LLM provider implementation for ChunkHound deep research.

Supports Claude 4.5, 4.6, 4.7, and Mythos generation models:
- Adaptive thinking (Opus 4.7, Opus 4.6, Sonnet 4.6, Mythos)
- Manual extended thinking with configurable budget_tokens (Opus 4.5 and older)
- Interleaved thinking for tool use (auto-enabled in adaptive mode)
- Effort parameter for token usage vs thoroughness tradeoff
- Automatic prompt caching
- Task budgets for agentic loops (Opus 4.7 beta)
- Context management for automatic tool result and thinking block clearing
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from chunkhound.core.utils import estimate_tokens_llm
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse

try:
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    AsyncAnthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic package not available")


# Beta headers
BETA_CONTEXT_MANAGEMENT = "context-management-2025-06-27"
BETA_INTERLEAVED_THINKING = "interleaved-thinking-2025-05-14"
BETA_STRUCTURED_OUTPUTS = "structured-outputs-2025-11-13"
BETA_TASK_BUDGETS = "task-budgets-2026-03-13"

# Model-family prefixes used for capability detection. Matches both the stable
# alias (e.g. "claude-opus-4-7") and dated variants (e.g. "claude-opus-4-7-20260416").
_EFFORT_FAMILY_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-mythos-preview",
)
_MAX_EFFORT_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-mythos-preview",
)
_XHIGH_EFFORT_PREFIXES: tuple[str, ...] = ("claude-opus-4-7",)
_ADAPTIVE_THINKING_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-mythos-preview",
)
_ADAPTIVE_ONLY_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-7",
    "claude-mythos-preview",
)
_TASK_BUDGET_PREFIXES: tuple[str, ...] = ("claude-opus-4-7",)


def _matches_family(model: str, prefixes: tuple[str, ...]) -> bool:
    """True if ``model`` matches any family prefix as an exact name or a
    prefix followed by a ``-`` delimiter.

    Guards against collisions with hypothetical future model names whose
    major-version component happens to begin with an existing prefix
    (e.g. a future ``claude-opus-4-50`` must not inherit Opus 4.5 flags).
    """
    return any(model == p or model.startswith(f"{p}-") for p in prefixes)


def supports_effort(model: str) -> bool:
    """True when the model accepts the output_config.effort parameter."""
    return _matches_family(model, _EFFORT_FAMILY_PREFIXES)


def supports_effort_level(model: str, level: str) -> bool:
    """True when the model supports the given effort level."""
    if level in ("low", "medium", "high"):
        return supports_effort(model)
    if level == "max":
        return _matches_family(model, _MAX_EFFORT_PREFIXES)
    if level == "xhigh":
        return _matches_family(model, _XHIGH_EFFORT_PREFIXES)
    return False


def supports_adaptive_thinking(model: str) -> bool:
    """True when the model accepts thinking.type == 'adaptive'."""
    return _matches_family(model, _ADAPTIVE_THINKING_PREFIXES)


def requires_adaptive_thinking(model: str) -> bool:
    """True when adaptive is the only supported thinking mode (rejects manual)."""
    return _matches_family(model, _ADAPTIVE_ONLY_PREFIXES)


def supports_task_budget(model: str) -> bool:
    """True when the model supports the task_budget beta."""
    return _matches_family(model, _TASK_BUDGET_PREFIXES)


class AnthropicLLMProvider(LLMProvider):
    """Anthropic LLM provider using Claude models.

    Supports adaptive/manual extended thinking, tool use, effort control, and
    automatic context management across Opus 4.5/4.6/4.7 and Sonnet 4.5/4.6.

    Thinking modes:
        - Adaptive (Opus 4.7, Opus 4.6, Sonnet 4.6, Mythos): Claude decides
          when to think. Interleaved thinking is auto-enabled. Opus 4.7 rejects
          manual mode.
        - Manual (Opus 4.5 and older): fixed thinking_budget_tokens.
        - Off: omit thinking.

    Effort parameter:
        - Supported on Opus 4.5, Opus 4.6, Opus 4.7, Sonnet 4.6, Mythos.
        - Levels: low, medium, high (default), max (4.6+), xhigh (Opus 4.7 only).
        - Affects all tokens: text, tool calls, and thinking.

    Interleaved thinking:
        - Auto-enabled in adaptive mode.
        - Manual mode on Sonnet 4.6: requires beta interleaved-thinking-2025-05-14.

    Context management:
        - Automatic clearing of tool results and thinking blocks.
        - Requires beta header: context-management-2025-06-27.

    Tool use:
        - Client tools (user-defined) and server tools (Anthropic-hosted).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
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
        thinking_mode: str | None = None,
        thinking_display: str | None = None,
        prompt_caching: bool = True,
        cache_ttl: str | None = None,
        task_budget_tokens: int | None = None,
    ):
        """Initialize Anthropic LLM provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name to use. Supported families:
                - claude-opus-4-7: adaptive thinking only. Effort levels
                  low/medium/high/xhigh/max. xhigh recommended for coding.
                - claude-opus-4-6: adaptive thinking. Effort low/medium/high/max.
                - claude-sonnet-4-6: adaptive thinking. Effort low/medium/high/max.
                - claude-opus-4-5-20251101: manual thinking, effort low/medium/high.
                - claude-sonnet-4-5-20250929: manual thinking, no effort.
                - claude-haiku-4-5-20251001: manual thinking, no effort.
            base_url: Base URL for Anthropic API (optional for custom endpoints)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
            thinking_enabled: Enable extended thinking. Combined with
                thinking_mode (or auto-selected when mode is None) to choose
                adaptive vs manual vs off.
            thinking_budget_tokens: Token budget for manual thinking
                (min 1024). Ignored in adaptive mode.
            interleaved_thinking: Enable thinking between tool calls in manual
                mode. Auto-enabled for adaptive mode regardless of this flag.
            effort: Token usage level - "low", "medium", "high", "xhigh"
                (Opus 4.7 only), or "max" (4.6+ only).
            context_management_enabled: Enable automatic context management
            clear_thinking_keep_turns: Number of thinking turns to preserve (None=all)
            clear_tool_uses_trigger_tokens: Token threshold to trigger tool clearing
            clear_tool_uses_keep: Number of tool use pairs to keep after clearing
            thinking_mode: Explicit thinking mode: "adaptive", "manual", "off",
                or "auto" / None for automatic selection based on model.
            thinking_display: "summarized" (default on 4.6) or "omitted"
                (default on Opus 4.7 / Mythos). Controls whether thinking text
                is returned in the response.
            prompt_caching: When True (default), sends a top-level
                cache_control={"type": "ephemeral"} so the Messages API caches
                the system prompt + message prefix automatically. Cache hits
                cost 10% of base input, writes cost 25% more for 5m TTL.
            cache_ttl: Cache TTL. None uses API default (5m). "1h" costs 2x
                writes but is useful when the same prefix is reused less
                often than every 5 minutes.
            task_budget_tokens: Total token budget across a full agentic loop
                (beta, Opus 4.7 only). Advisory cap visible to the
                model; min 20000. Leave None for open-ended quality work.
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not available")

        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._thinking_budget_tokens = max(1024, thinking_budget_tokens)
        self._interleaved_thinking = interleaved_thinking
        self._prompt_caching = prompt_caching
        self._cache_ttl = cache_ttl
        if task_budget_tokens is not None and not supports_task_budget(model):
            logger.warning(
                f"Model {model} does not support task_budget; ignoring "
                f"task_budget_tokens={task_budget_tokens}"
            )
            task_budget_tokens = None
        if task_budget_tokens is not None and task_budget_tokens < 20000:
            raise ValueError(
                f"task_budget_tokens ({task_budget_tokens}) is below the API "
                "minimum of 20000; set it to at least 20000 or pass None to "
                "disable."
            )
        self._task_budget_tokens = task_budget_tokens

        # Resolve the effective thinking mode. The historical API uses
        # thinking_enabled (bool); new callers can pass thinking_mode directly.
        requested_mode = (thinking_mode or "auto").lower()
        if requested_mode not in ("auto", "off", "manual", "adaptive"):
            raise ValueError(
                f"Unknown thinking_mode={thinking_mode!r}; "
                "expected one of auto, off, manual, adaptive"
            )

        if requested_mode in ("manual", "adaptive") and not thinking_enabled:
            raise ValueError(
                f"thinking_enabled=False conflicts with thinking_mode={requested_mode!r}. "
                "Pass thinking_enabled=True, or use thinking_mode='off' to disable."
            )

        if requested_mode == "off" and thinking_enabled:
            logger.debug(
                "thinking_mode='off' overrides thinking_enabled=True; thinking disabled."
            )

        if requested_mode == "off" or not thinking_enabled:
            resolved_mode = "off"
        elif requested_mode == "auto":
            resolved_mode = (
                "adaptive" if supports_adaptive_thinking(model) else "manual"
            )
        elif requested_mode == "manual" and requires_adaptive_thinking(model):
            logger.warning(
                f"Model {model} requires adaptive thinking; falling back to "
                "adaptive (thinking_budget_tokens is ignored in adaptive mode; "
                "thinking can consume up to max_tokens)"
            )
            resolved_mode = "adaptive"
        elif requested_mode == "adaptive" and not supports_adaptive_thinking(model):
            logger.warning(
                f"Model {model} does not support adaptive thinking; "
                "falling back to manual"
            )
            resolved_mode = "manual"
        else:
            resolved_mode = requested_mode

        self._thinking_mode = resolved_mode
        self._thinking_enabled = resolved_mode in ("manual", "adaptive")

        if thinking_display is not None and resolved_mode != "adaptive":
            logger.warning(
                f"thinking_display={thinking_display!r} only applies in adaptive "
                f"mode; resolved mode is {resolved_mode!r}, ignoring."
            )
            thinking_display = None
        self._thinking_display = thinking_display

        if effort is not None:
            effort = effort.strip().lower()
            if not supports_effort(model):
                logger.warning(
                    f"Model {model} does not accept the effort parameter; "
                    f"ignoring effort={effort!r}"
                )
                effort = None
            elif not supports_effort_level(model, effort):
                logger.warning(
                    f"Effort level {effort!r} is not supported on {model}; "
                    "dropping it (request would otherwise be rejected by the API)"
                )
                effort = None
        self._effort = effort

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

    def _get_beta_headers(self, thinking_active: bool | None = None) -> list[str]:
        """Build list of beta headers for the enabled features.

        Args:
            thinking_active: Override whether thinking is active for this call.
                The interleaved-thinking beta header is only emitted when
                manual-mode thinking is active for the request.

        Returns:
            Headers for context management, manual-mode interleaved thinking,
            and task budgets, in that order. Adaptive mode auto-enables
            interleaved thinking without a header.
        """
        headers: list[str] = []

        if self._context_management_enabled:
            headers.append(BETA_CONTEXT_MANAGEMENT)

        is_thinking_active = (
            thinking_active if thinking_active is not None else self._thinking_enabled
        )
        if (
            self._interleaved_thinking
            and self._thinking_mode == "manual"
            and is_thinking_active
        ):
            headers.append(BETA_INTERLEAVED_THINKING)

        if self._task_budget_tokens is not None:
            headers.append(BETA_TASK_BUDGETS)

        return headers

    def _build_thinking_config(
        self, thinking_active: bool | None = None
    ) -> dict[str, Any] | None:
        """Build the thinking request block for the configured mode.

        Args:
            thinking_active: Override whether thinking is active for this call.
                Pass False to force thinking off even when the provider has
                it enabled (e.g. when tool_choice is incompatible with
                extended thinking).

        Returns:
            dict shaped for the Messages API, or None when thinking is off.
        """
        is_active = (
            thinking_active if thinking_active is not None else self._thinking_enabled
        )
        if not is_active or self._thinking_mode == "off":
            return None

        if self._thinking_mode == "adaptive":
            cfg: dict[str, Any] = {"type": "adaptive"}
            if self._thinking_display:
                cfg["display"] = self._thinking_display
            return cfg

        # Manual mode
        return {
            "type": "enabled",
            "budget_tokens": self._thinking_budget_tokens,
        }

    def _ensure_thinking_max_tokens(
        self, request_kwargs: dict[str, Any], max_completion_tokens: int
    ) -> None:
        """Raise max_tokens to accommodate the manual thinking budget.

        Only meaningful in manual mode. Callers must gate invocation on
        ``self._thinking_mode == "manual"`` -- adaptive thinking has no fixed
        budget and uses max_tokens directly.
        """
        min_max_tokens = self._thinking_budget_tokens + 1000
        if max_completion_tokens < min_max_tokens:
            logger.warning(
                f"max_completion_tokens ({max_completion_tokens}) too small for "
                f"thinking budget ({self._thinking_budget_tokens}). "
                f"Increasing to {min_max_tokens}"
            )
            request_kwargs["max_tokens"] = min_max_tokens

    def _build_context_management(
        self, thinking_active: bool | None = None
    ) -> dict[str, Any] | None:
        """Build context management configuration if enabled.

        Args:
            thinking_active: Override for whether thinking is active in this request.
                If None, uses self._thinking_enabled. Pass False for requests
                where thinking is explicitly disabled.

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

    def _build_output_config(
        self, json_schema: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Build output_config with effort, structured-output format, and task budget.

        Effort is applied only when the model supports it. task_budget is
        applied only when task_budget_tokens was accepted in __init__ (which
        requires an Opus 4.7-family model).

        Args:
            json_schema: Optional JSON schema for structured outputs.

        Returns:
            output_config dict or None when empty.
        """
        cfg: dict[str, Any] = {}
        if self._effort and supports_effort_level(self._model, self._effort):
            cfg["effort"] = self._effort
        if json_schema is not None:
            cfg["format"] = {"type": "json_schema", "schema": json_schema}
        if self._task_budget_tokens is not None:
            cfg["task_budget"] = {
                "type": "tokens",
                "total": self._task_budget_tokens,
            }
        return cfg or None

    def _build_cache_control(self) -> dict[str, Any] | None:
        """Build the top-level cache_control for automatic prompt caching."""
        if not self._prompt_caching:
            return None
        cc: dict[str, Any] = {"type": "ephemeral"}
        if self._cache_ttl:
            cc["ttl"] = self._cache_ttl
        return cc

    def _apply_common_request_fields(
        self,
        request_kwargs: dict[str, Any],
        json_schema: dict[str, Any] | None = None,
        thinking_active: bool | None = None,
    ) -> None:
        """Attach output_config, cache_control, and context_management.

        Pass thinking_active=False when the caller disabled thinking for this
        specific request (e.g. tool_choice forced it off) so
        context_management does not emit a clear_thinking edit.
        """
        output_config = self._build_output_config(json_schema=json_schema)
        if output_config:
            request_kwargs["output_config"] = output_config
        cache_control = self._build_cache_control()
        if cache_control:
            request_kwargs["cache_control"] = cache_control
        context_management = self._build_context_management(
            thinking_active=thinking_active,
        )
        if context_management:
            request_kwargs["context_management"] = context_management

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

            # Thinking configuration (adaptive / manual / off)
            thinking_cfg = self._build_thinking_config()
            thinking_active = thinking_cfg is not None
            if thinking_cfg is not None:
                if self._thinking_mode == "manual":
                    self._ensure_thinking_max_tokens(
                        request_kwargs, max_completion_tokens
                    )
                request_kwargs["thinking"] = thinking_cfg

            beta_headers = self._get_beta_headers(thinking_active=thinking_active)
            if beta_headers:
                request_kwargs["betas"] = beta_headers

            self._apply_common_request_fields(
                request_kwargs, thinking_active=thinking_active
            )

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

        Uses Anthropic's native structured outputs with constrained decoding.
        This guarantees schema-compliant JSON through grammar-level constraints.

        Features:
        - Guaranteed valid JSON via constrained decoding (no parsing errors)
        - Compatible with extended thinking (grammar resets between sections)
        - Compatible with the effort parameter on models that support it
        - Compatible with context management

        Note: Incompatible with citations and message prefilling.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            Parsed JSON object conforming to schema

        Raises:
            RuntimeError: If response is refused, truncated, or contains invalid JSON
        """
        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Use provided timeout or fall back to default
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            # Build request kwargs for native structured outputs
            request_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_completion_tokens,
                "timeout": request_timeout,
            }

            # Add system prompt if provided
            if system:
                request_kwargs["system"] = system

            # Thinking configuration (adaptive / manual / off)
            thinking_cfg = self._build_thinking_config()
            thinking_active = thinking_cfg is not None
            if thinking_cfg is not None:
                if self._thinking_mode == "manual":
                    self._ensure_thinking_max_tokens(
                        request_kwargs, max_completion_tokens
                    )
                request_kwargs["thinking"] = thinking_cfg

            beta_headers = self._get_beta_headers(thinking_active=thinking_active)
            if beta_headers:
                request_kwargs["betas"] = beta_headers

            self._apply_common_request_fields(
                request_kwargs,
                json_schema=json_schema,
                thinking_active=thinking_active,
            )

            response = await self._create_message(request_kwargs)

            # Update usage statistics
            self._requests_made += 1
            if response.usage:
                self._prompt_tokens += response.usage.input_tokens
                self._completion_tokens += response.usage.output_tokens
                self._tokens_used += (
                    response.usage.input_tokens + response.usage.output_tokens
                )

            # Check for refusal (safety-related)
            if response.stop_reason == "refusal":
                raise RuntimeError(
                    "Model refused to generate structured output for safety reasons"
                )

            # Check for truncation
            if response.stop_reason == "max_tokens":
                usage_info = ""
                if response.usage:
                    usage_info = (
                        f" (input={response.usage.input_tokens:,}, "
                        f"output={response.usage.output_tokens:,})"
                    )
                raise RuntimeError(
                    f"Structured output truncated - increase max_completion_tokens{usage_info}"
                )

            # Extract text content from response
            # With structured outputs, the JSON is in the text block
            text_content = None
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    text_content = block.text
                    break

            if not text_content:
                raise RuntimeError(
                    "Model did not return text content for structured output. "
                    f"Stop reason: {response.stop_reason}"
                )

            # Parse and return JSON
            result: dict[str, Any] = json.loads(text_content)
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output as JSON: {e}")
            raise RuntimeError(f"Invalid JSON in structured output: {e}") from e
        except RuntimeError:
            # Re-raise RuntimeErrors from refusal, truncation, or missing content
            raise
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
                Optional: strict=True for guaranteed schema validation on inputs
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

        Note:
            When any tool has strict=True, the structured-outputs beta header
            is automatically added to guarantee schema-compliant tool inputs.
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

            # Thinking configuration (adaptive / manual / off), gated by
            # tool_choice compatibility.
            thinking_cfg = (
                self._build_thinking_config() if thinking_compatible else None
            )
            if thinking_cfg is not None:
                if self._thinking_mode == "manual":
                    self._ensure_thinking_max_tokens(
                        request_kwargs, max_completion_tokens
                    )
                request_kwargs["thinking"] = thinking_cfg

            # Structured outputs is a beta for strict tool use only; JSON
            # outputs via output_config.format don't need the header.
            thinking_active = thinking_cfg is not None
            beta_headers = self._get_beta_headers(thinking_active=thinking_active)
            if any(tool.get("strict") for tool in tools):
                beta_headers.insert(0, BETA_STRUCTURED_OUTPUTS)
            if beta_headers:
                request_kwargs["betas"] = beta_headers

            self._apply_common_request_fields(
                request_kwargs, thinking_active=thinking_active
            )

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
        return estimate_tokens_llm(text)

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
                "thinking_mode": self._thinking_mode,
                "interleaved_thinking": self._interleaved_thinking,
                "context_management_enabled": self._context_management_enabled,
            }
            if self._effort and supports_effort(self._model):
                result["effort"] = self._effort
            if self._thinking_display:
                result["thinking_display"] = self._thinking_display
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
    ) -> AsyncIterator[str]:
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

            # Thinking configuration (adaptive / manual / off)
            thinking_cfg = self._build_thinking_config()
            thinking_active = thinking_cfg is not None
            if thinking_cfg is not None:
                if self._thinking_mode == "manual":
                    self._ensure_thinking_max_tokens(
                        request_kwargs, max_completion_tokens
                    )
                request_kwargs["thinking"] = thinking_cfg

            beta_headers = self._get_beta_headers(thinking_active=thinking_active)
            if beta_headers:
                request_kwargs["betas"] = beta_headers

            self._apply_common_request_fields(
                request_kwargs, thinking_active=thinking_active
            )

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
