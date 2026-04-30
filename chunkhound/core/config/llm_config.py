"""
LLM configuration for ChunkHound deep research.

This module provides a type-safe, validated configuration system for LLM
providers with support for multiple configuration sources (environment
variables, config files, CLI arguments).
"""

import argparse
import os
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from chunkhound.core.config.claude_model_resolution import (
    CLAUDE_HAIKU_DEFAULT_SENTINEL,
)

REASONING_EFFORT_PROVIDERS: tuple[str, ...] = (
    "codex-cli",
    "openai",
    "opencode-cli",
)

OPENAI_REASONING_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high")

NO_KEY_PROVIDERS: tuple[str, ...] = (
    "ollama",
    "claude-code-cli",
    "codex-cli",
    "opencode-cli",
)

LLMProviderLiteral = Literal[
    "openai",
    "ollama",
    "claude-code-cli",
    "codex-cli",
    "gemini",
    "anthropic",
    "grok",
    "opencode-cli",
]

ReasoningEffortLiteral = Literal["minimal", "low", "medium", "high", "xhigh"]


class LLMConfig(BaseSettings):
    """
    LLM configuration for ChunkHound deep research.

    Configuration Sources (in order of precedence):
    1. CLI arguments
    2. Environment variables (CHUNKHOUND_LLM_*)
    3. Config files
    4. Default values

    Environment Variables:
        CHUNKHOUND_LLM_API_KEY=sk-...
        CHUNKHOUND_LLM_UTILITY_MODEL=gpt-5-nano
        CHUNKHOUND_LLM_SYNTHESIS_MODEL=gpt-5
        CHUNKHOUND_LLM_BASE_URL=https://api.openai.com/v1
        CHUNKHOUND_LLM_PROVIDER=openai
        CHUNKHOUND_LLM_CODEX_REASONING_EFFORT=medium
        CHUNKHOUND_LLM_CODEX_REASONING_EFFORT_UTILITY=low
        CHUNKHOUND_LLM_CODEX_REASONING_EFFORT_SYNTHESIS=high
    """

    model_config = SettingsConfigDict(
        env_prefix="CHUNKHOUND_LLM_",
        env_nested_delimiter="__",
        case_sensitive=False,
        validate_default=True,
        extra="ignore",  # Ignore unknown fields for forward compatibility
    )

    # Provider Selection
    provider: LLMProviderLiteral = Field(
        default="openai",
        description="Default LLM provider for both roles (utility, synthesis)",
    )

    # Optional per-role overrides (utility vs synthesis)
    utility_provider: LLMProviderLiteral | None = Field(
        default=None, description="Override provider for utility ops"
    )

    synthesis_provider: LLMProviderLiteral | None = Field(
        default=None, description="Override provider for synthesis ops"
    )

    # Model Configuration (dual-model architecture)
    model: str | None = Field(
        default=None,
        description=(
            "Convenience field to set both utility and synthesis models "
            "to the same value"
        ),
    )

    utility_model: str = Field(
        default="",  # Will be set by get_default_models() if empty
        description=(
            "Model for utility operations "
            "(query expansion, follow-ups, classification)"
        ),
    )

    synthesis_model: str = Field(
        default="",  # Will be set by get_default_models() if empty
        description="Model for final synthesis (large context analysis)",
    )

    codex_reasoning_effort: ReasoningEffortLiteral | None = Field(
        default=None,
        description=(
            "Default reasoning effort forwarded to codex-cli, openai, "
            "and opencode-cli providers"
        ),
    )
    codex_reasoning_effort_utility: ReasoningEffortLiteral | None = Field(
        default=None,
        description=(
            "Reasoning effort override for utility-stage operations "
            "(codex-cli, openai, opencode-cli)"
        ),
    )
    codex_reasoning_effort_synthesis: ReasoningEffortLiteral | None = Field(
        default=None,
        description=(
            "Reasoning effort override for synthesis-stage operations "
            "(codex-cli, openai, opencode-cli)"
        ),
    )

    map_hyde_provider: LLMProviderLiteral | None = Field(
        default=None,
        description=(
            "Override provider for Code Mapper HyDE planning "
            "(points-of-interest overview). "
            "Falls back to the synthesis provider when unset."
        ),
    )

    map_hyde_model: str | None = Field(
        default=None,
        description=(
            "Override model for Code Mapper HyDE planning "
            "(points-of-interest overview). "
            "Falls back to the synthesis model when unset."
        ),
    )

    map_hyde_reasoning_effort: ReasoningEffortLiteral | None = Field(
        default=None,
        description=(
            "Reasoning effort override for Code Mapper HyDE planning. "
            "Falls back to synthesis reasoning effort when unset."
        ),
    )

    autodoc_cleanup_provider: LLMProviderLiteral | None = Field(
        default=None,
        description=(
            "Override provider for AutoDoc LLM cleanup. "
            "Falls back to the synthesis provider when unset."
        ),
    )

    autodoc_cleanup_model: str | None = Field(
        default=None,
        description=(
            "Override model for AutoDoc LLM cleanup. "
            "Falls back to the synthesis model when unset."
        ),
    )

    autodoc_cleanup_reasoning_effort: ReasoningEffortLiteral | None = Field(
        default=None,
        description=(
            "Reasoning effort override for AutoDoc LLM cleanup. "
            "Falls back to synthesis reasoning effort when unset."
        ),
    )

    # Anthropic Extended Thinking Configuration
    anthropic_thinking_enabled: bool = Field(
        default=False,
        description=(
            "Enable Anthropic extended thinking "
            "(shows Claude's reasoning process)"
        ),
    )

    anthropic_thinking_mode: Literal["auto", "off", "manual", "adaptive"] | None = (
        Field(
            default=None,
            description=(
                "Explicit thinking mode. 'adaptive' lets Claude decide depth "
                "(Opus 4.6/4.7, Sonnet 4.6, Mythos). 'manual' uses a fixed "
                "budget (accepted on Opus 4.5/4.6, Sonnet 4.5/4.6, Haiku 4.5; "
                "rejected on Opus 4.7/Mythos). 'off' disables thinking. "
                "Default 'auto' picks adaptive for 4.6+ models and manual "
                "for older ones when anthropic_thinking_enabled is true."
            ),
        )
    )

    anthropic_thinking_display: Literal["summarized", "omitted"] | None = Field(
        default=None,
        description=(
            "Claude's thinking block display mode. 'summarized' (default on "
            "Opus 4.6 and Sonnet 4.6) returns summarized thinking text. "
            "'omitted' (default on Opus 4.7) keeps signatures for multi-turn "
            "continuity but returns empty thinking text. Applies to adaptive mode."
        ),
    )

    anthropic_thinking_budget_tokens: int = Field(
        default=10000,
        ge=1024,
        description=(
            "Token budget for Anthropic manual-mode thinking (min 1024, "
            "recommend 10000). Ignored in adaptive mode."
        ),
    )

    anthropic_interleaved_thinking: bool = Field(
        default=False,
        description=(
            "Enable interleaved thinking for tool use. Auto-enabled in "
            "adaptive mode (Opus 4.6/4.7, Sonnet 4.6, Mythos). For manual "
            "mode on any model that supports it (Opus 4.5, Sonnet 4.5/4.6, "
            "Haiku 4.5), sends the interleaved-thinking-2025-05-14 beta."
        ),
    )

    # Anthropic Effort Parameter (Opus 4.5/4.6/4.7, Sonnet 4.6, Mythos)
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = Field(
        default=None,
        description=(
            "Control token usage vs thoroughness tradeoff. Supported on "
            "Opus 4.5, Opus 4.6, Opus 4.7, Sonnet 4.6, and Mythos. "
            "low/medium/high on all supported models; max on 4.6+; "
            "xhigh is Opus 4.7 only. high is the API default; Sonnet 4.6 "
            "recommends medium."
        ),
    )

    # Anthropic Prompt Caching (opt-in, ephemeral)
    anthropic_prompt_caching: bool = Field(
        default=False,
        description=(
            "Opt in to Anthropic prompt caching. Disabled by default because "
            "ChunkHound requests rarely reuse prompt prefixes enough to offset "
            "cache-write costs. Cache hits cost 10% of base input; writes cost "
            "25% more for 5m TTL, 100% more for 1h TTL."
        ),
    )

    anthropic_cache_ttl: Literal["5m", "1h"] | None = Field(
        default=None,
        description=(
            "Cache TTL for prompt caching. Default 5m is free to refresh on "
            "each hit. Use '1h' (2x write cost) for workloads that reuse the "
            "same prefix less often than every 5 minutes."
        ),
    )

    # Anthropic Task Budgets (beta, Opus 4.7 only)
    anthropic_task_budget_tokens: int | None = Field(
        default=None,
        ge=20000,
        description=(
            "Advisory total token budget for a full agentic loop on Opus 4.7 "
            "(beta, requires task-budgets-2026-03-13). Unlike max_tokens this "
            "is visible to the model so it can pace itself. Min 20000. Leave "
            "unset for open-ended work where quality matters over throughput."
        ),
    )

    # Anthropic Context Management
    anthropic_context_management_enabled: bool = Field(
        default=False,
        description=(
            "Enable automatic context management "
            "(tool result and thinking block clearing)"
        ),
    )

    anthropic_clear_thinking_keep_turns: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of recent assistant turns with thinking blocks to preserve. "
            "Set to None to keep all thinking blocks. "
            "Only used when context_management_enabled=True."
        ),
    )

    anthropic_clear_tool_uses_trigger_tokens: int | None = Field(
        default=None,
        description=(
            "Input token threshold to trigger tool result clearing. "
            "Default is 100,000 tokens if not specified."
        ),
    )

    anthropic_clear_tool_uses_keep: int | None = Field(
        default=None,
        description=(
            "Number of recent tool use/result pairs to keep after clearing. "
            "Default is 3."
        ),
    )

    api_key: SecretStr | None = Field(
        default=None, description="API key for authentication (provider-specific)"
    )

    base_url: str | None = Field(
        default=None,
        description=(
            "Provider-specific base URL: "
            "OpenAI/Ollama: API endpoint (e.g., http://localhost:11434/v1)"
        ),
    )

    # Internal settings
    timeout: int = Field(default=60, description="Internal timeout for LLM calls")
    max_retries: int = Field(default=3, description="Internal max retries")

    @field_validator("base_url")
    def validate_base_url(cls, v: str | None) -> str | None:  # noqa: N805
        """Validate and normalize base URL."""
        if v is None:
            return v

        # Remove trailing slash for consistency
        v = v.rstrip("/")

        # Basic URL validation
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("base_url must start with http:// or https://")

        return v

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook to handle model field mapping."""
        # If model is provided, set both utility_model and synthesis_model
        if self.model is not None:
            if not self.utility_model:
                self.utility_model = self.model
            if not self.synthesis_model:
                self.synthesis_model = self.model

    @model_validator(mode="after")
    def validate_opencode_cli_model(self) -> "LLMConfig":
        """Validate opencode-cli models use provider/model format."""
        for role, provider_field, model_field in (
            ("utility", self.utility_provider, self.utility_model),
            ("synthesis", self.synthesis_provider, self.synthesis_model),
            ("map_hyde", self.map_hyde_provider, self.map_hyde_model),
            (
                "autodoc_cleanup",
                self.autodoc_cleanup_provider,
                self.autodoc_cleanup_model,
            ),
        ):
            # map_hyde and autodoc_cleanup model fields default to None,
            # meaning "fall back to synthesis model". Skip validation
            # when unset — synthesis is checked separately above.
            if model_field is None:
                continue
            resolved = provider_field or self.provider
            if resolved == "opencode-cli" and (
                not model_field or "/" not in model_field
            ):
                raise ValueError(
                    f"opencode-cli requires a model in provider/model format "
                    f"for {role} (e.g., opencode/gpt-5-nano). "
                    f"Got: {model_field!r}"
                )

        for role, provider_name, effort in (
            (
                "utility",
                self.utility_provider or self.provider,
                self.codex_reasoning_effort_utility or self.codex_reasoning_effort,
            ),
            (
                "synthesis",
                self.synthesis_provider or self.provider,
                self.codex_reasoning_effort_synthesis or self.codex_reasoning_effort,
            ),
            (
                "map_hyde",
                self.map_hyde_provider or self.synthesis_provider or self.provider,
                self.map_hyde_reasoning_effort,
            ),
            (
                "autodoc_cleanup",
                self.autodoc_cleanup_provider
                or self.synthesis_provider
                or self.provider,
                self.autodoc_cleanup_reasoning_effort,
            ),
        ):
            if (
                provider_name == "openai"
                and effort is not None
                and effort not in OPENAI_REASONING_EFFORTS
            ):
                raise ValueError(
                    f"openai does not support reasoning effort {effort!r} "
                    f"for {role}; use one of {', '.join(OPENAI_REASONING_EFFORTS)}"
                )
        return self

    @field_validator(
        "codex_reasoning_effort",
        "codex_reasoning_effort_utility",
        "codex_reasoning_effort_synthesis",
        "map_hyde_reasoning_effort",
        "autodoc_cleanup_reasoning_effort",
        "anthropic_effort",
        "anthropic_thinking_mode",
        "anthropic_thinking_display",
        "anthropic_cache_ttl",
        mode="before",
    )
    def _normalize_effort_strings(cls, v: str | None) -> str | None:  # noqa: N805
        """Normalize effort / mode / display / ttl strings to lowercase."""
        if v is None:
            return v
        if isinstance(v, str):
            return v.strip().lower()
        raise ValueError(f"Expected str or None, got {type(v).__name__}")

    def get_provider_configs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Get provider-specific configuration dictionaries.

        Returns:
            Tuple of (utility_config, synthesis_config)
        """
        # Get default models if not specified
        utility_default, synthesis_default = self.get_default_models()

        # Resolve providers per-role
        resolved_utility_provider = self.utility_provider or self.provider
        resolved_synthesis_provider = self.synthesis_provider or self.provider

        base_config: dict[str, Any] = {
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        # Add API key if available
        if self.api_key:
            base_config["api_key"] = self.api_key.get_secret_value()

        # Add base URL if available
        if self.base_url:
            base_config["base_url"] = self.base_url

        # Build utility config
        utility_config = base_config.copy()
        utility_config["provider"] = resolved_utility_provider
        utility_config["model"] = self.utility_model or utility_default

        # Build synthesis config
        synthesis_config = base_config.copy()
        synthesis_config["provider"] = resolved_synthesis_provider
        synthesis_config["model"] = self.synthesis_model or synthesis_default

        def _codex_effort_for(role: str) -> str | None:
            default_effort = self.codex_reasoning_effort
            if role == "utility":
                return self.codex_reasoning_effort_utility or default_effort
            if role == "synthesis":
                return self.codex_reasoning_effort_synthesis or default_effort
            return default_effort

        # Add reasoning_effort for providers that support it
        utility_effort = _codex_effort_for("utility")
        if (
            resolved_utility_provider in REASONING_EFFORT_PROVIDERS
            and utility_effort
        ):
            utility_config["reasoning_effort"] = utility_effort

        synthesis_effort = _codex_effort_for("synthesis")
        if (
            resolved_synthesis_provider in REASONING_EFFORT_PROVIDERS
            and synthesis_effort
        ):
            synthesis_config["reasoning_effort"] = synthesis_effort

        def _apply_anthropic(target: dict[str, Any]) -> None:
            target["thinking_enabled"] = self.anthropic_thinking_enabled
            target["thinking_budget_tokens"] = self.anthropic_thinking_budget_tokens
            target["interleaved_thinking"] = self.anthropic_interleaved_thinking
            if self.anthropic_thinking_mode:
                target["thinking_mode"] = self.anthropic_thinking_mode
            if self.anthropic_thinking_display:
                target["thinking_display"] = self.anthropic_thinking_display
            if self.anthropic_effort:
                target["effort"] = self.anthropic_effort
            target["prompt_caching"] = self.anthropic_prompt_caching
            if self.anthropic_cache_ttl:
                target["cache_ttl"] = self.anthropic_cache_ttl
            if self.anthropic_task_budget_tokens is not None:
                target["task_budget_tokens"] = self.anthropic_task_budget_tokens
            if self.anthropic_context_management_enabled:
                target["context_management_enabled"] = True
                if self.anthropic_clear_thinking_keep_turns is not None:
                    target["clear_thinking_keep_turns"] = (
                        self.anthropic_clear_thinking_keep_turns
                    )
                if self.anthropic_clear_tool_uses_trigger_tokens is not None:
                    target["clear_tool_uses_trigger_tokens"] = (
                        self.anthropic_clear_tool_uses_trigger_tokens
                    )
                if self.anthropic_clear_tool_uses_keep is not None:
                    target["clear_tool_uses_keep"] = self.anthropic_clear_tool_uses_keep

        if resolved_utility_provider == "anthropic":
            _apply_anthropic(utility_config)

        if resolved_synthesis_provider == "anthropic":
            _apply_anthropic(synthesis_config)

        return utility_config, synthesis_config

    def get_default_models(self) -> tuple[str, str]:
        """
        Get default model names for utility and synthesis based on provider.

        Returns:
            Tuple of (utility_model, synthesis_model)
        """
        # Provider-specific smart defaults
        if self.provider == "openai":
            return ("gpt-5-nano", "gpt-5")
        elif self.provider == "ollama":
            # Ollama: use same model for both (local deployment)
            return ("llama3.2", "llama3.2")
        elif self.provider == "claude-code-cli":
            # Claude Code CLI: shared Claude Haiku sentinel for both roles;
            # CLI construction resolves it offline to ChunkHound's pinned fallback.
            return (CLAUDE_HAIKU_DEFAULT_SENTINEL, CLAUDE_HAIKU_DEFAULT_SENTINEL)
        elif self.provider == "codex-cli":
            # Codex CLI: nominal label; require explicit model if desired
            return ("codex", "codex")
        elif self.provider == "gemini":
            # Gemini: Use Gemini 3 Pro for both (advanced reasoning)
            # Alternative models: gemini-2.5-pro (balanced), gemini-2.5-flash (fast)
            return ("gemini-3-pro-preview", "gemini-3-pro-preview")
        elif self.provider == "anthropic":
            # Anthropic intentionally uses Claude Haiku for both utility and
            # synthesis. Haiku is capable enough for synthesis and is Anthropic's
            # cheapest Claude model; Anthropic has no true low-cost utility tier.
            return (CLAUDE_HAIKU_DEFAULT_SENTINEL, CLAUDE_HAIKU_DEFAULT_SENTINEL)
        elif self.provider == "grok":
            # Grok reasoning models (especially grok-4-1-fast-reasoning)
            # are extremely strong across both roles — no benefit to splitting them.
            # grok-4-1-fast-reasoning: Advanced reasoning, structured outputs,
            # tool calling
            return ("grok-4-1-fast-reasoning", "grok-4-1-fast-reasoning")
        elif self.provider == "opencode-cli":
            # OpenCode CLI: No universal default — model depends on user config.
            # User must set model in provider/model format.
            return ("", "")
        else:
            # Safety fallback — all known providers are covered above
            return ("gpt-5-nano", "gpt-5")  # type: ignore[unreachable]

    def is_provider_configured(self) -> bool:
        """
        Check if the selected provider is properly configured.

        Returns:
            True if provider is properly configured
        """
        resolved_utility_provider = self.utility_provider or self.provider
        resolved_synthesis_provider = self.synthesis_provider or self.provider
        if (
            resolved_utility_provider in NO_KEY_PROVIDERS
            and resolved_synthesis_provider in NO_KEY_PROVIDERS
        ):
            # Ollama and Claude Code CLI don't require API key
            # Claude Code CLI uses subscription-based authentication
            return True
        else:
            # OpenAI and Anthropic require API key
            return self.api_key is not None

    def get_missing_config(self) -> list[str]:
        """
        Get list of missing required configuration.

        Returns:
            List of missing configuration parameter names
        """
        missing = []

        if (
            self.provider not in NO_KEY_PROVIDERS
            and not self.api_key
        ):
            missing.append("api_key (set CHUNKHOUND_LLM_API_KEY)")

        return missing

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add LLM-related CLI arguments."""
        parser.add_argument(
            "--llm-utility-model",
            help=(
                "Model for utility operations "
                "(query expansion, follow-ups, classification)"
            ),
        )

        parser.add_argument(
            "--llm-synthesis-model",
            help="Model for final synthesis (large context analysis)",
        )

        parser.add_argument(
            "--llm-api-key",
            help="API key for LLM provider (uses env var if not specified)",
        )

        parser.add_argument(
            "--llm-base-url",
            help="Base URL for LLM API (uses env var if not specified)",
        )

        parser.add_argument(
            "--llm-provider",
            choices=[
                "openai",
                "ollama",
                "claude-code-cli",
                "codex-cli",
                "anthropic",
                "gemini",
                "grok",
                "opencode-cli",
            ],
            help="Default LLM provider for both roles",
        )

        parser.add_argument(
            "--llm-utility-provider",
            choices=[
                "openai",
                "ollama",
                "claude-code-cli",
                "codex-cli",
                "anthropic",
                "gemini",
                "grok",
                "opencode-cli",
            ],
            help="Override LLM provider for utility operations",
        )

        parser.add_argument(
            "--llm-synthesis-provider",
            choices=[
                "openai",
                "ollama",
                "claude-code-cli",
                "codex-cli",
                "anthropic",
                "gemini",
                "grok",
                "opencode-cli",
            ],
            help="Override LLM provider for synthesis operations",
        )

        parser.add_argument(
            "--llm-codex-reasoning-effort",
            choices=["minimal", "low", "medium", "high", "xhigh"],
            help=(
                "Codex CLI reasoning effort (thinking depth) "
                "when using codex-cli, opencode-cli, or openai provider "
                "(openai excludes xhigh)"
            ),
        )

        parser.add_argument(
            "--llm-codex-reasoning-effort-utility",
            choices=["minimal", "low", "medium", "high", "xhigh"],
            help="Utility-stage Codex reasoning effort override",
        )

        parser.add_argument(
            "--llm-codex-reasoning-effort-synthesis",
            choices=["minimal", "low", "medium", "high", "xhigh"],
            help="Synthesis-stage Codex reasoning effort override",
        )

        parser.add_argument(
            "--llm-map-hyde-provider",
            choices=[
                "openai",
                "ollama",
                "claude-code-cli",
                "codex-cli",
                "anthropic",
                "gemini",
                "grok",
                "opencode-cli",
            ],
            help=(
                "Override provider for Code Mapper HyDE planning "
                "(falls back to synthesis)"
            ),
        )
        parser.add_argument(
            "--llm-map-hyde-model",
            help=(
                "Override model for Code Mapper HyDE planning "
                "(falls back to synthesis)"
            ),
        )
        parser.add_argument(
            "--llm-map-hyde-reasoning-effort",
            choices=["minimal", "low", "medium", "high", "xhigh"],
            help=(
                "Override reasoning effort for Code Mapper HyDE planning "
                "(openai excludes xhigh)"
            ),
        )

        parser.add_argument(
            "--llm-autodoc-cleanup-provider",
            choices=[
                "openai",
                "ollama",
                "claude-code-cli",
                "codex-cli",
                "anthropic",
                "gemini",
                "grok",
                "opencode-cli",
            ],
            help="Override provider for AutoDoc LLM cleanup (falls back to synthesis)",
        )
        parser.add_argument(
            "--llm-autodoc-cleanup-model",
            help="Override model for AutoDoc LLM cleanup (falls back to synthesis)",
        )
        parser.add_argument(
            "--llm-autodoc-cleanup-reasoning-effort",
            choices=["minimal", "low", "medium", "high", "xhigh"],
            help=(
                "Override reasoning effort for AutoDoc LLM cleanup "
                "(openai excludes xhigh)"
            ),
        )

        anthropic_bool = argparse.BooleanOptionalAction
        parser.add_argument(
            "--llm-anthropic-thinking",
            dest="llm_anthropic_thinking_enabled",
            action=anthropic_bool,
            default=None,
            help="Enable Anthropic extended thinking",
        )
        parser.add_argument(
            "--llm-anthropic-thinking-mode",
            choices=["auto", "off", "manual", "adaptive"],
            help="Anthropic thinking mode selector",
        )
        parser.add_argument(
            "--llm-anthropic-thinking-display",
            choices=["summarized", "omitted"],
            help="Thinking display mode (adaptive models only)",
        )
        parser.add_argument(
            "--llm-anthropic-thinking-budget-tokens",
            type=int,
            help="Token budget for Anthropic manual thinking (min 1024)",
        )
        parser.add_argument(
            "--llm-anthropic-interleaved-thinking",
            dest="llm_anthropic_interleaved_thinking",
            action=anthropic_bool,
            default=None,
            help="Enable interleaved thinking between tool calls (manual mode)",
        )
        parser.add_argument(
            "--llm-anthropic-effort",
            choices=["low", "medium", "high", "xhigh", "max"],
            help="Anthropic effort level",
        )
        parser.add_argument(
            "--llm-anthropic-prompt-caching",
            dest="llm_anthropic_prompt_caching",
            action=anthropic_bool,
            default=None,
            help="Enable opt-in Anthropic prompt caching",
        )
        parser.add_argument(
            "--llm-anthropic-cache-ttl",
            choices=["5m", "1h"],
            help="Anthropic prompt cache TTL",
        )
        parser.add_argument(
            "--llm-anthropic-task-budget-tokens",
            type=int,
            help="Advisory task budget for agentic loops (Opus 4.7; min 20000)",
        )
        parser.add_argument(
            "--llm-anthropic-context-management",
            dest="llm_anthropic_context_management_enabled",
            action=anthropic_bool,
            default=None,
            help="Enable automatic Anthropic context management",
        )
        parser.add_argument(
            "--llm-anthropic-clear-thinking-keep-turns",
            type=int,
            help="Number of thinking turns to preserve when clearing",
        )
        parser.add_argument(
            "--llm-anthropic-clear-tool-uses-trigger-tokens",
            type=int,
            help="Input token threshold that triggers tool-result clearing",
        )
        parser.add_argument(
            "--llm-anthropic-clear-tool-uses-keep",
            type=int,
            help="Number of recent tool use/result pairs to keep after clearing",
        )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load LLM config from environment variables."""
        config: dict[str, Any] = {}

        if api_key := os.getenv("CHUNKHOUND_LLM_API_KEY"):
            config["api_key"] = api_key
        if base_url := os.getenv("CHUNKHOUND_LLM_BASE_URL"):
            config["base_url"] = base_url
        if provider := os.getenv("CHUNKHOUND_LLM_PROVIDER"):
            config["provider"] = provider
        if u_provider := os.getenv("CHUNKHOUND_LLM_UTILITY_PROVIDER"):
            config["utility_provider"] = u_provider
        if s_provider := os.getenv("CHUNKHOUND_LLM_SYNTHESIS_PROVIDER"):
            config["synthesis_provider"] = s_provider
        if utility_model := os.getenv("CHUNKHOUND_LLM_UTILITY_MODEL"):
            config["utility_model"] = utility_model
        if synthesis_model := os.getenv("CHUNKHOUND_LLM_SYNTHESIS_MODEL"):
            config["synthesis_model"] = synthesis_model
        if codex_effort := os.getenv("CHUNKHOUND_LLM_CODEX_REASONING_EFFORT"):
            config["codex_reasoning_effort"] = codex_effort.strip().lower()
        if codex_effort_util := os.getenv(
            "CHUNKHOUND_LLM_CODEX_REASONING_EFFORT_UTILITY"
        ):
            config["codex_reasoning_effort_utility"] = codex_effort_util.strip().lower()
        if codex_effort_syn := os.getenv(
            "CHUNKHOUND_LLM_CODEX_REASONING_EFFORT_SYNTHESIS"
        ):
            config["codex_reasoning_effort_synthesis"] = (
                codex_effort_syn.strip().lower()
            )

        if map_hyde_provider := os.getenv("CHUNKHOUND_LLM_MAP_HYDE_PROVIDER"):
            config["map_hyde_provider"] = map_hyde_provider
        if map_hyde_model := os.getenv("CHUNKHOUND_LLM_MAP_HYDE_MODEL"):
            config["map_hyde_model"] = map_hyde_model
        if map_hyde_effort := os.getenv("CHUNKHOUND_LLM_MAP_HYDE_REASONING_EFFORT"):
            config["map_hyde_reasoning_effort"] = map_hyde_effort.strip().lower()

        if cleanup_provider := os.getenv("CHUNKHOUND_LLM_AUTODOC_CLEANUP_PROVIDER"):
            config["autodoc_cleanup_provider"] = cleanup_provider
        if cleanup_model := os.getenv("CHUNKHOUND_LLM_AUTODOC_CLEANUP_MODEL"):
            config["autodoc_cleanup_model"] = cleanup_model
        if cleanup_effort := os.getenv(
            "CHUNKHOUND_LLM_AUTODOC_CLEANUP_REASONING_EFFORT"
        ):
            config["autodoc_cleanup_reasoning_effort"] = cleanup_effort.strip().lower()

        def _bool_env(name: str) -> bool | None:
            raw = os.getenv(name)
            if raw is None:
                return None
            return raw.strip().lower() in ("1", "true", "yes", "on")

        bool_fields = (
            ("CHUNKHOUND_LLM_ANTHROPIC_THINKING_ENABLED", "anthropic_thinking_enabled"),
            (
                "CHUNKHOUND_LLM_ANTHROPIC_INTERLEAVED_THINKING",
                "anthropic_interleaved_thinking",
            ),
            ("CHUNKHOUND_LLM_ANTHROPIC_PROMPT_CACHING", "anthropic_prompt_caching"),
            (
                "CHUNKHOUND_LLM_ANTHROPIC_CONTEXT_MANAGEMENT_ENABLED",
                "anthropic_context_management_enabled",
            ),
        )
        for env_name, key in bool_fields:
            bool_val = _bool_env(env_name)
            if bool_val is not None:
                config[key] = bool_val

        str_fields = (
            ("CHUNKHOUND_LLM_ANTHROPIC_THINKING_MODE", "anthropic_thinking_mode"),
            ("CHUNKHOUND_LLM_ANTHROPIC_THINKING_DISPLAY", "anthropic_thinking_display"),
            ("CHUNKHOUND_LLM_ANTHROPIC_EFFORT", "anthropic_effort"),
            ("CHUNKHOUND_LLM_ANTHROPIC_CACHE_TTL", "anthropic_cache_ttl"),
        )
        for env_name, key in str_fields:
            raw = os.getenv(env_name)
            if raw:
                config[key] = raw.strip().lower()

        int_fields = (
            (
                "CHUNKHOUND_LLM_ANTHROPIC_THINKING_BUDGET_TOKENS",
                "anthropic_thinking_budget_tokens",
            ),
            (
                "CHUNKHOUND_LLM_ANTHROPIC_TASK_BUDGET_TOKENS",
                "anthropic_task_budget_tokens",
            ),
            (
                "CHUNKHOUND_LLM_ANTHROPIC_CLEAR_THINKING_KEEP_TURNS",
                "anthropic_clear_thinking_keep_turns",
            ),
            (
                "CHUNKHOUND_LLM_ANTHROPIC_CLEAR_TOOL_USES_TRIGGER_TOKENS",
                "anthropic_clear_tool_uses_trigger_tokens",
            ),
            (
                "CHUNKHOUND_LLM_ANTHROPIC_CLEAR_TOOL_USES_KEEP",
                "anthropic_clear_tool_uses_keep",
            ),
        )
        for env_name, key in int_fields:
            raw = os.getenv(env_name)
            if raw:
                config[key] = int(raw)

        return config

    @classmethod
    def extract_cli_overrides(cls, args: Any) -> dict[str, Any]:
        """Extract LLM config from CLI arguments."""
        overrides = {}

        if hasattr(args, "llm_utility_model") and args.llm_utility_model:
            overrides["utility_model"] = args.llm_utility_model
        if hasattr(args, "llm_synthesis_model") and args.llm_synthesis_model:
            overrides["synthesis_model"] = args.llm_synthesis_model
        if hasattr(args, "llm_api_key") and args.llm_api_key:
            overrides["api_key"] = args.llm_api_key
        if hasattr(args, "llm_base_url") and args.llm_base_url:
            overrides["base_url"] = args.llm_base_url
        if hasattr(args, "llm_provider") and args.llm_provider:
            overrides["provider"] = args.llm_provider
        if hasattr(args, "llm_utility_provider") and args.llm_utility_provider:
            overrides["utility_provider"] = args.llm_utility_provider
        if hasattr(args, "llm_synthesis_provider") and args.llm_synthesis_provider:
            overrides["synthesis_provider"] = args.llm_synthesis_provider
        if (
            hasattr(args, "llm_codex_reasoning_effort")
            and args.llm_codex_reasoning_effort
        ):
            overrides["codex_reasoning_effort"] = args.llm_codex_reasoning_effort
        if (
            hasattr(args, "llm_codex_reasoning_effort_utility")
            and args.llm_codex_reasoning_effort_utility
        ):
            overrides["codex_reasoning_effort_utility"] = (
                args.llm_codex_reasoning_effort_utility
            )
        if (
            hasattr(args, "llm_codex_reasoning_effort_synthesis")
            and args.llm_codex_reasoning_effort_synthesis
        ):
            overrides["codex_reasoning_effort_synthesis"] = (
                args.llm_codex_reasoning_effort_synthesis
            )

        if hasattr(args, "llm_map_hyde_provider") and args.llm_map_hyde_provider:
            overrides["map_hyde_provider"] = args.llm_map_hyde_provider
        if hasattr(args, "llm_map_hyde_model") and args.llm_map_hyde_model:
            overrides["map_hyde_model"] = args.llm_map_hyde_model
        if (
            hasattr(args, "llm_map_hyde_reasoning_effort")
            and args.llm_map_hyde_reasoning_effort
        ):
            overrides["map_hyde_reasoning_effort"] = args.llm_map_hyde_reasoning_effort

        if (
            hasattr(args, "llm_autodoc_cleanup_provider")
            and args.llm_autodoc_cleanup_provider
        ):
            overrides["autodoc_cleanup_provider"] = args.llm_autodoc_cleanup_provider
        if (
            hasattr(args, "llm_autodoc_cleanup_model")
            and args.llm_autodoc_cleanup_model
        ):
            overrides["autodoc_cleanup_model"] = args.llm_autodoc_cleanup_model
        if (
            hasattr(args, "llm_autodoc_cleanup_reasoning_effort")
            and args.llm_autodoc_cleanup_reasoning_effort
        ):
            overrides["autodoc_cleanup_reasoning_effort"] = (
                args.llm_autodoc_cleanup_reasoning_effort
            )

        anthropic_flag_map = {
            "llm_anthropic_thinking_enabled": "anthropic_thinking_enabled",
            "llm_anthropic_thinking_mode": "anthropic_thinking_mode",
            "llm_anthropic_thinking_display": "anthropic_thinking_display",
            "llm_anthropic_thinking_budget_tokens": "anthropic_thinking_budget_tokens",
            "llm_anthropic_interleaved_thinking": "anthropic_interleaved_thinking",
            "llm_anthropic_effort": "anthropic_effort",
            "llm_anthropic_prompt_caching": "anthropic_prompt_caching",
            "llm_anthropic_cache_ttl": "anthropic_cache_ttl",
            "llm_anthropic_task_budget_tokens": "anthropic_task_budget_tokens",
            "llm_anthropic_context_management_enabled": (
                "anthropic_context_management_enabled"
            ),
            "llm_anthropic_clear_thinking_keep_turns": (
                "anthropic_clear_thinking_keep_turns"
            ),
            "llm_anthropic_clear_tool_uses_trigger_tokens": (
                "anthropic_clear_tool_uses_trigger_tokens"
            ),
            "llm_anthropic_clear_tool_uses_keep": "anthropic_clear_tool_uses_keep",
        }
        for arg_name, config_key in anthropic_flag_map.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                overrides[config_key] = value

        return overrides

    def __repr__(self) -> str:
        """String representation hiding sensitive information."""
        api_key_display = "***" if self.api_key else None
        utility_model, synthesis_model = self.get_default_models()
        utility_display = self.utility_model or utility_model
        synthesis_display = self.synthesis_model or synthesis_model
        return (
            f"LLMConfig("
            f"provider={self.provider}, "
            f"utility_model={utility_display}, "
            f"synthesis_model={synthesis_display}, "
            f"api_key={api_key_display}, "
            f"base_url={self.base_url})"
        )
