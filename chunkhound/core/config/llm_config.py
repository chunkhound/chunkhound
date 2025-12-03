"""
LLM configuration for ChunkHound deep research.

This module provides a type-safe, validated configuration system for LLM
providers with support for multiple configuration sources (environment
variables, config files, CLI arguments).
"""

import argparse
import os
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    provider: Literal[
        "openai",
        "ollama",
        "claude-code-cli",
        "codex-cli",
        "gemini",
        "anthropic",
    ] = Field(
        default="openai",
        description="Default LLM provider for both roles (utility, synthesis)",
    )

    # Optional per-role overrides (utility vs synthesis)
    utility_provider: Literal[
        "openai",
        "ollama",
        "claude-code-cli",
        "codex-cli",
        "anthropic",
        "gemini",
    ] | None = Field(default=None, description="Override provider for utility ops")

    synthesis_provider: Literal[
        "openai",
        "ollama",
        "claude-code-cli",
        "codex-cli",
        "anthropic",
        "gemini",
    ] | None = Field(default=None, description="Override provider for synthesis ops")

    # Model Configuration (dual-model architecture)
    utility_model: str = Field(
        default="",  # Will be set by get_default_models() if empty
        description="Model for utility operations (query expansion, follow-ups, classification)",
    )

    synthesis_model: str = Field(
        default="",  # Will be set by get_default_models() if empty
        description="Model for final synthesis (large context analysis)",
    )

    codex_reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = Field(
        default=None,
        description="Default Codex CLI reasoning effort (Responses API thinking level)",
    )
    codex_reasoning_effort_utility: (
        Literal["minimal", "low", "medium", "high"] | None
    ) = Field(
        default=None,
        description="Codex CLI reasoning effort override for utility-stage operations",
    )
    codex_reasoning_effort_synthesis: (
        Literal["minimal", "low", "medium", "high"] | None
    ) = Field(
        default=None,
        description="Codex CLI reasoning effort override for synthesis-stage operations",
    )

    # Anthropic Extended Thinking Configuration
    anthropic_thinking_enabled: bool = Field(
        default=False,
        description="Enable Anthropic extended thinking (shows Claude's reasoning process)",
    )

    anthropic_thinking_budget_tokens: int = Field(
        default=10000,
        ge=1024,
        description="Token budget for Anthropic thinking (min 1024, recommend 10000)",
    )

    anthropic_interleaved_thinking: bool = Field(
        default=False,
        description=(
            "Enable interleaved thinking for tool use (Claude 4 models only). "
            "Allows Claude to think between tool calls for more sophisticated reasoning."
        ),
    )

    # Anthropic Effort Parameter (Opus 4.5 only)
    anthropic_effort: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description=(
            "Control token usage vs thoroughness tradeoff (Opus 4.5 only). "
            "high=maximum capability (default), medium=balanced, low=most efficient"
        ),
    )

    # Anthropic Context Management
    anthropic_context_management_enabled: bool = Field(
        default=False,
        description="Enable automatic context management (tool result and thinking block clearing)",
    )

    anthropic_clear_thinking_keep_turns: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of recent assistant turns with thinking blocks to preserve. "
            "Set to None to keep all thinking blocks. Only used when context_management_enabled=True."
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
        description="Number of recent tool use/result pairs to keep after clearing. Default is 3.",
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

    @field_validator(
        "codex_reasoning_effort",
        "codex_reasoning_effort_utility",
        "codex_reasoning_effort_synthesis",
        mode="before",
    )
    def normalize_codex_effort(cls, v: str | None) -> str | None:  # noqa: N805
        if v is None:
            return v
        if isinstance(v, str):
            return v.strip().lower()
        return v

    def get_provider_configs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Get provider-specific configuration dictionaries for utility and synthesis models.

        Returns:
            Tuple of (utility_config, synthesis_config)
        """
        # Get default models if not specified
        utility_default, synthesis_default = self.get_default_models()

        # Resolve providers per-role
        resolved_utility_provider = self.utility_provider or self.provider
        resolved_synthesis_provider = self.synthesis_provider or self.provider

        base_config = {
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

        # Add reasoning_effort for providers that support it (OpenAI and Codex)
        utility_effort = _codex_effort_for("utility")
        if resolved_utility_provider in ("codex-cli", "openai") and utility_effort:
            utility_config["reasoning_effort"] = utility_effort

        synthesis_effort = _codex_effort_for("synthesis")
        if resolved_synthesis_provider in ("codex-cli", "openai") and synthesis_effort:
            synthesis_config["reasoning_effort"] = synthesis_effort

        # Add Anthropic configuration
        if resolved_utility_provider == "anthropic":
            utility_config["thinking_enabled"] = self.anthropic_thinking_enabled
            utility_config["thinking_budget_tokens"] = (
                self.anthropic_thinking_budget_tokens
            )
            utility_config["interleaved_thinking"] = self.anthropic_interleaved_thinking
            if self.anthropic_effort:
                utility_config["effort"] = self.anthropic_effort
            if self.anthropic_context_management_enabled:
                utility_config["context_management_enabled"] = True
                if self.anthropic_clear_thinking_keep_turns is not None:
                    utility_config["clear_thinking_keep_turns"] = (
                        self.anthropic_clear_thinking_keep_turns
                    )
                if self.anthropic_clear_tool_uses_trigger_tokens is not None:
                    utility_config["clear_tool_uses_trigger_tokens"] = (
                        self.anthropic_clear_tool_uses_trigger_tokens
                    )
                if self.anthropic_clear_tool_uses_keep is not None:
                    utility_config["clear_tool_uses_keep"] = (
                        self.anthropic_clear_tool_uses_keep
                    )

        if resolved_synthesis_provider == "anthropic":
            synthesis_config["thinking_enabled"] = self.anthropic_thinking_enabled
            synthesis_config["thinking_budget_tokens"] = (
                self.anthropic_thinking_budget_tokens
            )
            synthesis_config["interleaved_thinking"] = (
                self.anthropic_interleaved_thinking
            )
            if self.anthropic_effort:
                synthesis_config["effort"] = self.anthropic_effort
            if self.anthropic_context_management_enabled:
                synthesis_config["context_management_enabled"] = True
                if self.anthropic_clear_thinking_keep_turns is not None:
                    synthesis_config["clear_thinking_keep_turns"] = (
                        self.anthropic_clear_thinking_keep_turns
                    )
                if self.anthropic_clear_tool_uses_trigger_tokens is not None:
                    synthesis_config["clear_tool_uses_trigger_tokens"] = (
                        self.anthropic_clear_tool_uses_trigger_tokens
                    )
                if self.anthropic_clear_tool_uses_keep is not None:
                    synthesis_config["clear_tool_uses_keep"] = (
                        self.anthropic_clear_tool_uses_keep
                    )

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
            # Claude Code CLI: Haiku 4.5 for both utility and synthesis
            # (Haiku 4.5 is synthesis-capable and cost-effective, unlike 3.5)
            return ("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001")
        elif self.provider == "codex-cli":
            # Codex CLI: nominal label; require explicit model if desired
            return ("codex", "codex")
        elif self.provider == "gemini":
            # Gemini: Use Gemini 3 Pro for both (advanced reasoning)
            # Alternative models: gemini-2.5-pro (balanced), gemini-2.5-flash (fast)
            return ("gemini-3-pro-preview", "gemini-3-pro-preview")
        elif self.provider == "anthropic":
            # Anthropic: Haiku 4.5 for utility (fast/cheap), Sonnet 4.5 for synthesis (powerful)
            # Claude 4.5 generation models:
            # - claude-haiku-4-5-20251001: Fastest model with near-frontier intelligence
            # - claude-sonnet-4-5-20250929: Smartest model for complex agents and coding
            # - claude-opus-4-5-20251101: Most capable model with effort control (supports effort parameter)
            # - claude-opus-4-1-20250805: Exceptional model for specialized reasoning (legacy)
            return ("claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929")
        else:
            return ("gpt-5-nano", "gpt-5")

    def is_provider_configured(self) -> bool:
        """
        Check if the selected provider is properly configured.

        Returns:
            True if provider is properly configured
        """
        resolved_utility_provider = self.utility_provider or self.provider
        resolved_synthesis_provider = self.synthesis_provider or self.provider
        no_key_required = {"ollama", "claude-code-cli", "codex-cli"}
        if (
            resolved_utility_provider in no_key_required
            and resolved_synthesis_provider in no_key_required
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
            self.provider not in ("ollama", "claude-code-cli", "codex-cli")
            and not self.api_key
        ):
            missing.append("api_key (set CHUNKHOUND_LLM_API_KEY)")

        return missing

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add LLM-related CLI arguments."""
        parser.add_argument(
            "--llm-utility-model",
            help="Model for utility operations (query expansion, follow-ups, classification)",
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
            choices=["openai", "ollama", "claude-code-cli", "codex-cli", "anthropic", "gemini"],
            help="Default LLM provider for both roles",
        )

        parser.add_argument(
            "--llm-utility-provider",
            choices=["openai", "ollama", "claude-code-cli", "codex-cli", "anthropic", "gemini"],
            help="Override LLM provider for utility operations",
        )

        parser.add_argument(
            "--llm-synthesis-provider",
            choices=["openai", "ollama", "claude-code-cli", "codex-cli", "anthropic", "gemini"],
            help="Override LLM provider for synthesis operations",
        )

        parser.add_argument(
            "--llm-codex-reasoning-effort",
            choices=["minimal", "low", "medium", "high"],
            help="Codex CLI reasoning effort (thinking depth) when using codex-cli provider",
        )

        parser.add_argument(
            "--llm-codex-reasoning-effort-utility",
            choices=["minimal", "low", "medium", "high"],
            help="Utility-stage Codex reasoning effort override",
        )

        parser.add_argument(
            "--llm-codex-reasoning-effort-synthesis",
            choices=["minimal", "low", "medium", "high"],
            help="Synthesis-stage Codex reasoning effort override",
        )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load LLM config from environment variables."""
        config = {}

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
