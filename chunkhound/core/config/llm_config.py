"""
LLM configuration for ChunkHound deep research.

This module provides a type-safe, validated configuration system for LLM
providers with support for multiple configuration sources (environment
variables, config files, CLI arguments).
"""

import argparse
import os
from typing import Any, Literal, get_args

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import assert_never

from chunkhound.core.config.claude_model_resolution import (
    CLAUDE_HAIKU_SENTINEL,
)
from chunkhound.core.config.provider_registry import OPENAI_COMPATIBLE_PROVIDERS
from chunkhound.core.exceptions.core import ConfigurationError
from chunkhound.core.utils.openai_utils import is_official_openai_endpoint

from ._utils import _parse_env_bool

REASONING_EFFORT_PROVIDERS: tuple[str, ...] = (
    "codex-cli",
    "grok",
    "openai",
    "opencode-cli",
)

OPENAI_REASONING_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high")
GROK_REASONING_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high")
GEMINI_THINKING_LEVELS: tuple[str, ...] = ("low", "medium", "high")

NO_KEY_PROVIDERS: tuple[str, ...] = (
    "ollama",
    "claude-code-cli",
    "codex-cli",
    "opencode-cli",
    "antigravity-cli",
)

LLMProviderLiteral = Literal[
    "openai",
    "ollama",
    "claude-code-cli",
    "codex-cli",
    "deepseek",
    "gemini",
    "anthropic",
    "grok",
    "opencode-cli",
    "antigravity-sdk",
    "antigravity-cli",
]

_PROVIDER_CHOICES: list[str] = list(get_args(LLMProviderLiteral))

ReasoningEffortLiteral = Literal["minimal", "low", "medium", "high", "xhigh"]

DEFAULT_LLM_TIMEOUT = 120
# "openai" uses OpenAILLMProvider (special class with Responses API), not the
# generic OpenAICompatibleProvider path.  It is included here because it speaks
# the same Chat Completions protocol — used for config-level validation.
OPENAI_COMPATIBLE_LLM_PROVIDERS: set[str] = {"openai"} | set(
    OPENAI_COMPATIBLE_PROVIDERS.keys()
)
BASE_URL_CAPABLE_LLM_PROVIDERS: set[str] = OPENAI_COMPATIBLE_LLM_PROVIDERS | {
    "anthropic"
}

# Native providers without a baked-in default model — user must set llm.model explicitly
NO_DEFAULT_MODEL_PROVIDERS: set[str] = {"gemini"}

REMOVED_PROVIDERS: dict[str, str] = {
    "ollama": (
        "The 'ollama' provider has been removed. "
        "Use provider='openai' with base_url pointing to your Ollama "
        "instance (e.g. http://localhost:11434/v1)."
    ),
}

CLI_PROVIDER_CHOICES = (
    "openai",
    "claude-code-cli",
    "codex-cli",
    "deepseek",
    "anthropic",
    "gemini",
    "grok",
    "opencode-cli",
    "antigravity-sdk",
    "antigravity-cli",
)



def _parse_llm_provider_arg(value: str) -> str:
    """Parse CLI provider arguments with migration hints for removed providers."""
    normalized = value.strip().lower()
    if normalized in REMOVED_PROVIDERS:
        raise argparse.ArgumentTypeError(REMOVED_PROVIDERS[normalized])
    if normalized not in CLI_PROVIDER_CHOICES:
        allowed = ", ".join(CLI_PROVIDER_CHOICES)
        raise argparse.ArgumentTypeError(
            f"invalid choice: {value!r} (choose from {allowed})"
        )
    return normalized


class LLMConfig(BaseSettings):
    """
    LLM configuration for ChunkHound deep research.

    Note: At the application level, Config() applies this final precedence order:
      CLI args > explicit --config file > local .chunkhound.json > env vars > defaults.
    Within this class, pydantic-settings applies: init kwargs > env vars > defaults.

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
            "Model for utility operations (query expansion, follow-ups, classification)"
        ),
    )

    synthesis_model: str = Field(
        default="",  # Will be set by get_default_models() if empty
        description="Model for final synthesis (large context analysis)",
    )

    codex_reasoning_effort: ReasoningEffortLiteral | None = Field(
        default=None,
        description=(
            "Default reasoning effort forwarded to codex-cli, grok, openai, "
            "and opencode-cli providers"
        ),
    )
    codex_reasoning_effort_utility: ReasoningEffortLiteral | None = Field(
        default=None,
        description=(
            "Reasoning effort override for utility-stage operations "
            "(codex-cli, grok, openai, opencode-cli)"
        ),
    )
    codex_reasoning_effort_synthesis: ReasoningEffortLiteral | None = Field(
        default=None,
        description=(
            "Reasoning effort override for synthesis-stage operations "
            "(codex-cli, grok, openai, opencode-cli)"
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
            "Unset means no role-specific effort override."
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
            "Unset means no role-specific effort override."
        ),
    )

    # Anthropic Extended Thinking Configuration
    anthropic_thinking_enabled: bool = Field(
        default=False,
        description=(
            "Enable Anthropic extended thinking (shows Claude's reasoning process)"
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
            "Opus 4.5, Opus 4.6, Opus 4.7, Opus 4.8, Sonnet 4.6, and Mythos. "
            "low/medium/high on all supported models; max on 4.6+; "
            "xhigh on Opus 4.7/4.8. high is the API default; Sonnet 4.6 "
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

    # Gemini Thinking Configuration
    gemini_thinking_level: str | None = Field(
        default=None,
        description=(
            "Thinking depth for Gemini 3+ series. "
            'One of "low", "medium", "high". '
            "When set, forwarded as ``thinking_level`` to the Google Gen AI SDK."
        ),
    )

    gemini_thinking_budget: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Fixed thinking token budget for Gemini 2.5+ series. "
            "When set, forwarded as ``ThinkingConfig(thinking_budget=...)`` "
            "to the Google Gen AI SDK."
        ),
    )

    api_key: SecretStr | None = Field(
        default=None, description="API key for authentication (provider-specific)"
    )

    base_url: str | None = Field(
        default=None,
        description=(
            "Provider-specific base URL: "
            "OpenAI-compatible: API endpoint (e.g., https://api.example.com/v1)"
        ),
    )
    ssl_verify: bool = Field(
        default=True,
        description=(
            "Verify TLS certificates for requests sent to the configured base_url. "
            "Ignored when base_url is unset."
        ),
    )

    # Internal settings
    timeout: int = Field(
        default=DEFAULT_LLM_TIMEOUT, description="Internal timeout for LLM calls"
    )
    max_retries: int = Field(default=3, description="Internal max retries")
    supports_structured_outputs: bool | None = Field(
        default=None,
        description=(
            "Override structured output support detection. Set to false for "
            "providers whose API does not support OpenAI's native json_schema "
            "response_format (e.g. DeepSeek)"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def reject_removed_providers(cls, data: Any) -> Any:
        """Reject removed providers with actionable migration hints."""
        if not isinstance(data, dict):
            return data
        provider_fields = (
            "provider",
            "utility_provider",
            "synthesis_provider",
            "map_hyde_provider",
            "autodoc_cleanup_provider",
        )
        for field in provider_fields:
            v = data.get(field)
            if isinstance(v, str) and v.lower() in REMOVED_PROVIDERS:
                raise ValueError(f"{REMOVED_PROVIDERS[v.lower()]} (found in '{field}')")
        return data

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
        """Post-initialization hook that runs AFTER field validators.

        This is where we propagate ``self.model`` to role-specific model fields
        and run cross-field validation. Pydantic v2 guarantees this runs after
        all ``@field_validator`` and ``@model_validator(mode="before")`` hooks,
        so ``utility_model``, ``synthesis_model``, and all effort fields are
        already normalized at this point.

        Moved from a ``@model_validator(mode="after")`` to fix the ordering
        dependency where the validator ran before model_post_init (M1 fix).
        """
        self._propagate_default_models()
        self._validate_provider_switches_require_model()
        self._validate_opencode_model_formats()
        self._validate_reasoning_effort_values()

    def _propagate_default_models(self) -> None:
        """Propagate ``self.model`` to role-specific fields when unset."""
        if self.model is not None:
            if not self.utility_model:
                self.utility_model = self.model
            if not self.synthesis_model:
                self.synthesis_model = self.model

    def _validate_provider_switches_require_model(self) -> None:
        """Override providers require an explicit role-specific model.

        Any provider override — even within the same family — requires
        an explicit model so that the role does not accidentally inherit
        a model designed for a different provider.
        """
        resolved_synthesis_provider = self.synthesis_provider or self.provider

        for role, role_provider, role_model in (
            ("map_hyde", self.map_hyde_provider, self.map_hyde_model),
            (
                "autodoc_cleanup",
                self.autodoc_cleanup_provider,
                self.autodoc_cleanup_model,
            ),
        ):
            if (
                role_provider is not None
                and role_provider != resolved_synthesis_provider
                and not role_model
            ):
                raise ValueError(
                    f"{role} provider override requires an explicit {role}_model "
                    f"when switching providers from "
                    f"{resolved_synthesis_provider!r} to {role_provider!r}."
                )

    def _validate_opencode_model_formats(self) -> None:
        """OpenCode CLI models must be in ``provider/model`` format."""
        resolved_synthesis_provider = self.synthesis_provider or self.provider

        for role, provider_name, role_model in (
            ("utility", self.utility_provider or self.provider, self.utility_model),
            ("synthesis", resolved_synthesis_provider, self.synthesis_model),
            (
                "map_hyde",
                self.map_hyde_provider or resolved_synthesis_provider,
                self.map_hyde_model or self.synthesis_model,
            ),
            (
                "autodoc_cleanup",
                self.autodoc_cleanup_provider or resolved_synthesis_provider,
                self.autodoc_cleanup_model or self.synthesis_model,
            ),
        ):
            if provider_name != "opencode-cli":
                continue
            model_name = role_model or ""
            if not model_name or "/" not in model_name:
                raise ValueError(
                    f"opencode-cli requires a model in provider/model format "
                    f"for {role} (e.g., opencode/gpt-5-nano). "
                    f"Got: {model_name!r}"
                )
            model_provider_name, model_part = model_name.split("/", 1)
            if not model_provider_name.strip():
                raise ValueError(
                    f"opencode-cli model for {role} has an empty provider segment "
                    f"before '/'. Got: {model_name!r}"
                )
            if not model_part.strip():
                raise ValueError(
                    f"opencode-cli model for {role} has an empty model segment "
                    f"after '/'. Got: {model_name!r}"
                )

    def _validate_reasoning_effort_values(self) -> None:
        """Validate provider-specific reasoning effort values for each role."""
        for role in ("utility", "synthesis", "map_hyde", "autodoc_cleanup"):
            resolved_provider, _resolved_model, effort = self._resolve_role_config(role)
            if (
                resolved_provider == "openai"
                and effort is not None
                and effort not in OPENAI_REASONING_EFFORTS
            ):
                raise ValueError(
                    f"openai does not support reasoning effort {effort!r} "
                    f"for {role}; use one of {', '.join(OPENAI_REASONING_EFFORTS)}"
                )
            if (
                resolved_provider == "grok"
                and effort is not None
                and effort not in GROK_REASONING_EFFORTS
            ):
                raise ValueError(
                    f"grok does not support reasoning effort {effort!r} "
                    f"for {role}; use one of {', '.join(GROK_REASONING_EFFORTS)}"
                )

    def _resolve_role_config(self, role: str) -> tuple[str, str, str | None]:
        """Resolve provider, model, and effort for a given role.

        Centralizes the fallback chain so that ``model_post_init``,
        ``get_provider_configs``, and external consumers stay in sync
        without manual duplication (fix M2).

        Args:
            role: One of ``"utility"``, ``"synthesis"``, ``"map_hyde"``,
                or ``"autodoc_cleanup"``.

        Returns:
            Tuple of ``(provider, model, reasoning_effort)`` where
            *reasoning_effort* is ``None`` if unset or the provider does
            not support it.
        """
        resolved_synthesis_provider = self.synthesis_provider or self.provider

        if role == "utility":
            provider = self.utility_provider or self.provider
            model = self.utility_model or self._get_default_models_for(provider)[0]
            effort = self.codex_reasoning_effort_utility or self.codex_reasoning_effort
        elif role == "synthesis":
            provider = resolved_synthesis_provider
            model = self.synthesis_model or self._get_default_models_for(provider)[1]
            effort = (
                self.codex_reasoning_effort_synthesis or self.codex_reasoning_effort
            )
        elif role == "map_hyde":
            provider = self.map_hyde_provider or resolved_synthesis_provider
            model = (
                self.map_hyde_model
                or self.synthesis_model
                or self._get_default_models_for(provider)[1]
            )
            # map_hyde does not inherit codex_reasoning_effort (by design)
            effort = self.map_hyde_reasoning_effort
        elif role == "autodoc_cleanup":
            provider = self.autodoc_cleanup_provider or resolved_synthesis_provider
            model = (
                self.autodoc_cleanup_model
                or self.synthesis_model
                or self._get_default_models_for(provider)[1]
            )
            # autodoc_cleanup does not inherit codex_reasoning_effort (by design)
            effort = self.autodoc_cleanup_reasoning_effort
        else:
            raise ValueError(f"Unknown role: {role}")

        if provider not in REASONING_EFFORT_PROVIDERS:
            effort = None

        return provider, model, effort

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
        "gemini_thinking_level",
        mode="before",
    )
    def _normalize_string_fields(cls, v: str | None) -> str | None:  # noqa: N805
        """Normalize string config fields to trimmed lowercase."""
        if v is None:
            return v
        if isinstance(v, str):
            return v.strip().lower()
        raise ValueError(f"Expected str or None, got {type(v).__name__}")

    @field_validator("gemini_thinking_level")
    def validate_gemini_thinking_level(cls, v: str | None) -> str | None:  # noqa: N805
        """Reject unsupported Gemini thinking levels during config validation."""
        if v is None:
            return v
        if v not in GEMINI_THINKING_LEVELS:
            allowed = ", ".join(GEMINI_THINKING_LEVELS)
            raise ValueError(f"gemini_thinking_level must be one of: {allowed}")
        return v

    def _base_url_for_provider(self, provider: str) -> str | None:
        """Return the configured base URL only for providers that support it."""
        if provider in BASE_URL_CAPABLE_LLM_PROVIDERS:
            return self.base_url
        return None

    def build_provider_config(
        self,
        *,
        provider: str,
        model: str,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Build a provider config using the current shared config fields."""
        # Registry and native providers without a baked-in default model
        # (deepseek, grok, gemini) require an explicit model — fail at config
        # consumption time (not during config construction) so that introspection
        # methods like ``get_missing_config`` still work.
        if (
            provider in OPENAI_COMPATIBLE_PROVIDERS
            or provider in NO_DEFAULT_MODEL_PROVIDERS
        ) and not model:
            raise ConfigurationError(
                config_key="llm.model",
                reason=(
                    f"Model is required for '{provider}'. "
                    "Set `llm.model`, CHUNKHOUND_LLM_MODEL, "
                    "or a per-role model override in your configuration."
                ),
            )

        config: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        if self.api_key and provider not in NO_KEY_PROVIDERS:
            config["api_key"] = self.api_key.get_secret_value()

        if base_url := self._base_url_for_provider(provider):
            config["base_url"] = base_url
            config["ssl_verify"] = self.ssl_verify

        if provider in REASONING_EFFORT_PROVIDERS and reasoning_effort:
            config["reasoning_effort"] = reasoning_effort

        if provider == "anthropic":
            config["thinking_enabled"] = self.anthropic_thinking_enabled
            config["thinking_budget_tokens"] = self.anthropic_thinking_budget_tokens
            config["interleaved_thinking"] = self.anthropic_interleaved_thinking
            if self.anthropic_thinking_mode is not None:
                config["thinking_mode"] = self.anthropic_thinking_mode
            if self.anthropic_thinking_display is not None:
                config["thinking_display"] = self.anthropic_thinking_display
            if self.anthropic_effort is not None:
                config["effort"] = self.anthropic_effort
            config["prompt_caching"] = self.anthropic_prompt_caching
            if self.anthropic_cache_ttl is not None:
                config["cache_ttl"] = self.anthropic_cache_ttl
            if self.anthropic_task_budget_tokens is not None:
                config["task_budget_tokens"] = self.anthropic_task_budget_tokens
            if self.anthropic_context_management_enabled:
                config["context_management_enabled"] = True
                if self.anthropic_clear_thinking_keep_turns is not None:
                    config["clear_thinking_keep_turns"] = (
                        self.anthropic_clear_thinking_keep_turns
                    )
                if self.anthropic_clear_tool_uses_trigger_tokens is not None:
                    config["clear_tool_uses_trigger_tokens"] = (
                        self.anthropic_clear_tool_uses_trigger_tokens
                    )
                if self.anthropic_clear_tool_uses_keep is not None:
                    config["clear_tool_uses_keep"] = self.anthropic_clear_tool_uses_keep

        if provider == "gemini":
            if self.gemini_thinking_level is not None:
                config["thinking_level"] = self.gemini_thinking_level
            if self.gemini_thinking_budget is not None:
                config["thinking_budget"] = self.gemini_thinking_budget

        return config

    def get_provider_config_for_role(self, role: str) -> dict[str, Any]:
        """Resolve a single role to the exact provider config used at runtime.

        Centralises role resolution so that consumers (autodoc_cleanup,
        code_mapper, get_provider_configs) never duplicate the fallback chain.

        Args:
            role: One of ``"utility"``, ``"synthesis"``, ``"map_hyde"``,
                or ``"autodoc_cleanup"``.

        Returns:
            dict with keys ``provider``, ``model``, and optionally
            ``reasoning_effort``, ``supports_structured_outputs``, and
            Anthropic-specific keys when ``provider == "anthropic"``.

        Raises:
            ValueError: If *role* is not one of the four recognised roles.
        """
        provider, model, effort = self._resolve_role_config(role)
        role_config = self.build_provider_config(
            provider=provider,
            model=model,
            reasoning_effort=effort,
        )

        # Propagate supports_structured_outputs to primary roles unconditionally
        # and to secondary roles only when their resolved provider matches
        # the synthesis provider exactly. Explicit overrides to a different
        # provider do not inherit capability flags — the provider uses its
        # own registry default (e.g. False for DeepSeek).
        #
        # We use exact match here (not _provider_family) because registry
        # providers like DeepSeek and Grok share the same "openai-compatible"
        # family but may have different capability support. Model inheritance
        # (in _configured_model_for_role) still uses family match since
        # OpenAI-compatible models share the same API surface.
        if self.supports_structured_outputs is not None and (
            role in {"utility", "synthesis"}
            or provider == (self.synthesis_provider or self.provider)
        ):
            role_config["supports_structured_outputs"] = (
                self.supports_structured_outputs
            )

        return role_config

    def get_provider_configs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Get provider-specific configuration dictionaries.

        Returns:
            Tuple of (utility_config, synthesis_config)
        """
        utility_config = self.get_provider_config_for_role("utility")
        synthesis_config = self.get_provider_config_for_role("synthesis")
        return utility_config, synthesis_config

    @staticmethod
    def _get_default_models_for(provider: LLMProviderLiteral) -> tuple[str, str]:
        """Get default utility and synthesis model names for a provider."""
        # Registry and native providers with no baked-in default:
        # user must set llm.model (or CHUNKHOUND_LLM_MODEL) explicitly.
        if (
            provider in OPENAI_COMPATIBLE_PROVIDERS
            or provider in NO_DEFAULT_MODEL_PROVIDERS
        ):
            return ("", "")

        # Provider-specific smart defaults
        if provider == "openai":
            return ("gpt-5-nano", "gpt-5")
        elif provider == "ollama":
            # Ollama: use same model for both (local deployment)
            return ("llama3.2", "llama3.2")
        elif provider == "claude-code-cli":
            # Claude Code CLI: shared Claude Haiku sentinel for both roles;
            # the CLI resolves ``haiku`` to the latest available model.
            return (CLAUDE_HAIKU_SENTINEL, CLAUDE_HAIKU_SENTINEL)
        elif provider == "codex-cli":
            # Codex CLI: nominal label; require explicit model if desired
            return ("codex", "codex")
        elif provider == "anthropic":
            # Anthropic intentionally uses Claude Haiku for both utility and
            # synthesis. Haiku is capable enough for synthesis and is Anthropic's
            # cheapest Claude model; Anthropic has no true low-cost utility tier.
            return (CLAUDE_HAIKU_SENTINEL, CLAUDE_HAIKU_SENTINEL)
        elif provider == "opencode-cli":
            # OpenCode CLI: No universal default — model depends on user config.
            # User must set model in provider/model format.
            return ("", "")
        elif provider in ("antigravity-sdk", "antigravity-cli"):
            return ("gemini-3.5-flash", "gemini-3.1-pro")
        else:
            # Type-level exhaustiveness check — mypy will flag if a new
            # LLMProviderLiteral variant is added without a matching branch.
            assert_never(provider)

    def get_default_models(self) -> tuple[str, str]:
        """
        Get default model names for utility and synthesis based on resolved providers.

        Returns:
            Tuple of (utility_model, synthesis_model)
        """
        resolved_utility_provider = self.utility_provider or self.provider
        resolved_synthesis_provider = self.synthesis_provider or self.provider
        return (
            self._get_default_models_for(resolved_utility_provider)[0],
            self._get_default_models_for(resolved_synthesis_provider)[1],
        )

    def _provider_family(self, provider: str) -> str:
        """Return the compatibility family for a provider.

        Providers in the same family can share model inheritance and
        capability flags (e.g., supports_structured_outputs) without
        requiring explicit role-specific overrides.
        """
        if provider in OPENAI_COMPATIBLE_PROVIDERS:
            return "openai-compatible"
        return provider

    def _resolved_provider_for_role(self, role: str) -> str:
        """Resolve the effective provider for a role."""
        resolved_synth = self.synthesis_provider or self.provider
        if role == "utility":
            return self.utility_provider or self.provider
        if role == "synthesis":
            return resolved_synth
        if role == "map_hyde":
            return self.map_hyde_provider or resolved_synth
        if role == "autodoc_cleanup":
            return self.autodoc_cleanup_provider or resolved_synth
        raise ValueError(f"Unknown LLM role: {role}")

    def _role_uses_synthesis_provider_fallback(self, role: str) -> bool:
        """Return whether a role falls back to synthesis provider settings."""
        if role == "map_hyde":
            return self.map_hyde_provider is None
        if role == "autodoc_cleanup":
            return self.autodoc_cleanup_provider is None
        return False

    def _explicit_model_for_role(self, role: str) -> str | None:
        """Return the explicit model selection for a role, if any.

        Primary roles (utility/synthesis) fall back to ``self.model``.
        Secondary roles do not — they inherit via synthesis rather than
        the global model field.
        """
        if role == "utility":
            return self.utility_model or self.model or None
        if role == "synthesis":
            return self.synthesis_model or self.model or None
        if role == "map_hyde":
            return self.map_hyde_model or None
        if role == "autodoc_cleanup":
            return self.autodoc_cleanup_model or None
        raise ValueError(f"Unknown LLM role: {role}")

    def _configured_model_for_role(self, role: str) -> str | None:
        """Return a user-configured model for a role without injecting defaults."""
        explicit_model = self._explicit_model_for_role(role)
        if explicit_model:
            return explicit_model

        if role in {"utility", "synthesis"}:
            return None

        if self._role_uses_synthesis_provider_fallback(role):
            return self._configured_model_for_role("synthesis")

        synthesis_provider = self._resolved_provider_for_role("synthesis")
        role_provider = self._resolved_provider_for_role(role)
        if self._provider_family(role_provider) != self._provider_family(
            synthesis_provider
        ):
            return None

        return self._configured_model_for_role("synthesis")

    def resolve_model_for_role(self, role: str) -> str | None:
        """Resolve the effective model for a role, injecting provider defaults.

        Delegates the family-aware cross-role fallback to
        ``_configured_model_for_role`` and only adds provider-specific
        defaults (``_get_default_models_for``) for primary roles.
        """
        configured = self._configured_model_for_role(role)
        if configured is not None:
            return configured

        if role in {"utility", "synthesis"}:
            provider = self._resolved_provider_for_role(role)
            defaults = self._get_default_models_for(provider)  # type: ignore[arg-type]
            return defaults[0] if role == "utility" else defaults[1]

        return None

    def _is_custom_openai_endpoint(self) -> bool:
        """Return whether this config targets a non-official OpenAI-compatible endpoint."""
        return not is_official_openai_endpoint(self.base_url)

    def _require_explicit_model_for_custom_openai(
        self, roles: tuple[str, ...]
    ) -> list[str]:
        """Return roles that rely on a custom OpenAI endpoint without valid model selection."""
        if not self._is_custom_openai_endpoint():
            return []

        missing_roles: list[str] = []
        for role in roles:
            resolved_provider = self._resolved_provider_for_role(role)
            if resolved_provider not in OPENAI_COMPATIBLE_LLM_PROVIDERS:
                continue
            if self._configured_model_for_role(role) is None:
                missing_roles.append(role)

        return missing_roles

    def _require_explicit_model_for_registry_providers(
        self, roles: tuple[str, ...]
    ) -> list[str]:
        """Return roles that use a registry provider without an explicit model."""
        missing_roles: list[str] = []
        for role in roles:
            resolved_provider = self._resolved_provider_for_role(role)
            if resolved_provider not in OPENAI_COMPATIBLE_PROVIDERS:
                continue
            if self._configured_model_for_role(role) is not None:
                continue
            missing_roles.append(role)

        return missing_roles

    def _provider_requires_api_key(self, provider: str) -> bool:
        """Return whether a provider requires an API key for the current config.

        The current config model has a single top-level base_url shared by
        OpenAI-compatible roles. Non-OpenAI-compatible providers ignore it.
        """
        if provider in {"claude-code-cli", "codex-cli", "opencode-cli", "antigravity-cli"}:
            return False
        if provider in OPENAI_COMPATIBLE_LLM_PROVIDERS:
            return is_official_openai_endpoint(self._base_url_for_provider(provider))
        return True

    def is_provider_configured(self) -> bool:
        """
        Check if the selected provider is properly configured.

        Returns:
            True if provider is properly configured
        """
        return not self.get_missing_config()

    def get_missing_config_for_roles(self, roles: tuple[str, ...]) -> list[str]:
        """Get missing config for a specific set of roles."""
        missing: list[str] = []

        if any(
            self._provider_requires_api_key(self._resolved_provider_for_role(role))
            for role in roles
        ):
            if not self.api_key:
                missing.append("api_key (set CHUNKHOUND_LLM_API_KEY)")

        registry_provider_roles = self._require_explicit_model_for_registry_providers(
            roles
        )
        if registry_provider_roles:
            roles_text = ", ".join(registry_provider_roles)
            missing.append(
                f"explicit model selection required for registry provider roles: {roles_text}"
            )

        custom_endpoint_roles = [
            role
            for role in self._require_explicit_model_for_custom_openai(roles)
            if role not in registry_provider_roles
        ]
        if custom_endpoint_roles:
            roles_text = ", ".join(custom_endpoint_roles)
            missing.append(
                "explicit model selection required for custom OpenAI-compatible "
                f"endpoint roles: {roles_text}"
            )

        # Native providers with no baked-in default require explicit model
        no_default_model_roles = [
            role
            for role in roles
            if self._resolved_provider_for_role(role) in NO_DEFAULT_MODEL_PROVIDERS
            and self._configured_model_for_role(role) is None
        ]
        if no_default_model_roles:
            roles_text = ", ".join(no_default_model_roles)
            missing.append(
                f"explicit model selection required for gemini roles: {roles_text}"
            )

        return missing

    def get_missing_config(self) -> list[str]:
        """
        Get list of missing required configuration.

        Returns:
            List of missing configuration parameter names
        """
        return self.get_missing_config_for_roles(
            ("utility", "synthesis", "map_hyde", "autodoc_cleanup")
        )

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add LLM-related CLI arguments."""
        try:
            boolean_optional_action = argparse.BooleanOptionalAction
        except AttributeError:  # pragma: no cover - older Python
            boolean_optional_action = None

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
        if boolean_optional_action is not None:
            parser.add_argument(
                "--llm-ssl-verify",
                action=boolean_optional_action,
                default=None,
                help=(
                    "Verify TLS certificates for requests sent to --llm-base-url. "
                    "Ignored when llm.base_url is unset."
                ),
            )

        parser.add_argument(
            "--llm-provider",
            type=_parse_llm_provider_arg,
            metavar="{" + ",".join(CLI_PROVIDER_CHOICES) + "}",
            help="Default LLM provider for both roles",
        )

        parser.add_argument(
            "--llm-utility-provider",
            type=_parse_llm_provider_arg,
            metavar="{" + ",".join(CLI_PROVIDER_CHOICES) + "}",
            help="Override LLM provider for utility operations",
        )

        parser.add_argument(
            "--llm-synthesis-provider",
            type=_parse_llm_provider_arg,
            metavar="{" + ",".join(CLI_PROVIDER_CHOICES) + "}",
            help="Override LLM provider for synthesis operations",
        )

        parser.add_argument(
            "--llm-codex-reasoning-effort",
            choices=["minimal", "low", "medium", "high", "xhigh"],
            help=(
                "Codex CLI reasoning effort (thinking depth) "
                "when using codex-cli, opencode-cli, or openai provider "
                "(openai and grok exclude xhigh)"
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
            "--llm-supports-structured-outputs",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=(
                "Override structured output support detection. "
                "Set to false for providers without native json_schema "
                "response_format (e.g., DeepSeek)."
            ),
        )

        parser.add_argument(
            "--llm-map-hyde-provider",
            type=_parse_llm_provider_arg,
            metavar="{" + ",".join(CLI_PROVIDER_CHOICES) + "}",
            help="Override provider for Code Mapper HyDE planning (falls back to synthesis)",
        )
        parser.add_argument(
            "--llm-map-hyde-model",
            help=(
                "Override model for Code Mapper HyDE planning (falls back to synthesis)"
            ),
        )
        parser.add_argument(
            "--llm-map-hyde-reasoning-effort",
            choices=["minimal", "low", "medium", "high", "xhigh"],
            help=(
                "Override reasoning effort for Code Mapper HyDE planning "
                "(openai and grok exclude xhigh)"
            ),
        )

        parser.add_argument(
            "--llm-autodoc-cleanup-provider",
            type=_parse_llm_provider_arg,
            metavar="{" + ",".join(CLI_PROVIDER_CHOICES) + "}",
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
                "(openai and grok exclude xhigh)"
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
        parser.add_argument(
            "--llm-gemini-thinking-level",
            choices=["low", "medium", "high"],
            help="Thinking depth for Gemini 3+ series (low, medium, high)",
        )
        parser.add_argument(
            "--llm-gemini-thinking-budget",
            type=int,
            help="Token budget for Gemini 2.5+ series thinking",
        )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load LLM config from environment variables."""
        config: dict[str, Any] = {}

        if api_key := os.getenv("CHUNKHOUND_LLM_API_KEY"):
            config["api_key"] = api_key
        if base_url := os.getenv("CHUNKHOUND_LLM_BASE_URL"):
            config["base_url"] = base_url
        if ssl_verify_raw := os.getenv("CHUNKHOUND_LLM_SSL_VERIFY"):
            if (ssl_verify := _parse_env_bool(ssl_verify_raw)) is not None:
                config["ssl_verify"] = ssl_verify
        if provider := os.getenv("CHUNKHOUND_LLM_PROVIDER"):
            config["provider"] = provider
        if model := os.getenv("CHUNKHOUND_LLM_MODEL"):
            config["model"] = model
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

        sso_val = _bool_env("CHUNKHOUND_LLM_SUPPORTS_STRUCTURED_OUTPUTS")
        if sso_val is not None:
            config["supports_structured_outputs"] = sso_val

        for env_name, key in bool_fields:
            bool_val = _bool_env(env_name)
            if bool_val is not None:
                config[key] = bool_val

        str_fields = (
            ("CHUNKHOUND_LLM_ANTHROPIC_THINKING_MODE", "anthropic_thinking_mode"),
            ("CHUNKHOUND_LLM_ANTHROPIC_THINKING_DISPLAY", "anthropic_thinking_display"),
            ("CHUNKHOUND_LLM_ANTHROPIC_EFFORT", "anthropic_effort"),
            ("CHUNKHOUND_LLM_ANTHROPIC_CACHE_TTL", "anthropic_cache_ttl"),
            ("CHUNKHOUND_LLM_GEMINI_THINKING_LEVEL", "gemini_thinking_level"),
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
            (
                "CHUNKHOUND_LLM_GEMINI_THINKING_BUDGET",
                "gemini_thinking_budget",
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
        if hasattr(args, "llm_ssl_verify") and args.llm_ssl_verify is not None:
            overrides["ssl_verify"] = args.llm_ssl_verify
        if hasattr(args, "llm_provider") and args.llm_provider:
            overrides["provider"] = args.llm_provider
        if hasattr(args, "llm_utility_provider") and args.llm_utility_provider:
            overrides["utility_provider"] = args.llm_utility_provider
        if hasattr(args, "llm_synthesis_provider") and args.llm_synthesis_provider:
            overrides["synthesis_provider"] = args.llm_synthesis_provider

        sso = getattr(args, "llm_supports_structured_outputs", None)
        if sso is not None:
            overrides["supports_structured_outputs"] = sso

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

        gemini_flag_map = {
            "llm_gemini_thinking_level": "gemini_thinking_level",
            "llm_gemini_thinking_budget": "gemini_thinking_budget",
        }
        for arg_name, config_key in gemini_flag_map.items():
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
            f"base_url={self.base_url}, "
            f"ssl_verify={self.ssl_verify})"
        )
