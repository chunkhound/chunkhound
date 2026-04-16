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

from chunkhound.core.config.openai_utils import is_official_openai_endpoint

DEFAULT_LLM_TIMEOUT = 120
OPENAI_COMPATIBLE_LLM_PROVIDERS = {"openai", "grok"}
BASE_URL_CAPABLE_LLM_PROVIDERS = OPENAI_COMPATIBLE_LLM_PROVIDERS | {"anthropic"}

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
    "anthropic",
    "gemini",
    "grok",
    "opencode-cli",
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


def _parse_env_bool(value: str) -> bool | None:
    """Parse a boolean environment variable value."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


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
        "claude-code-cli",
        "codex-cli",
        "gemini",
        "anthropic",
        "grok",
        "opencode-cli",
    ] = Field(
        default="openai",
        description="Default LLM provider for both roles (utility, synthesis)",
    )

    # Optional per-role overrides (utility vs synthesis)
    utility_provider: (
        Literal[
            "openai",
            "claude-code-cli",
            "codex-cli",
            "anthropic",
            "gemini",
            "grok",
            "opencode-cli",
        ]
        | None
    ) = Field(default=None, description="Override provider for utility ops")

    synthesis_provider: (
        Literal[
            "openai",
            "claude-code-cli",
            "codex-cli",
            "anthropic",
            "gemini",
            "grok",
            "opencode-cli",
        ]
        | None
    ) = Field(default=None, description="Override provider for synthesis ops")

    # Model Configuration (dual-model architecture)
    model: str | None = Field(
        default=None,
        description="Convenience field to set both utility and synthesis models to the same value",
    )

    utility_model: str = Field(
        default="",  # Will be set by get_default_models() if empty
        description="Model for utility operations (query expansion, follow-ups, classification)",
    )

    synthesis_model: str = Field(
        default="",  # Will be set by get_default_models() if empty
        description="Model for final synthesis (large context analysis)",
    )

    codex_reasoning_effort: (
        Literal["minimal", "low", "medium", "high", "xhigh"] | None
    ) = Field(
        default=None,
        description="Default Codex CLI reasoning effort (Responses API thinking level)",
    )
    codex_reasoning_effort_utility: (
        Literal["minimal", "low", "medium", "high", "xhigh"] | None
    ) = Field(
        default=None,
        description="Codex CLI reasoning effort override for utility-stage operations",
    )
    codex_reasoning_effort_synthesis: (
        Literal["minimal", "low", "medium", "high", "xhigh"] | None
    ) = Field(
        default=None,
        description="Codex CLI reasoning effort override for synthesis-stage operations",
    )

    map_hyde_provider: (
        Literal[
            "openai",
            "claude-code-cli",
            "codex-cli",
            "gemini",
            "anthropic",
            "grok",
            "opencode-cli",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Override provider for Code Mapper HyDE planning (points-of-interest overview). "
            "Falls back to the synthesis provider when unset."
        ),
    )

    map_hyde_model: str | None = Field(
        default=None,
        description=(
            "Override model for Code Mapper HyDE planning (points-of-interest overview). "
            "Falls back to the synthesis model when unset."
        ),
    )

    map_hyde_reasoning_effort: (
        Literal[
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Codex/OpenAI reasoning effort override for Code Mapper HyDE planning. "
            "Falls back to synthesis reasoning effort when unset."
        ),
    )

    autodoc_cleanup_provider: (
        Literal[
            "openai",
            "claude-code-cli",
            "codex-cli",
            "gemini",
            "anthropic",
            "grok",
            "opencode-cli",
        ]
        | None
    ) = Field(
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

    autodoc_cleanup_reasoning_effort: (
        Literal[
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Codex/OpenAI reasoning effort override for AutoDoc LLM cleanup. "
            "Falls back to synthesis reasoning effort when unset."
        ),
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
                raise ValueError(
                    f"{REMOVED_PROVIDERS[v.lower()]} (found in '{field}')"
                )
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
        """Post-initialization hook to handle model field mapping."""
        # If model is provided, set both utility_model and synthesis_model
        if self.model is not None:
            if not self.utility_model:
                self.utility_model = self.model
            if not self.synthesis_model:
                self.synthesis_model = self.model

    @field_validator(
        "codex_reasoning_effort",
        "codex_reasoning_effort_utility",
        "codex_reasoning_effort_synthesis",
        "map_hyde_reasoning_effort",
        "autodoc_cleanup_reasoning_effort",
        mode="before",
    )
    def normalize_codex_effort(cls, v: str | None) -> str | None:  # noqa: N805
        """Normalize Codex effort strings."""
        if v is None:
            return v
        if isinstance(v, str):
            return v.strip().lower()
        # This branch is technically reachable if v is not None and not a str
        # but pydantic should prevent that based on the type hint.
        # However, for type safety we return v as is.
        return v

    def _get_default_models_for_provider(self, provider: str) -> tuple[str, str]:
        """Return the provider-specific utility/synthesis defaults."""
        if provider == "openai":
            return ("gpt-5-nano", "gpt-5")
        elif provider == "claude-code-cli":
            return ("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001")
        elif provider == "codex-cli":
            return ("codex", "codex")
        elif provider == "gemini":
            return ("gemini-3-pro-preview", "gemini-3-pro-preview")
        elif provider == "anthropic":
            return ("claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929")
        elif provider == "grok":
            return ("grok-4-1-fast-reasoning", "grok-4-1-fast-reasoning")
        elif provider == "opencode-cli":
            return ("opencode/grok-code", "opencode/grok-code")
        return ("gpt-5-nano", "gpt-5")

    def build_provider_config(
        self,
        *,
        provider: str,
        model: str,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Build a provider config using the current shared config fields."""
        config: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        if self.api_key:
            config["api_key"] = self.api_key.get_secret_value()

        if base_url := self._base_url_for_provider(provider):
            config["base_url"] = base_url
            config["ssl_verify"] = self.ssl_verify

        if provider in ("codex-cli", "openai") and reasoning_effort:
            config["reasoning_effort"] = reasoning_effort

        if provider == "anthropic":
            config["thinking_enabled"] = self.anthropic_thinking_enabled
            config["thinking_budget_tokens"] = self.anthropic_thinking_budget_tokens
            config["interleaved_thinking"] = self.anthropic_interleaved_thinking
            if self.anthropic_effort:
                config["effort"] = self.anthropic_effort
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
                    config["clear_tool_uses_keep"] = (
                        self.anthropic_clear_tool_uses_keep
                    )

        return config

    def get_provider_configs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Get provider-specific configuration dictionaries for utility and synthesis models.

        Returns:
            Tuple of (utility_config, synthesis_config)
        """
        # Resolve providers per-role
        resolved_utility_provider = self.utility_provider or self.provider
        resolved_synthesis_provider = self.synthesis_provider or self.provider
        utility_default, _ = self._get_default_models_for_provider(
            resolved_utility_provider
        )
        _, synthesis_default = self._get_default_models_for_provider(
            resolved_synthesis_provider
        )

        def _codex_effort_for(role: str) -> str | None:
            default_effort = self.codex_reasoning_effort
            if role == "utility":
                return self.codex_reasoning_effort_utility or default_effort
            if role == "synthesis":
                return self.codex_reasoning_effort_synthesis or default_effort
            return default_effort

        utility_config = self.build_provider_config(
            provider=resolved_utility_provider,
            model=self.utility_model or utility_default,
            reasoning_effort=_codex_effort_for("utility"),
        )
        synthesis_config = self.build_provider_config(
            provider=resolved_synthesis_provider,
            model=self.synthesis_model or synthesis_default,
            reasoning_effort=_codex_effort_for("synthesis"),
        )

        return utility_config, synthesis_config

    def get_default_models(self) -> tuple[str, str]:
        """
        Get default model names for utility and synthesis based on provider.

        Returns:
            Tuple of (utility_model, synthesis_model)
        """
        return self._get_default_models_for_provider(self.provider)

    def _provider_family(self, provider: str) -> str:
        """Return the compatibility family for a provider."""
        if provider in OPENAI_COMPATIBLE_LLM_PROVIDERS:
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
        """Return the explicit model selection for a role, if any."""
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
        """Resolve the effective model for a role.

        For roles that fall back to synthesis settings, only inherit the synthesis
        model when the resolved provider stays in the same compatibility family.
        """
        explicit_model = self._explicit_model_for_role(role)
        if explicit_model:
            return explicit_model

        if role in {"utility", "synthesis"}:
            provider = self._resolved_provider_for_role(role)
            defaults = self._get_default_models_for_provider(provider)
            return defaults[0] if role == "utility" else defaults[1]

        if self._role_uses_synthesis_provider_fallback(role):
            return self.resolve_model_for_role("synthesis")

        synthesis_provider = self._resolved_provider_for_role("synthesis")
        role_provider = self._resolved_provider_for_role(role)
        if self._provider_family(role_provider) != self._provider_family(
            synthesis_provider
        ):
            return None

        return self.resolve_model_for_role("synthesis")

    def _base_url_for_provider(self, provider: str) -> str | None:
        """Return the configured base URL only for providers that support it."""
        if provider in BASE_URL_CAPABLE_LLM_PROVIDERS:
            return self.base_url
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

    def _require_explicit_model_for_cross_family_role_overrides(
        self, roles: tuple[str, ...]
    ) -> list[str]:
        """Return secondary roles that override to an incompatible provider family."""
        missing_roles: list[str] = []
        synthesis_provider = self._resolved_provider_for_role("synthesis")

        for role in roles:
            if role in {"utility", "synthesis"}:
                continue
            if self._role_uses_synthesis_provider_fallback(role):
                continue

            role_provider = self._resolved_provider_for_role(role)
            if self._provider_family(role_provider) == self._provider_family(
                synthesis_provider
            ):
                continue
            if self._configured_model_for_role(role) is not None:
                continue

            missing_roles.append(role)

        return missing_roles

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

        custom_endpoint_roles = self._require_explicit_model_for_custom_openai(roles)
        if custom_endpoint_roles:
            roles_text = ", ".join(custom_endpoint_roles)
            missing.append(
                "explicit model selection required for custom OpenAI-compatible "
                f"endpoint roles: {roles_text}"
            )

        cross_family_roles = [
            role
            for role in self._require_explicit_model_for_cross_family_role_overrides(
                roles
            )
            if role not in custom_endpoint_roles
        ]
        if cross_family_roles:
            roles_text = ", ".join(cross_family_roles)
            missing.append(
                "explicit provider-compatible model required for roles: "
                f"{roles_text}"
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

    def _provider_requires_api_key(self, provider: str) -> bool:
        """Return whether a provider requires an API key for the current config.

        The current config model has a single top-level base_url shared by
        OpenAI-compatible roles. Non-OpenAI-compatible providers ignore it.
        """
        if provider in {"claude-code-cli", "codex-cli", "opencode-cli"}:
            return False
        if provider in OPENAI_COMPATIBLE_LLM_PROVIDERS:
            return is_official_openai_endpoint(self._base_url_for_provider(provider))
        return True

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add LLM-related CLI arguments."""
        try:
            boolean_optional_action = argparse.BooleanOptionalAction
        except AttributeError:  # pragma: no cover - older Python
            boolean_optional_action = None

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
            help="Codex CLI reasoning effort (thinking depth) when using codex-cli provider",
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
            type=_parse_llm_provider_arg,
            metavar="{" + ",".join(CLI_PROVIDER_CHOICES) + "}",
            help="Override provider for Code Mapper HyDE planning (falls back to synthesis)",
        )
        parser.add_argument(
            "--llm-map-hyde-model",
            help="Override model for Code Mapper HyDE planning (falls back to synthesis)",
        )
        parser.add_argument(
            "--llm-map-hyde-reasoning-effort",
            choices=["minimal", "low", "medium", "high", "xhigh"],
            help="Override Codex/OpenAI reasoning effort for Code Mapper HyDE planning",
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
            help="Override Codex/OpenAI reasoning effort for AutoDoc LLM cleanup",
        )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load LLM config from environment variables."""
        config = {}

        if api_key := os.getenv("CHUNKHOUND_LLM_API_KEY"):
            config["api_key"] = api_key
        if base_url := os.getenv("CHUNKHOUND_LLM_BASE_URL"):
            config["base_url"] = base_url
        if ssl_verify_raw := os.getenv("CHUNKHOUND_LLM_SSL_VERIFY"):
            if (ssl_verify := _parse_env_bool(ssl_verify_raw)) is not None:
                config["ssl_verify"] = ssl_verify
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
