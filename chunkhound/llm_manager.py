"""LLM Manager with factory pattern for ChunkHound deep research."""

from typing import Any

from loguru import logger

from chunkhound.interfaces.llm_provider import LLMProvider
from chunkhound.providers.llm.anthropic_llm_provider import AnthropicLLMProvider
from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider
from chunkhound.providers.llm.claude_code_cli_provider import ClaudeCodeCLIProvider
from chunkhound.providers.llm.openai_llm_provider import OpenAILLMProvider
from chunkhound.providers.llm.gemini_llm_provider import GeminiLLMProvider


class LLMManager:
    """Manager for LLM providers with factory pattern.

    Supports dual-model architecture:
    - Utility provider: For fast, cheap operations (query expansion, follow-ups)
    - Synthesis provider: For high-quality, large-context operations (final analysis)
    """

    # Registry of available providers
    _providers: dict[str, type[LLMProvider] | Any] = {
        "openai": OpenAILLMProvider,
        "anthropic": AnthropicLLMProvider,
        "claude-code-cli": ClaudeCodeCLIProvider,
        "codex-cli": CodexCLIProvider,
        "gemini": GeminiLLMProvider,
    }

    def __init__(
        self, utility_config: dict[str, Any], synthesis_config: dict[str, Any]
    ):
        """Initialize LLM manager with dual providers.

        Args:
            utility_config: Configuration for utility operations provider
            synthesis_config: Configuration for synthesis operations provider
        """
        self._utility_config = utility_config
        self._synthesis_config = synthesis_config
        self._utility_provider: LLMProvider | None = None
        self._synthesis_provider: LLMProvider | None = None

        # Initialize both providers
        self._initialize_utility_provider()
        self._initialize_synthesis_provider()

    def _create_provider(self, config: dict[str, Any]) -> LLMProvider:
        """Create a provider instance from configuration.

        Args:
            config: Provider configuration dictionary

        Returns:
            Initialized LLMProvider instance

        Raises:
            ValueError: If provider name is unknown
        """
        provider_name = config.get("provider", "openai")

        if provider_name not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ValueError(
                f"Unknown LLM provider: {provider_name}. "
                f"Available providers: {available}"
            )

        provider_class = self._providers[provider_name]

        try:
            # Build provider initialization parameters
            provider_kwargs = {
                "api_key": config.get("api_key"),
                "model": config.get("model", "gpt-5-nano"),
                "base_url": config.get("base_url"),
                "timeout": config.get("timeout", 60),
                "max_retries": config.get("max_retries", 3),
            }

            # Pass reasoning_effort to OpenAI and Codex providers
            if provider_name in ("openai", "codex-cli"):
                effort = config.get("reasoning_effort")
                if effort:
                    provider_kwargs["reasoning_effort"] = effort
            elif provider_name == "anthropic":
                # Add Anthropic configuration
                # Extended thinking
                provider_kwargs["thinking_enabled"] = config.get(
                    "thinking_enabled", False
                )
                provider_kwargs["thinking_budget_tokens"] = config.get(
                    "thinking_budget_tokens", 10000
                )
                provider_kwargs["interleaved_thinking"] = config.get(
                    "interleaved_thinking", False
                )

                # Effort parameter (Opus 4.5 only)
                if effort := config.get("effort"):
                    provider_kwargs["effort"] = effort

                # Context management
                if config.get("context_management_enabled"):
                    provider_kwargs["context_management_enabled"] = True
                    if (
                        keep_turns := config.get("clear_thinking_keep_turns")
                    ) is not None:
                        provider_kwargs["clear_thinking_keep_turns"] = keep_turns
                    if (
                        trigger := config.get("clear_tool_uses_trigger_tokens")
                    ) is not None:
                        provider_kwargs["clear_tool_uses_trigger_tokens"] = trigger
                    if (keep := config.get("clear_tool_uses_keep")) is not None:
                        provider_kwargs["clear_tool_uses_keep"] = keep

            provider = provider_class(**provider_kwargs)
            return provider
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider {provider_name}: {e}")
            raise

    def _initialize_utility_provider(self) -> None:
        """Initialize the utility LLM provider."""
        self._utility_provider = self._create_provider(self._utility_config)
        logger.info(
            f"Initialized utility LLM provider: {self._utility_config.get('provider')} "
            f"with model: {self._utility_provider.model}"
        )

    def _initialize_synthesis_provider(self) -> None:
        """Initialize the synthesis LLM provider."""
        self._synthesis_provider = self._create_provider(self._synthesis_config)
        logger.info(
            f"Initialized synthesis LLM provider: {self._synthesis_config.get('provider')} "
            f"with model: {self._synthesis_provider.model}"
        )

    def get_utility_provider(self) -> LLMProvider:
        """Get the utility LLM provider (for fast operations).

        Returns:
            Utility LLMProvider instance

        Raises:
            ValueError: If provider not initialized
        """
        if self._utility_provider is None:
            raise ValueError("Utility LLM provider not configured.")
        return self._utility_provider

    def get_synthesis_provider(self) -> LLMProvider:
        """Get the synthesis LLM provider (for high-quality operations).

        Returns:
            Synthesis LLMProvider instance

        Raises:
            ValueError: If provider not initialized
        """
        if self._synthesis_provider is None:
            raise ValueError("Synthesis LLM provider not configured.")
        return self._synthesis_provider

    def is_configured(self) -> bool:
        """Check if both LLM providers are configured and available.

        Returns:
            True if both providers are configured
        """
        return (
            self._utility_provider is not None and self._synthesis_provider is not None
        )

    def list_providers(self) -> list[str]:
        """List available LLM providers.

        Returns:
            List of provider names
        """
        return list(self._providers.keys())

    @classmethod
    def register_provider(cls, name: str, provider_class: type[LLMProvider]) -> None:
        """Register a new LLM provider.

        Args:
            name: Provider name
            provider_class: Provider class
        """
        cls._providers[name] = provider_class
        logger.debug(f"Registered LLM provider: {name}")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on both configured providers.

        Returns:
            Health check results for both providers
        """
        results = {}

        if self._utility_provider:
            results["utility"] = await self._utility_provider.health_check()
        else:
            results["utility"] = {
                "status": "not_configured",
                "message": "Utility provider not configured",
            }

        if self._synthesis_provider:
            results["synthesis"] = await self._synthesis_provider.health_check()
        else:
            results["synthesis"] = {
                "status": "not_configured",
                "message": "Synthesis provider not configured",
            }

        return results

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics from both configured providers.

        Returns:
            Usage statistics for both providers
        """
        stats = {}

        if self._utility_provider:
            stats["utility"] = self._utility_provider.get_usage_stats()

        if self._synthesis_provider:
            stats["synthesis"] = self._synthesis_provider.get_usage_stats()

        return stats
