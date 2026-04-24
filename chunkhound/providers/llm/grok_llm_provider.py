"""Grok (xAI) LLM provider implementation for ChunkHound deep research.

Uses xAI's Grok API via OpenAI-compatible client.
Supports structured outputs, tool calling, and reasoning models.

API: https://api.x.ai/v1/chat/completions
Docs: https://docs.x.ai/docs/models
Auth: API key from https://console.x.ai
"""

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.providers.llm.openai_compatible_provider import OpenAICompatibleProvider


class GrokLLMProvider(OpenAICompatibleProvider):
    """xAI Grok LLM provider using OpenAI-compatible API.

    Supports Grok models with reasoning, structured outputs, and tool calling.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "grok-4-1-fast-reasoning",
        base_url: str | None = None,
        ssl_verify: bool = True,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_retries: int = 3,
    ):
        """Initialize Grok LLM provider.

        Args:
            api_key: xAI API key (passed from config)
            model: Model name (default: "grok-4-1-fast-reasoning")
            base_url: Base URL (defaults to https://api.x.ai/v1)
            ssl_verify: Whether to verify SSL certificates
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts
        """

        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            ssl_verify=ssl_verify,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _get_default_base_url(self) -> str:
        """Get the default xAI API base URL."""
        return "https://api.x.ai/v1"

    def _get_provider_name(self) -> str:
        """Get the provider name."""
        return "grok"

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations.

        Returns:
            5 for Grok (similar to Anthropic)
        """
        return 5
