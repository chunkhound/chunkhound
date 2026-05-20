"""Grok (xAI) LLM provider implementation for ChunkHound deep research.

Uses xAI's Grok API via OpenAI-compatible client.
Supports structured outputs, tool calling, and reasoning models.

API: https://api.x.ai/v1/chat/completions
Docs: https://docs.x.ai/docs/models
Auth: API key from https://console.x.ai
"""

from typing import Any

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
        timeout: int = 60,
        max_retries: int = 3,
        supports_structured_outputs: bool | None = None,
        reasoning_effort: str | None = None,
    ):
        """Initialize Grok LLM provider.

        Args:
            api_key: xAI API key (passed from config)
            model: Model name (default: "grok-4-1-fast-reasoning")
            base_url: Base URL (defaults to https://api.x.ai/v1)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts
            supports_structured_outputs: Override class-level structured
                output support flag
            reasoning_effort: Reasoning effort for Grok reasoning models.
                Use None to omit it; supported values are minimal, low,
                medium, and high.
        """

        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            supports_structured_outputs=supports_structured_outputs,
        )
        self._reasoning_effort = reasoning_effort

    def _get_default_base_url(self) -> str:
        """Get the default xAI API base URL."""
        return "https://api.x.ai/v1"

    def _get_provider_name(self) -> str:
        """Get the provider name."""
        return "grok"

    def _build_chat_completion_kwargs(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int,
        timeout: int,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add Grok reasoning_effort without forking parent completion logic."""
        kwargs = super()._build_chat_completion_kwargs(
            messages,
            max_completion_tokens,
            timeout,
            response_format=response_format,
        )
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        return kwargs

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations.

        Returns:
            5 for Grok (similar to Anthropic)
        """
        return 5
