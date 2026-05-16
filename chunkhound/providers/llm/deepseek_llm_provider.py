"""DeepSeek LLM provider implementation for ChunkHound deep research.

Uses DeepSeek's API via OpenAI-compatible client.
Does NOT support native ``response_format`` with ``type: "json_schema"``
(OpenAI Structured Outputs) — falls back to prompt-based schema injection.

API: https://api.deepseek.com/chat/completions
Docs: https://platform.deepseek.com/api-docs
Auth: API key from https://platform.deepseek.com
"""

from chunkhound.providers.llm.openai_compatible_provider import OpenAICompatibleProvider


class DeepSeekLLMProvider(OpenAICompatibleProvider):
    """DeepSeek LLM provider using OpenAI-compatible API.

    DeepSeek's API does not support OpenAI's native ``json_schema``
    response format. This provider overrides ``_supports_structured_outputs``
    so that ``complete_structured()`` injects the schema into the system
    prompt instead.
    """

    _supports_structured_outputs: bool = False

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-v4-flash",
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        supports_structured_outputs: bool | None = None,
    ):
        """Initialize DeepSeek LLM provider.

        Args:
            api_key: DeepSeek API key. When None, AsyncOpenAI client falls back to
                OPENAI_API_KEY.
            model: Model name (default: "deepseek-v4-flash")
            base_url: Base URL (defaults to https://api.deepseek.com)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts
            supports_structured_outputs: Override class-level structured
                output support flag (default False for DeepSeek)
        """
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            supports_structured_outputs=supports_structured_outputs,
        )

    def _get_default_base_url(self) -> str:
        """Get the default DeepSeek API base URL."""
        return "https://api.deepseek.com"

    def _get_provider_name(self) -> str:
        """Get the provider name."""
        return "deepseek"

    def _get_max_completion_tokens_param_name(self) -> str:
        """DeepSeek chat completions use max_tokens, not max_completion_tokens."""
        return "max_tokens"

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations.

        Returns:
            10 for DeepSeek (individual developer use).
            Safe for typical usage; production can scale higher.
        """
        return 10
