"""LLM Provider Interface for ChunkHound deep research."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Response from LLM completion."""

    content: str
    tokens_used: int
    model: str
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier."""
        ...

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: User prompt
            system: Optional system message
            max_completion_tokens: Maximum completion tokens to generate
            timeout: Optional timeout in seconds for the request

        Returns:
            LLMResponse with content and metadata
        """
        ...

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Generate a structured JSON completion conforming to the given schema.

        Best practice for GPT-5-Nano and modern LLMs: Use response_format with
        json_schema and strict: true to guarantee valid, parseable output.

        Args:
            prompt: User prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system message
            max_completion_tokens: Maximum completion tokens to generate

        Returns:
            Parsed JSON object conforming to schema

        Raises:
            NotImplementedError: If provider doesn't support structured outputs
        """
        raise NotImplementedError(
            f"{self.name} provider does not support structured outputs"
        )

    @abstractmethod
    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        """
        Generate completions for multiple prompts concurrently.

        Args:
            prompts: List of user prompts
            system: Optional system message (same for all)
            max_completion_tokens: Maximum completion tokens to generate per completion

        Returns:
            List of LLMResponse objects
        """
        ...

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health status dictionary
        """
        ...

    @abstractmethod
    def get_usage_stats(self) -> dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Usage stats dictionary
        """
        ...

    def get_synthesis_concurrency(self) -> int:
        """
        Get recommended concurrency for parallel synthesis operations.

        Returns:
            Number of concurrent synthesis tasks this provider can handle.
            Used for map-reduce synthesis to execute cluster summaries in parallel.
            Default implementations return provider-specific values based on rate limits.
        """
        return 3  # Conservative default
