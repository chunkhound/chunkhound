"""Test-only fake LLM providers for deterministic benchmarking.

These providers are intentionally simple and deterministic:
- They never call external services.
- They can simulate latency based on requested max tokens.
- They record parameters passed to `complete()` so tests can assert behavior.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse


@dataclass
class CompleteCall:
    prompt: str
    system: str | None
    max_completion_tokens: int
    timeout: int | None


class SleepyDeterministicLLMProvider(LLMProvider):
    """Deterministic provider that simulates latency and records call params."""

    def __init__(
        self,
        *,
        model: str = "fake-synthesis",
        tokens_per_second: float = 1_000_000.0,
        fixed_response: str = (
            "## Overview\n"
            "This is a deterministic synthetic answer used for benchmarking and unit tests. "
            "It is intentionally long enough to satisfy synthesis minimum-length guards.\n\n"
            "[1]"
        ),
    ) -> None:
        self._model = model
        self._tokens_per_second = tokens_per_second
        self._fixed_response = fixed_response
        self.calls: list[CompleteCall] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        self.calls.append(
            CompleteCall(
                prompt=prompt,
                system=system,
                max_completion_tokens=max_completion_tokens,
                timeout=timeout,
            )
        )

        # Simulate cost/latency scaling with max tokens requested
        await asyncio.sleep(max_completion_tokens / self._tokens_per_second)

        return LLMResponse(
            content=self._fixed_response,
            tokens_used=max_completion_tokens,
            model=self._model,
            finish_reason="stop",
        )

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        results: list[LLMResponse] = []
        for p in prompts:
            results.append(
                await self.complete(
                    p, system=system, max_completion_tokens=max_completion_tokens
                )
            )
        return results

    def estimate_tokens(self, text: str) -> int:
        # Approximate; consistent with other CLI providers.
        return len(text) // 4

    async def health_check(self) -> dict[str, Any]:
        return {"status": "healthy", "provider": self.name, "model": self._model}

    def get_usage_stats(self) -> dict[str, Any]:
        return {"requests_made": len(self.calls), "tokens_used": 0}
