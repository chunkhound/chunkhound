from typing import Any
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse

class AntigravityLLMProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "",
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return "antigravity-sdk"

    @property
    def model(self) -> str:
        return self._model

    @property
    def timeout(self) -> int:
        return self._timeout

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        pass

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        pass

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        pass

    def estimate_tokens(self, text: str) -> int:
        return 0

    async def health_check(self) -> dict[str, Any]:
        return {"status": "ok"}

    def get_usage_stats(self) -> dict[str, Any]:
        return {}

    def get_synthesis_concurrency(self) -> int:
        return 5
