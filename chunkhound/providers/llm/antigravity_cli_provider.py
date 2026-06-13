from typing import Any
from chunkhound.providers.llm.base_cli_provider import BaseCLIProvider

class AntigravityCLIProvider(BaseCLIProvider):
    def _get_provider_name(self) -> str:
        return "antigravity-cli"

    async def _run_cli_command(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        return ""

    def get_synthesis_concurrency(self) -> int:
        return 1
