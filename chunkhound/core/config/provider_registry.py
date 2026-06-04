"""Data-driven registry for OpenAI-compatible LLM providers.

Each entry describes what's stable about a provider's API protocol.
Model names are NEVER baked in — they come from user configuration.

Lives in ``core/config/`` (not ``providers/llm/``) as it defines
config-domain data (API spec entries) — both ``llm_config.py``
and ``llm_manager.py`` import here.

To add a new OpenAI-compatible provider you must also touch:
  - ``LLMProviderLiteral`` in ``llm_config.py``
  - ``CLI_PROVIDER_CHOICES`` in ``llm_config.py``
  - test ``SPECS`` in ``test_openai_compatible_provider.py``
  - ``REASONING_EFFORT_PROVIDERS`` in ``llm_config.py`` (if applicable)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAICompatibleSpec:
    """Stable API properties of an OpenAI-compatible provider.

    Attributes:
        name: Provider identifier string (matches config ``provider`` value)
        default_base_url: API endpoint base URL
        supports_structured_outputs: Whether native ``json_schema``
            response_format is supported
        supports_reasoning_effort: Whether ``reasoning_effort`` API parameter
            is accepted
        max_tokens_param_name: API parameter name for output token limit
            (``"max_completion_tokens"`` for newer APIs, ``"max_tokens"`` for older)
        synthesis_concurrency: Recommended parallel synthesis operations count
        docs_url: External API documentation URL
        auth_url: Authentication portal URL
    """
    name: str
    default_base_url: str
    supports_structured_outputs: bool = True
    supports_reasoning_effort: bool = False
    max_tokens_param_name: str = "max_completion_tokens"
    synthesis_concurrency: int = 3
    docs_url: str = ""
    auth_url: str = ""


# ── Provider specs ─────────────────────────────────────────────────────────
# Append one entry here, then update LLMProviderLiteral, CLI_PROVIDER_CHOICES,
# and test SPECS — see module docstring above.
OPENAI_COMPATIBLE_PROVIDERS: dict[str, OpenAICompatibleSpec] = {
    "deepseek": OpenAICompatibleSpec(
        name="deepseek",
        default_base_url="https://api.deepseek.com",
        supports_structured_outputs=False,
        max_tokens_param_name="max_tokens",
        synthesis_concurrency=10,
        docs_url="https://platform.deepseek.com/api-docs",
        auth_url="https://platform.deepseek.com",
    ),
    "grok": OpenAICompatibleSpec(
        name="grok",
        default_base_url="https://api.x.ai/v1",
        supports_reasoning_effort=True,
        synthesis_concurrency=5,
        docs_url="https://docs.x.ai/docs/models",
        auth_url="https://console.x.ai",
    ),
}
