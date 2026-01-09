from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AgentDocMetadata:
    """Commit/LLM metadata embedded in Code Mapper output."""

    created_from_sha: str
    previous_target_sha: str
    target_sha: str
    generated_at: str
    llm_config: dict[str, str] = field(default_factory=dict)
    generation_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class HydeConfig:
    max_scope_files: int
    max_snippet_files: int
    max_snippet_chars: int
    max_completion_tokens: int
    max_snippet_tokens: int

    def __post_init__(self) -> None:
        values: dict[str, int] = {
            "max_scope_files": self.max_scope_files,
            "max_snippet_files": self.max_snippet_files,
            "max_snippet_chars": self.max_snippet_chars,
            "max_completion_tokens": self.max_completion_tokens,
            "max_snippet_tokens": self.max_snippet_tokens,
        }

        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"HydeConfig.{name} must be an int (got {type(value)})"
                )

        non_negative = ("max_scope_files", "max_snippet_files", "max_snippet_chars")
        for name in non_negative:
            if values[name] < 0:
                raise ValueError(f"HydeConfig.{name} must be >= 0 (got {values[name]})")

        positive = ("max_completion_tokens", "max_snippet_tokens")
        for name in positive:
            if values[name] <= 0:
                raise ValueError(f"HydeConfig.{name} must be > 0 (got {values[name]})")

    @classmethod
    def from_env(cls) -> HydeConfig:
        max_scope = 200
        max_snippet_files = 0
        max_snippet_chars = 0
        max_tokens = 30_000
        max_snippet_tokens = 50_000

        def _parse_positive_int(env_name: str, default: int) -> int:
            value = os.getenv(env_name)
            if not value:
                return default
            try:
                parsed = int(value)
            except ValueError:
                return default
            if parsed <= 0:
                return default
            return parsed

        max_scope = _parse_positive_int("CH_AGENT_DOC_HYDE_MAX_SCOPE_FILES", max_scope)
        max_snippet_files = _parse_positive_int(
            "CH_AGENT_DOC_HYDE_MAX_SNIPPET_FILES",
            max_snippet_files,
        )
        max_snippet_chars = _parse_positive_int(
            "CH_AGENT_DOC_HYDE_MAX_SNIPPET_CHARS",
            max_snippet_chars,
        )
        max_snippet_tokens = _parse_positive_int(
            "CH_AGENT_DOC_HYDE_SNIPPET_TOKENS",
            max_snippet_tokens,
        )

        tokens_env = os.getenv("CH_AGENT_DOC_HYDE_COMPLETION_TOKENS")
        if tokens_env:
            try:
                parsed_tokens = int(tokens_env)
                if parsed_tokens > 0:
                    max_tokens = min(parsed_tokens, 30_000)
            except ValueError:
                pass

        return cls(
            max_scope_files=max_scope,
            max_snippet_files=max_snippet_files,
            max_snippet_chars=max_snippet_chars,
            max_completion_tokens=max_tokens,
            max_snippet_tokens=max_snippet_tokens,
        )


CodeMapperPOIMode = Literal["architectural", "operational"]


@dataclass(frozen=True)
class CodeMapperPOI:
    mode: CodeMapperPOIMode
    text: str
