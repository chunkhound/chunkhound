"""Shared Claude model default resolution."""

from __future__ import annotations

import functools
import os
from datetime import datetime
from typing import Any

from loguru import logger

CLAUDE_HAIKU_DEFAULT_SENTINEL = "claude-haiku"
CLAUDE_HAIKU_FALLBACK_MODEL = "claude-haiku-4-5-20251001"


def resolve_claude_haiku_model(
    requested: str | None,
    api_key: str | None = None,
    *,
    discover: bool = True,
) -> str:
    """Resolve the Claude Haiku default model.

    The sentinel value ``claude-haiku`` is ChunkHound's shared Anthropic
    default for both utility and synthesis. This is intentional: current Claude
    Haiku is capable enough for synthesis and is Anthropic's cheapest available
    Claude model. Explicit model names pass through unchanged.

    Discovery is opt-in for call sites that can tolerate startup network I/O;
    otherwise the sentinel resolves to ChunkHound's pinned fallback.
    """
    model_name = (requested or "").strip()
    if model_name and model_name.lower() != CLAUDE_HAIKU_DEFAULT_SENTINEL:
        return model_name

    env_override = os.getenv("CHUNKHOUND_CLAUDE_DEFAULT_HAIKU_MODEL")
    if env_override:
        return env_override.strip()

    if discover:
        discovered = get_latest_available_haiku_model(api_key)
        return discovered or CLAUDE_HAIKU_FALLBACK_MODEL

    return CLAUDE_HAIKU_FALLBACK_MODEL


@functools.lru_cache(maxsize=8)
def get_latest_available_haiku_model(api_key: str | None = None) -> str | None:
    """Discover the newest available Claude Haiku model from Anthropic.

    Returns None when discovery is unavailable, unauthenticated, or fails.
    """
    resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_key or not resolved_key.startswith("sk-ant-"):
        return None

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=resolved_key)
        models = client.models.list(limit=100, timeout=10)
        haiku_models = [
            model
            for model in models
            if _is_claude_haiku_model(getattr(model, "id", ""))
        ]
        if not haiku_models:
            return None
        latest = max(haiku_models, key=_model_sort_key)
        return str(latest.id)
    except Exception as e:  # pragma: no cover - depends on network/credentials
        logger.debug(f"Claude Haiku model discovery failed: {e}")
        return None


def _is_claude_haiku_model(model_id: str) -> bool:
    model = model_id.lower()
    return model.startswith("claude-") and "haiku" in model


def _model_sort_key(model: Any) -> tuple[datetime, str]:
    created_at = getattr(model, "created_at", None)
    if isinstance(created_at, datetime):
        return created_at, str(getattr(model, "id", ""))
    return datetime.min, str(getattr(model, "id", ""))
