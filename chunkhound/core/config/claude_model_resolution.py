"""Shared Claude model sentinel helpers.

Anthropic API requests resolve sentinels to concrete model IDs via:
- ``CHUNKHOUND_CLAUDE_DEFAULT_<TIER>_MODEL`` env overrides
- optional dynamic discovery via ``anthropic.models.list()``
- pinned fallbacks when discovery is disabled or unavailable

Claude Code CLI requests reuse the same env-override handling but otherwise
preserve sentinels so the CLI can map them to bare aliases and pick the
freshest available model itself.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypedDict

from loguru import logger


class ClaudeSentinelConfig(TypedDict):
    """Config for a Claude model sentinel tier."""

    fallback: str
    env_var: str
    filter: Callable[[str], bool]


# ── Sentinels ────────────────────────────────────────────────────────────

CLAUDE_HAIKU_SENTINEL = "claude-haiku"
CLAUDE_SONNET_SENTINEL = "claude-sonnet"
CLAUDE_OPUS_SENTINEL = "claude-opus"

# ── Pinned fallbacks (used when discovery is disabled or fails) ──────────

CLAUDE_HAIKU_FALLBACK = "claude-haiku-4-5-20251001"
CLAUDE_SONNET_FALLBACK = "claude-sonnet-4-6"
CLAUDE_OPUS_FALLBACK = "claude-opus-4-8"

# ── Sentinel config lookup table ─────────────────────────────────────────

_CLAUDE_SENTINELS: dict[str, ClaudeSentinelConfig] = {
    CLAUDE_HAIKU_SENTINEL: {
        "fallback": CLAUDE_HAIKU_FALLBACK,
        "env_var": "CHUNKHOUND_CLAUDE_DEFAULT_HAIKU_MODEL",
        "filter": lambda id: "haiku" in id.lower(),
    },
    CLAUDE_SONNET_SENTINEL: {
        "fallback": CLAUDE_SONNET_FALLBACK,
        "env_var": "CHUNKHOUND_CLAUDE_DEFAULT_SONNET_MODEL",
        "filter": lambda id: "sonnet" in id.lower(),
    },
    CLAUDE_OPUS_SENTINEL: {
        "fallback": CLAUDE_OPUS_FALLBACK,
        "env_var": "CHUNKHOUND_CLAUDE_DEFAULT_OPUS_MODEL",
        "filter": lambda id: "opus" in id.lower(),
    },
}

# ── Discovery cache ──────────────────────────────────────────────────────
#
# Cache by (sentinel, auth identity) so one account's visible model set
# cannot bleed into another account within the same process.
#
# The auth identity stores only a short digest of the resolved API key, never
# the raw credential itself.
#
# Failed discoveries are intentionally NOT cached so that transient failures
# (e.g. credentials appearing mid-process) self-heal on the next call.
_MODEL_CACHE: dict[tuple[str, str], str] = {}


def clear_claude_cache(sentinel: str | None = None) -> None:
    """Clear cached discovery results (for testing).

    Args:
        sentinel: If given, clear only that sentinel's cached entry.
            If ``None``, clear the entire cache.
    """
    if sentinel is None:
        _MODEL_CACHE.clear()
    else:
        for cache_key in [key for key in _MODEL_CACHE if key[0] == sentinel]:
            _MODEL_CACHE.pop(cache_key, None)


# ── Public resolution entry point ────────────────────────────────────────

def resolve_claude_model(
    requested: str | None,
    api_key: str | None = None,
    *,
    discover: bool = True,
) -> str:
    """Resolve a Claude model sentinel to a concrete model ID.

    Recognised sentinels: ``claude-haiku``, ``claude-sonnet``, ``claude-opus``.

    Explicit model names (e.g. ``"claude-sonnet-4-6"``) pass through unchanged.

    Resolution order (first-match wins):
    1. Explicit (non-sentinel) model name — returned as-is
    2. ``CHUNKHOUND_CLAUDE_DEFAULT_<TIER>_MODEL`` environment variable
    3. ``discover=True`` → ``anthropic.models.list()`` → newest available model
    4. Pinned fallback constant

    Args:
        requested: Model name or sentinel string (or ``None``).
        api_key: Anthropic API key for the discovery call
            (defaults to ``ANTHROPIC_API_KEY`` env var).
        discover: When ``True``, dynamically discover the newest available
            model via the Anthropic API (cached per sentinel + auth identity).
            When ``False``, return the pinned fallback directly.

    Returns:
        Concrete model ID string.
    """
    model_name = (requested or "").strip()
    if not model_name:
        return CLAUDE_HAIKU_FALLBACK

    config = _CLAUDE_SENTINELS.get(model_name.lower())
    if config is None:
        # Not a sentinel — explicit model name passes through
        return model_name

    env_override = _get_env_override(config)
    if env_override is not None:
        return env_override

    if discover:
        discovered = _discover_latest_model(model_name.lower(), api_key, config)
        if discovered is not None:
            return discovered

    return config["fallback"]


def resolve_claude_cli_model(requested: str | None) -> str:
    """Resolve a Claude CLI model request without forcing pinned fallbacks.

    Resolution order:
    1. Empty request → default Haiku sentinel
    2. Explicit (non-sentinel) model name — returned as-is
    3. ``CHUNKHOUND_CLAUDE_DEFAULT_<TIER>_MODEL`` environment variable
    4. Original sentinel — preserved so the CLI can choose the latest alias
    """
    model_name = (requested or "").strip()
    if not model_name:
        return CLAUDE_HAIKU_SENTINEL

    sentinel = model_name.lower()
    config = _CLAUDE_SENTINELS.get(sentinel)
    if config is None:
        return model_name

    env_override = _get_env_override(config)
    if env_override is not None:
        return env_override

    return sentinel


# ── API discovery ────────────────────────────────────────────────────────

def _get_env_override(config: ClaudeSentinelConfig) -> str | None:
    """Return a non-empty CHUNKHOUND Claude model override, if configured."""
    env_override = os.getenv(config["env_var"])
    if env_override is None:
        return None

    value = env_override.strip()
    if not value:
        return None

    return value


def _discover_latest_model(
    sentinel: str,
    api_key: str | None,
    config: ClaudeSentinelConfig,
) -> str | None:
    """Discover the newest available model for *sentinel* via the Anthropic API.

    The result is cached in ``_MODEL_CACHE`` after the first successful
    discovery for that sentinel + auth identity (process lifetime).
    Returns ``None`` when the API is unavailable, credentials are missing,
    or no matching models are found.
    """
    resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_key or not resolved_key.startswith("sk-ant-"):
        return None

    cache_key = (sentinel, _cache_identity_for_api_key(resolved_key))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=resolved_key)
        models = client.models.list(limit=100, timeout=10)
        matches = [
            model
            for model in models
            if config["filter"](getattr(model, "id", ""))
        ]
        if not matches:
            return None
        latest = max(matches, key=_model_sort_key)
        _MODEL_CACHE[cache_key] = str(latest.id)
        return _MODEL_CACHE[cache_key]
    except Exception as e:  # pragma: no cover - depends on network/credentials
        logger.debug(f"Claude model discovery failed ({sentinel}): {e}")
        return None


def _cache_identity_for_api_key(api_key: str) -> str:
    """Return a stable non-secret cache identity for an Anthropic API key."""
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return digest[:16]


# ── Sorting helper ───────────────────────────────────────────────────────

def _model_sort_key(model: Any) -> tuple[datetime, str]:
    """Sort key: newest ``created_at`` first, model ID as tiebreaker."""
    created_at = getattr(model, "created_at", None)
    if isinstance(created_at, datetime):
        return created_at, str(getattr(model, "id", ""))
    return datetime.min, str(getattr(model, "id", ""))


# ═════════════════════════════════════════════════════════════════════════
# Backward-compatibility aliases
# ═════════════════════════════════════════════════════════════════════════

CLAUDE_HAIKU_DEFAULT_SENTINEL = CLAUDE_HAIKU_SENTINEL
CLAUDE_HAIKU_FALLBACK_MODEL = CLAUDE_HAIKU_FALLBACK
resolve_claude_haiku_model = resolve_claude_model


def get_latest_available_haiku_model(api_key: str | None = None) -> str | None:
    """Backward-compat: discover newest Haiku model (legacy entry point).

    Delegates to the generalized ``_discover_latest_model``.
    """
    config = _CLAUDE_SENTINELS[CLAUDE_HAIKU_SENTINEL]
    return _discover_latest_model(CLAUDE_HAIKU_SENTINEL, api_key, config)


def clear_haiku_cache() -> None:
    """Backward-compat: clear only the Haiku cache entry."""
    clear_claude_cache(CLAUDE_HAIKU_SENTINEL)
