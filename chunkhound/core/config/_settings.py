"""Context-local source policy for ChunkHound nested ``BaseSettings``.

ChunkHound's nested configs — ``EmbeddingConfig``, ``LLMConfig``,
``ResearchConfig``, ``FetchUrlConfig`` — are ``pydantic_settings.BaseSettings``
and therefore read ``CHUNKHOUND_*`` env vars through their OWN env source
inside their constructors. That re-read defeats an outer
``skip_layers={"env"}`` gate: Config already stopped merging env into the
layer dict, yet the nested model re-populated missing fields from env in
its constructor, leaking standing env vars past the env-layer boundary —
e.g. a persisted-gate snapshot (``snapshot_for_persisted_gate``) would
commit a file carrying a hazard an env var masks on this machine.

The previous fix popped nested env vars from the process-global
``os.environ`` around the snapshot (now-removed
``_hide_nested_basesettings_env``). That was not thread-safe: every
process reader of ``os.environ`` — including a standalone ``Config`` or a
``BaseSettings`` built in another thread — observed the vars as absent for
the snapshot's duration.

This module replaces the pop-and-restore hack with a context-local
``ContextVar`` policy. Nested models consult it in
``settings_customise_sources`` and omit the env/dotenv/secrets sources when
suppression is active. ``os.environ`` is never mutated, so concurrent or
standalone readers always see the real environment. A new
``threading.Thread`` gets a fresh context (no inherited suppression), which
is the property the concurrency regression contract relies on.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
)

# Source categories suppressed together when the env layer is skipped.
# These are exactly the env-derived sources pydantic-settings would
# otherwise consult independently of Config's layer merge.
_ENV_SOURCE_CATEGORIES: frozenset[str] = frozenset({"env", "dotenv", "secrets"})

# Active suppression set for the current context. Empty == no suppression
# (== normal standalone settings behavior: init > env > dotenv > secrets).
_active_policy: ContextVar[frozenset[str]] = ContextVar(
    "_active_policy", default=frozenset()
)


def env_sources_disabled() -> bool:
    """True when the active context suppresses env/dotenv/secrets sources."""
    return bool(_active_policy.get() & _ENV_SOURCE_CATEGORIES)


@contextmanager
def suppressed_env_sources() -> Iterator[None]:
    """Temporarily suppress env/dotenv/secrets sources within this context.

    Context-local: a new ``threading.Thread`` gets a fresh context; asyncio
    tasks created inside the scope inherit suppression. ``os.environ`` is
    never touched, so a concurrent reader never observes a
    half-restored environment.
    """
    token = _active_policy.set(_ENV_SOURCE_CATEGORIES)
    try:
        yield
    finally:
        _active_policy.reset(token)


def apply_source_policy(
    init_settings: PydanticBaseSettingsSource,
    *env_sources: PydanticBaseSettingsSource,
) -> tuple[PydanticBaseSettingsSource, ...]:
    """Return the source tuple honoring the active suppression policy.

    When the env layer is suppressed, only ``init_settings`` survives —
    the nested model is then built from resolved layer data + defaults, with
    no env re-read. Otherwise the supplied env sources are kept in order,
    preserving the normal (init > env > dotenv > secrets) contract.
    """
    if env_sources_disabled():
        return (init_settings,)
    return (init_settings, *env_sources)


class _ChunkhoundBaseSettings(BaseSettings):
    """Base for ChunkHound nested ``BaseSettings`` configs.

    Honors the context-local source policy: ``settings_customise_sources``
    omits env/dotenv/secrets when suppression is active and keeps the
    default source order otherwise — so normal standalone construction is
    unchanged.

    Config scopes all nested validation and default factories under this
    policy when it skips env.
    """

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return apply_source_policy(
            init_settings, env_settings, dotenv_settings, file_secret_settings
        )
