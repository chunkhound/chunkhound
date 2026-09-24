"""Construction/lifecycle glue for the Rust-native `AnalyticsRecorder`.

This module deliberately owns no bookkeeping of its own -- that all lives in
`chunkhound_native.AnalyticsRecorder` (handle-based, see src/analytics/ and
src/AGENTS.md). What lives here:

- `build_recorder()`: turns a validated `AnalyticsConfig` + the resolved
  target directory into a constructed native recorder. Always succeeds --
  falls back to a disabled recorder on any construction failure, since
  analytics must never be the reason a host command fails to start.
- A `contextvars`-based "current (recorder, handle, save_sensitive_data)"
  convenience, used by call sites that don't have an explicit handle to
  hand (LLM/embedding provider calls, several stack frames below the
  MCP/CLI hook that opened the command). This is safe for ordinary
  single-threaded Python call chains -- each asyncio Task gets its own copy
  of the context -- and is NOT used across the Rust rayon thread-pool
  boundary during indexing; that boundary threads the recorder through
  explicitly instead (see chunkhound/pipeline_bridge.py and the Phase 4
  native-path wiring).
- `redact_action_fields()`: the single enforcement point for
  `save_sensitive_data`, applied inside both `start_command()` and
  `update_action()` so no caller can forget it -- see those functions.
"""

import contextvars
import getpass
import json
from pathlib import Path
from typing import Any

from loguru import logger

import chunkhound_native
from chunkhound import __version__
from chunkhound.core.config.analytics_config import AnalyticsConfig

_ANALYTICS_DIR = Path.home() / ".config" / "chunkhound" / "analytics"

# Free-text, user-authored action fields -- as opposed to structural
# metadata like a commit hash/range, which carries no proprietary content.
# This is the single source of truth for what "sensitive" means across both
# the MCP and CLI dispatch chokepoints (see redact_action_fields()).
SENSITIVE_ACTION_FIELDS = frozenset({"query", "question", "url"})

# (recorder, handle, save_sensitive_data) for the command currently open in
# this asyncio Task/thread of control, or None. Never shared across asyncio
# Tasks or OS threads -- see module docstring. save_sensitive_data is
# captured here (set once by start_command) so update_action() can enforce
# redaction on its own, without every caller having to remember to do it --
# see redact_action_fields().
_current: "contextvars.ContextVar[tuple[Any, int, bool] | None]" = (
    contextvars.ContextVar("chunkhound_analytics_current", default=None)
)


def get_current() -> tuple[Any, int, bool] | None:
    """Read the (recorder, handle, save_sensitive_data) currently open in
    this Task, or None.

    For callers that need to explicitly carry the binding across a
    boundary the ContextVar can't cross on its own -- e.g. resolving it on
    the CLI's own thread/task before handing off to a Rust rayon thread
    pool via `functools.partial` (see pipeline_bridge.py). Ordinary call
    sites should use `record_provider_call`/`record_internal_error`
    instead of reading this directly.
    """
    return _current.get()


def bind_current(
    recorder: Any | None, handle: int, save_sensitive_data: bool = False
) -> None:
    """Explicitly set (recorder, handle, save_sensitive_data) as "current"
    on this OS thread.

    For callers that can't rely on ContextVar auto-propagation -- a Rust
    rayon worker thread invoking a Python embed callback gets a fresh,
    empty context (contextvars don't cross OS thread boundaries the way
    they cross asyncio Task boundaries). Call this once, on that thread,
    immediately before the instrumented provider call it's meant to cover.
    `recorder=None` clears any stale binding rather than setting a
    (None, handle, ...) pair that would itself need a None-check everywhere.
    """
    _current.set(
        (recorder, handle, save_sensitive_data) if recorder is not None else None
    )


def build_recorder(config: AnalyticsConfig | None, target_dir: Path) -> Any:
    """Construct the process's AnalyticsRecorder from validated config.

    No try/except here: `chunkhound_native.AnalyticsRecorder`'s constructor
    is deliberately built to never raise -- every actually-fallible
    operation it performs (the git shell-out in `resolve_repository_name`,
    `S3Target::new`'s config validation, `fs::create_dir_all`, building the
    `reqwest` client, salt-file I/O) is caught and gracefully degraded to a
    disabled/no-op state *inside* the Rust constructor itself (see
    `src/analytics/recorder.rs`'s `build_inner_from_raw`), so a bad
    analytics environment can never prevent the host command from starting.
    A Python-side try/except here would just be dead code shadowing that
    guarantee. `config=None` (e.g. a caller/test double whose Config-like
    object has no `analytics` attribute at all) is treated the same as a
    disabled config.
    """
    if config is None:
        return chunkhound_native.AnalyticsRecorder({"enabled": False})
    config_dict = {
        "enabled": config.enabled,
        "privacy_mode": config.anonymize,
        "s3_endpoint_url": config.s3_endpoint_url,
        "s3_bucket": config.s3_bucket,
        # SecretStr -- sourced from .chunkhound.json or the
        # CHUNKHOUND_AWS_ACCESS_KEY_ID/CHUNKHOUND_AWS_SECRET_ACCESS_KEY env
        # vars via AnalyticsConfig.load_from_env(); see analytics_config.py's
        # module docstring.
        "s3_access_key": (
            config.s3_access_key.get_secret_value() if config.s3_access_key else None
        ),
        "s3_secret_key": (
            config.s3_secret_key.get_secret_value() if config.s3_secret_key else None
        ),
        "flush_interval_seconds": config.flush_interval_seconds,
        "flush_batch_size": config.flush_batch_size,
        "max_upload_retries": config.max_upload_retries,
        "buffer_dir": str(_ANALYTICS_DIR),
        "salt_path": str(_ANALYTICS_DIR / "salt"),
        "repository_dir": str(target_dir),
        "os_username": _get_os_username(),
        "chunkhound_version": __version__,
    }
    return chunkhound_native.AnalyticsRecorder(config_dict)


def redact_action_fields(
    fields: dict[str, Any], save_sensitive_data: bool
) -> dict[str, Any]:
    """Null out sensitive action field values when `save_sensitive_data` is
    false, keeping the field present (so downstream consumers still see it
    existed) rather than dropping the key. A pure pass-through when
    `save_sensitive_data` is true.

    Called internally by `start_command()`/`update_action()` -- this is the
    enforcement point, not something callers need to invoke themselves.
    Exposed as a module-level function (rather than nested/private) so it
    stays independently unit-testable."""
    if save_sensitive_data:
        return fields
    return {
        k: (None if k in SENSITIVE_ACTION_FIELDS else v) for k, v in fields.items()
    }


def start_command(
    recorder: Any | None,
    command: str,
    source: str,
    action: dict[str, Any],
    save_sensitive_data: bool = False,
) -> int:
    """Open a command, set it as "current" for this Task, and return its handle.

    `recorder=None` (no recorder was wired for this call site) is a silent
    no-op returning handle 0 -- callers never need to guard this call with
    an `if recorder is not None`.

    `action` is redacted here via `redact_action_fields()` before it's ever
    serialized -- callers should pass the raw, unredacted action dict.
    `save_sensitive_data` is also captured as "current" alongside the
    recorder/handle so a later `update_action()` call (which may run deep
    inside a command's own implementation, with no access to config) can
    enforce the same redaction policy on its own, rather than trusting every
    future caller to redact before calling in.

    Callers (MCP/CLI hooks) should still hold the returned handle explicitly
    and pass it to `end_command` -- the ContextVar is only a convenience for
    deeper call sites, not a replacement for that.
    """
    if recorder is None:
        return 0
    redacted = redact_action_fields(action, save_sensitive_data)
    try:
        handle = int(
            recorder.start_command(command, source, json.dumps(redacted, default=str))
        )
    except Exception:
        logger.opt(exception=True).debug(
            "analytics: start_command failed, dropping event"
        )
        return 0
    _current.set((recorder, handle, save_sensitive_data))
    return handle


def update_action(action: dict[str, Any]) -> None:
    """Merge new fields into whatever command is currently open in this
    Task's action, for fields only known after the command runs (e.g.
    `index`'s `file_count`/`total_chunks`, unavailable at `start_command`
    time). Reads the "current" handle the same way `record_provider_call`
    does -- for calling from deep inside a command's own implementation,
    not from the CLI/MCP dispatch chokepoint itself. A silent no-op if no
    command is open.

    Applies `redact_action_fields()` using the `save_sensitive_data` value
    captured by `start_command()` for this command -- callers here, same as
    `start_command`'s callers, should pass the raw, unredacted dict; this
    is the enforcement point, not the caller's responsibility."""
    current = _current.get()
    if current is None:
        return
    recorder, handle, save_sensitive_data = current
    redacted = redact_action_fields(action, save_sensitive_data)
    try:
        recorder.update_action(handle, json.dumps(redacted, default=str))
    except Exception:
        logger.opt(exception=True).debug("analytics: update_action failed")


def end_command(recorder: Any | None, handle: int, success: bool) -> None:
    """Finalize a command and clear it as "current" for this Task.
    `recorder=None` is a silent no-op, matching `start_command`."""
    if recorder is None:
        return
    try:
        recorder.end_command(handle, success)
    except Exception:
        logger.opt(exception=True).debug("analytics: end_command failed")
    finally:
        _current.set(None)


def record_provider_call(
    kind: str,
    provider: str,
    model: str,
    success: bool,
    error_type: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Record one provider-call attempt against whatever command is
    currently open in this Task. A silent no-op if none is open -- matches
    the design's "shouldn't normally happen" fallback for provider calls
    made outside any tracked command."""
    current = _current.get()
    if current is None:
        return
    recorder, handle, _save_sensitive_data = current
    try:
        recorder.record_provider_call(
            handle,
            kind,
            provider,
            model,
            success,
            error_type,
            input_tokens,
            output_tokens,
        )
    except Exception:
        logger.opt(exception=True).debug("analytics: record_provider_call failed")


def record_internal_error(error_type: str) -> None:
    """Record a command-level failure not caused by any vendor call, against
    whatever command is currently open in this Task."""
    current = _current.get()
    if current is None:
        return
    recorder, handle, _save_sensitive_data = current
    try:
        recorder.record_internal_error(handle, error_type)
    except Exception:
        logger.opt(exception=True).debug("analytics: record_internal_error failed")


def shutdown(recorder: Any | None, timeout_ms: int = 3000) -> None:
    """Best-effort final flush before process exit. Bounded by `timeout_ms`
    so a slow/unreachable S3 endpoint can never hang shutdown -- call from
    the CLI's `finally` and the MCP server's shutdown path. A hard kill
    (SIGKILL, crashed process) skips this entirely; the orphan sweep on a
    subsequent run picks up the leftover buffer instead. `recorder=None` is
    a silent no-op, matching every other call site in this module."""
    if recorder is None:
        return
    try:
        recorder.shutdown(timeout_ms)
    except Exception:
        logger.opt(exception=True).debug("analytics: shutdown failed")


def _get_os_username() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"
