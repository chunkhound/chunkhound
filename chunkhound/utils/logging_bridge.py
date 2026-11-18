"""Single bridge from standard-library logging into loguru.

Two callers need it: the CLI (``pyo3-log`` forwards Rust log records into the
stdlib root logger) and native backends (watchdog's FSEvents emitter threads
report fatal stream failures through the ``fsevents`` logger). ChunkHound
surfaces logs via loguru only, so unrouted records stay invisible to
operators and test captures.
"""

from __future__ import annotations

import logging
from types import FrameType

from loguru import logger

# Rust pipeline records are tagged so the CLI sink can show them at INFO even
# without --verbose — see ``rich_output.default_sink_filter``.
_RUST_LOGGER_PREFIX = "chunkhound_native"


class _LoguruHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Preserve the record's real level: a Rust log::warn! must stay WARNING,
        # distinct from routine log::info! progress lines.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            # Fallback for stdlib levels loguru has no registered name for.
            level = record.levelno
        logger.bind(rust_native=record.name.startswith(_RUST_LOGGER_PREFIX)).opt(
            depth=_caller_depth(), exception=record.exc_info
        ).log(level, record.getMessage())


def _caller_depth() -> int:
    # Logging methods and handler plumbing add frames; attribute to the caller.
    frame: FrameType | None = logging.currentframe()
    depth = 0
    while frame is not None and (
        depth < 2 or frame.f_code.co_filename == logging.__file__
    ):
        frame = frame.f_back
        depth += 1
    return depth - 1


def bridge_root_logger(*, verbose: bool = False) -> None:
    """Route the root stdlib logger into loguru (replaces its handlers).

    Must run after ``basicConfig`` so the stdlib's default handler does not
    emit records a second time.
    """
    handler = _LoguruHandler(logging.DEBUG if verbose else logging.INFO)
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    # Lower, never raise, the root threshold so the handler sees INFO records.
    if root.level > handler.level:
        root.setLevel(handler.level)


def _effective_level(level: int | None) -> int:
    """Resolve the named-bridge threshold from the root logger's verbosity.

    ``bridge_root_logger`` is the one place verbosity is decided (the CLI
    passes ``--verbose`` there), so this bridge must not pin its own level: a
    hardcoded WARNING would hide watchdog INFO records from operators who run
    verbose. Non-verbose processes keep the WARNING default.
    """
    if level is not None:
        return level
    root = logging.getLogger()
    return logging.DEBUG if root.isEnabledFor(logging.DEBUG) else logging.WARNING


def bridge_std_logger(name: str, *, level: int | None = None) -> None:
    """Route one named stdlib logger into loguru (idempotent).

    The bridge owns the logger: propagation stays off, else a root bridge
    forwards every record a second time to the same loguru sinks.
    """
    std_logger = logging.getLogger(name)
    # Apply on every call: a logger bridged before this policy existed would
    # otherwise keep duplicating through the root bridge, and a re-bridge must
    # track a verbosity change made since the first call.
    std_logger.propagate = False
    std_logger.setLevel(_effective_level(level))
    if any(isinstance(handler, _LoguruHandler) for handler in std_logger.handlers):
        return
    std_logger.addHandler(_LoguruHandler())
