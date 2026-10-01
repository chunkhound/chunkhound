"""Per-user usage analytics.

The recorder itself (buffer file, background flush thread, S3 upload,
handle-based command bookkeeping) lives in the Rust extension
(`chunkhound_native.AnalyticsRecorder`) -- see src/AGENTS.md. This package
is the thin Python side: building the recorder from validated config, and a
`contextvars`-based convenience for call sites that don't have an explicit
handle to hand (LLM/embedding provider calls, several stack frames below the
MCP/CLI hook that opened the command).
"""

from .recorder import (
    bind_current,
    build_recorder,
    end_command,
    get_current,
    record_internal_error,
    record_provider_call,
    redact_action_fields,
    shutdown,
    start_command,
    update_action,
)

__all__ = [
    "build_recorder",
    "start_command",
    "update_action",
    "end_command",
    "record_provider_call",
    "record_internal_error",
    "get_current",
    "bind_current",
    "shutdown",
    "redact_action_fields",
]
