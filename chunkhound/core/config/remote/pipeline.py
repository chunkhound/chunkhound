"""Remote-config pipeline orchestrator.

Step order:
1. URL discovery — restricted merge (skip local & --config layers) so
   project-local files can never redirect the operator's URL.
2. Fetch envelope.
3. Envelope validation (parse, version==1, min_chunkhound_version) so a
   payload targeted at a newer format never lands on an older client.
4. Rule loop against a working-copy dict:
   4a apply matching rules
   4b scrub refused fields (operator-owned paths that remote must never
      influence — DB location, target dir, embeddings kill-switch)
   4c self-register remote_config.url / .auth_header so subsequent runs
      converge on the same discovery inputs without CLI/env repetition
   4d terminal delta-only gate: the post-rules config must not introduce
      any new command-validation errors that weren't already present.
      Checked across the current command plus every persistence-hazard
      command, so a `search` invocation can't silently make `mcp` unsafe.
   4e write iff dict changed (avoid touching mtime on a no-op run).
5. Re-load happens naturally when ``create_validated_config`` constructs
   the final ``Config(args=args)`` after this function returns.
"""

import copy
from typing import Any

from pydantic import ValidationError

import chunkhound
from chunkhound.core.config.config import (
    PERSISTENCE_HAZARD_COMMANDS,
    Config,
    ConfigErrorCode,
)
from chunkhound.utils.logging_guard import log_if_not_mcp

from . import fetcher, persistence, rules

# Operator-owned fields that remote config must never touch. DB path and
# target dir describe the local install; embeddings_disabled is a kill-switch
# the operator sets deliberately; the *_config_file paths are discovery
# inputs to the loader and would recurse if remote could override them.
# The *_config_file paths are `exclude=True` today; listed defensively
# against a future serialization change.
# `remote_config.*` is intentionally mutable — server-driven URL/header
# rotation is a supported migration path via envelope rules on those keys.
# Discovery-layer self-registration seeds the on-disk value on the first
# successful fetch and never overwrites it thereafter; see
# _self_register_remote.
REFUSED_PATHS: tuple[str, ...] = (
    "database.path",
    "target_dir",
    "embeddings_disabled",
    "local_config_file",
    "global_config_file",
    "config_file",
)


async def run(args: Any, command: str) -> None:
    """Execute the remote-config pipeline. Idempotent, silent on failure."""
    try:
        await _run(args, command)
    except Exception as exc:
        # SystemExit from persistence.backup_and_write propagates past this
        # handler unaffected (it inherits from BaseException, not Exception)
        # — the operator needs a loud signal that persistence broke.
        #
        # Everything else is defense-in-depth: this pipeline runs on every
        # invocation. An unexpected failure in rule application, snapshot
        # construction, or persistence must not take down `chunkhound search`
        # — log at ERROR with a traceback and let the loader see whatever's
        # already on disk. Every *known* failure mode has a narrow `except`
        # inside `_run`, so reaching here means a bug worth the stack.
        log_if_not_mcp(
            "ERROR",
            "Remote-config: unexpected {} — pipeline aborted ({})",
            type(exc).__name__,
            exc,
            exception=True,
        )


async def _run(args: Any, command: str) -> None:
    # Step 1 — restricted merge for URL discovery
    try:
        discovery = Config(
            args=args,
            skip_layers={"local_config", "config_file"},
        )
    except (ValueError, ValidationError) as exc:
        log_if_not_mcp(
            "WARNING",
            "Remote-config: URL-discovery config failed to build ({}); skipping.",
            exc,
        )
        return

    remote = discovery.remote_config
    if remote is None or not remote.url:
        return  # feature disabled for this invocation

    # Step 2 — fetch
    envelope = await fetcher.fetch(remote.url, remote.auth_header)
    if envelope is None:
        return

    # Step 3 — envelope validation
    if not isinstance(envelope, dict):
        log_if_not_mcp(
            "WARNING",
            "Remote-config envelope_parse_error: expected JSON object, "
            "got {}",
            type(envelope).__name__,
        )
        return

    version = envelope.get("version")
    if version != 1:
        log_if_not_mcp(
            "WARNING",
            "Remote-config envelope_version_unsupported: {!r}",
            version,
        )
        return

    min_version = envelope.get("min_chunkhound_version")
    current_version = chunkhound.__version__
    if min_version is not None:
        from packaging.version import InvalidVersion
        from packaging.version import parse as parse_version

        try:
            if parse_version(current_version) < parse_version(str(min_version)):
                log_if_not_mcp(
                    "WARNING",
                    "Remote-config envelope_version_gate: envelope requires "
                    ">= {}, running {}",
                    min_version,
                    current_version,
                )
                return
        except InvalidVersion:
            log_if_not_mcp(
                "WARNING",
                "Remote-config envelope_version_gate: unparseable version "
                "{!r} — discarding payload",
                min_version,
            )
            return

    # Step 4 — rule loop
    target = persistence.resolve_target()
    try:
        on_disk_dict = persistence.read_target(target)
    except (OSError, ValueError) as exc:
        # A read failure here means we can't compute a proper delta; fail safe.
        log_if_not_mcp(
            "WARNING",
            "Remote-config: could not read {} ({}); skipping.",
            target,
            exc,
        )
        return

    working_copy = copy.deepcopy(on_disk_dict)

    # Half-merged snapshot for predicates. Built once from the on-disk
    # global JSON so `when.existing` reflects reality, not mid-pipeline
    # mutations.
    try:
        half_merged = Config._snapshot_from_global_dict(on_disk_dict)
    except (ValueError, ValidationError) as exc:
        log_if_not_mcp(
            "WARNING",
            "Remote-config: half-merged snapshot failed ({}); skipping.",
            exc,
        )
        return

    # 4a — apply rules
    rule_list = envelope.get("rules") or []
    if not isinstance(rule_list, list):
        log_if_not_mcp(
            "WARNING",
            "Remote-config envelope schema_error: 'rules' must be a list",
        )
        return

    for rule_index, rule in enumerate(rule_list, start=1):
        if not isinstance(rule, dict):
            log_if_not_mcp(
                "WARNING",
                "Remote-config rule {} schema_error: entry is not an object",
                rule_index,
            )
            continue
        rules.apply_rule(
            rule,
            working_copy,
            half_merged,
            current_version,
            rule_index=rule_index,
        )

    # 4b — scrub refused paths (restore from on_disk_dict when present)
    for path in REFUSED_PATHS:
        _restore_refused(working_copy, on_disk_dict, path)

    # 4c — self-register remote_config discovery inputs
    _self_register_remote(working_copy, on_disk_dict, remote)

    # 4d — terminal delta-only gate. The pre-rules snapshot is the same
    # Config as `half_merged` (identical layer selector, same on_disk_dict,
    # no env mutation between here and there) — reuse it.
    try:
        post_snapshot = Config._snapshot_from_global_dict(working_copy)
    except (ValueError, ValidationError) as exc:
        log_if_not_mcp(
            "ERROR",
            "Remote-config: post-snapshot rejected — payload discarded ({})",
            exc,
        )
        return

    if not _delta_ok(half_merged, post_snapshot, command):
        return

    # 4e — write iff dict changed
    if working_copy == on_disk_dict:
        return
    persistence.backup_and_write(target, working_copy)


def _restore_refused(
    working_copy: dict[str, Any],
    on_disk_dict: dict[str, Any],
    path: str,
) -> None:
    """Restore a refused path from the pre-rules on-disk dict."""
    try:
        segments = rules.parse_path(path)
    except ValueError:
        return

    on_disk_present, on_disk_value = _lookup(on_disk_dict, segments)
    if on_disk_present:
        _set(working_copy, segments, on_disk_value)
    else:
        _delete(working_copy, segments)


def _lookup(root: dict[str, Any], segments: list[str]) -> tuple[bool, Any]:
    node: Any = root
    for seg in segments:
        if not isinstance(node, dict) or seg not in node:
            return False, None
        node = node[seg]
    return True, node


def _set(root: dict[str, Any], segments: list[str], value: Any) -> None:
    node = root
    for seg in segments[:-1]:
        if seg not in node or not isinstance(node[seg], dict):
            node[seg] = {}
        node = node[seg]
    node[segments[-1]] = value


def _delete(root: dict[str, Any], segments: list[str]) -> None:
    node: Any = root
    for seg in segments[:-1]:
        if not isinstance(node, dict) or seg not in node:
            return
        node = node[seg]
    if isinstance(node, dict):
        node.pop(segments[-1], None)


def _self_register_remote(
    working_copy: dict[str, Any],
    on_disk_dict: dict[str, Any],
    remote: Any,
) -> None:
    """Seed the discovery-layer ``remote_config.url`` / ``.auth_header`` into
    the on-disk global JSON on the first successful fetch, but never
    overwrite an existing on-disk value.

    Gap-fill only. A value already on disk — put there by a prior seeding
    run, an operator hand-edit, or an ``op: set`` rule — is durable across
    later runs. Without this guard, a one-off ``--remote-config-url`` (or a
    transient env override) would silently replace the operator's persisted
    URL on every invocation. To change a persisted value, edit the global
    JSON directly or push a rule that sets it.

    Persists the source layer's value verbatim. Intentional trade-off: a
    literal ``auth_header`` lands on disk in plaintext; use ``${VAR}`` to
    persist only the placeholder.
    """
    on_disk_block = on_disk_dict.get("remote_config") or {}
    working_block = working_copy.get("remote_config") or {}
    to_write: dict[str, Any] = {}
    for key, value in (("url", remote.url), ("auth_header", remote.auth_header)):
        if value is None:
            continue
        if on_disk_block.get(key) is not None:
            continue  # already persisted — never clobber
        if working_block.get(key) is not None:
            continue  # a rule set it this run — rule wins
        to_write[key] = value
    if to_write:
        merged = dict(working_block)
        merged.update(to_write)
        working_copy["remote_config"] = merged


def _codes_for(
    snapshot: Config, commands: set[str]
) -> set[tuple[str, ConfigErrorCode]]:
    """Collect (command, code) pairs across all commands under evaluation."""
    result: set[tuple[str, ConfigErrorCode]] = set()
    for cmd in commands:
        for code, _msg in snapshot.validate_for_command_structured(cmd, None):
            result.add((cmd, code))
    return result


def _delta_ok(pre: Config, post: Config, current_command: str) -> bool:
    """Accept iff ``E_post ⊆ E_pre``. Log new codes on rejection with
    the command each came from.
    """
    commands = {current_command} | set(PERSISTENCE_HAZARD_COMMANDS)
    e_pre = _codes_for(pre, commands)
    e_post = _codes_for(post, commands)
    new_codes = e_post - e_pre
    if not new_codes:
        return True
    for cmd, code in sorted(new_codes, key=lambda x: (x[0], x[1].value)):
        log_if_not_mcp(
            "ERROR",
            "Remote-config rejected: {}: {}",
            cmd,
            code.value,
        )
    return False
