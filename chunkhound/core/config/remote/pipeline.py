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
      Scored twice — persisted global JSON (overlays skipped) AND the
      fully merged invocation — across the current command plus every
      persistence-hazard command, so a `search` invocation can't
      silently make `mcp` unsafe and a one-off CLI flag can't mask a
      newly written global hazard.
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
from .fetcher import url_scheme_ok

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
        half_merged = Config.snapshot_from_global_dict(on_disk_dict)
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

    # 4c — enforce URL-scheme policy on rule-set values, then self-register.
    #      Validate first so an unsafe rule URL is reverted (to the on-disk
    #      value, or removed if there is none) *before* self-registration
    #      decides whether the URL slot is empty. If the guard removed the
    #      unsafe rule URL and no prior value is on disk, self-register can
    #      still seed the discovery-layer URL — the fetcher scheme-checked
    #      it before the fetch, so it does not need a second guard here.
    _validate_remote_url(working_copy, on_disk_dict)
    _self_register_remote(working_copy, on_disk_dict, remote)

    # 4d — terminal delta-only gate. Three substrates (persisted /
    # half_merged / active); see `snapshot_for_persisted_gate` /
    # `snapshot_from_global_dict` / `snapshot_for_delta_gate` docstrings.
    # `half_merged` from step 4a is reused as the pre side — identical
    # layer selector and `on_disk_dict` is not mutated between there and
    # here. Persisted must skip env too, so it is built fresh here.
    try:
        pre_persisted = Config.snapshot_for_persisted_gate(on_disk_dict)
        post_persisted = Config.snapshot_for_persisted_gate(working_copy)
        post_half_merged = Config.snapshot_from_global_dict(working_copy)
        pre_active = Config.snapshot_for_delta_gate(on_disk_dict, args)
        post_active = Config.snapshot_for_delta_gate(working_copy, args)
    except (ValueError, ValidationError) as exc:
        log_if_not_mcp(
            "ERROR",
            "Remote-config: post-snapshot rejected — payload discarded ({})",
            exc,
        )
        return

    if not _delta_ok(
        pre_persisted=pre_persisted,
        post_persisted=post_persisted,
        pre_half_merged=half_merged,
        post_half_merged=post_half_merged,
        pre_active=pre_active,
        post_active=post_active,
        current_command=command,
    ):
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
    """Restore a refused path from the pre-rules on-disk dict.

    If a rule mutated the refused path, emit a WARNING naming the key —
    apply_rule's INFO ``applied`` line would otherwise silently disagree
    with disk state. The rule value is not logged (refused-path values
    leak install topology).
    """
    try:
        segments = rules.parse_path(path)
    except ValueError:
        return

    working_present, working_value = _lookup(working_copy, segments)
    on_disk_present, on_disk_value = _lookup(on_disk_dict, segments)

    if working_present == on_disk_present and working_value == on_disk_value:
        # Rule didn't change it — nothing to revert or warn about.
        return

    log_if_not_mcp(
        "WARNING",
        "Remote-config refused_path {!r}: rule effect scrubbed "
        "(operator-owned key)",
        path,
    )

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
            # A rule set it this run — rule wins. (Unsafe URL rules were
            # already reverted or removed by `_validate_remote_url`, so
            # any surviving working value came from a safe rule; a
            # pre-rules on-disk value would have been caught by the
            # `on_disk_block.get(key) is not None` check above.)
            continue
        to_write[key] = value
    if to_write:
        merged = dict(working_block)
        merged.update(to_write)
        working_copy["remote_config"] = merged


def _validate_remote_url(
    working_copy: dict[str, Any],
    on_disk_dict: dict[str, Any],
) -> None:
    """Revert an unsafe rule-set ``remote_config.url`` before self-register.

    Same policy the fetcher enforces on the initial URL and every redirect
    hop: ``https``, or ``http`` to a loopback host. Any ``https://`` URL
    passes — server-driven rotation to a new origin is the supported design,
    and this function does not second-guess it. The only case blocked is
    ``http://non-loopback/x``, which the fetcher will refuse on the next
    run: persisting it would brick the pipeline until an operator hand-edits
    the global config.

    Runs before ``_self_register_remote`` so that when the guard deletes an
    unsafe rule URL that has no on-disk fallback, self-register can still
    seed the discovery-layer (CLI / env / global) URL — one successful
    ``--remote-config-url`` remains sufficient to make later runs
    self-sufficient, even when the envelope also pushed a bad URL rule.
    The discovery URL itself does not need a second scheme check here
    because the fetcher already applied the same rule before the fetch.

    A pre-existing unsafe value that this run did not modify is left alone
    and silent: the fetcher will refuse it on the next attempt, and warning
    every run about the same operator hand-edit would be noise.
    """
    segments = ["remote_config", "url"]
    working_present, working_value = _lookup(working_copy, segments)
    if not working_present or working_value is None:
        return
    if isinstance(working_value, str) and url_scheme_ok(working_value):
        return

    on_disk_present, on_disk_value = _lookup(on_disk_dict, segments)
    if on_disk_present and on_disk_value == working_value:
        return  # unchanged this run — not a rule/discovery violation

    log_if_not_mcp(
        "WARNING",
        "Remote-config refused remote_config.url: scheme must be https "
        "(or http to a loopback host) — reverting to on-disk value",
    )

    if on_disk_present:
        _set(working_copy, segments, on_disk_value)
    else:
        _delete(working_copy, segments)


def _codes_for(
    snapshot: Config, commands: set[str]
) -> set[tuple[str, ConfigErrorCode]]:
    """Collect (command, code) pairs across all commands under evaluation.

    Each command is evaluated against its worst-case runtime-state
    snapshot (see ``Config._hazard_snapshot_for_command``) so guards
    gated on fields like ``mcp.transport`` fire regardless of the
    snapshot's resolved transport. The substitution applies
    symmetrically to both pre and post sides via ``_delta_ok``, so
    pre-existing hazards still appear in ``E_pre`` and remain accepted
    under ``E_post ⊆ E_pre``.
    """
    result: set[tuple[str, ConfigErrorCode]] = set()
    for cmd in commands:
        target = snapshot._hazard_snapshot_for_command(cmd)
        for code, _msg in target.validate_for_command_structured(cmd, None):
            result.add((cmd, code))
    return result


def _new_codes(
    pre: Config, post: Config, commands: set[str]
) -> set[tuple[str, ConfigErrorCode]]:
    """Return ``E_post - E_pre`` for ``commands``."""
    return _codes_for(post, commands) - _codes_for(pre, commands)


def _delta_ok(
    *,
    pre_persisted: Config,
    post_persisted: Config,
    pre_half_merged: Config,
    post_half_merged: Config,
    pre_active: Config,
    post_active: Config,
    current_command: str,
) -> bool:
    """Accept iff ``E_post ⊆ E_pre`` on *all three* gate substrates.

    ``persisted`` is the on-disk global JSON (overlays skipped).
    ``half_merged`` is env + on-disk JSON (CLI / local / --config
    skipped) — catches hazards that env activates and that a future
    invocation without the current CLI/local overlays would surface.
    ``active`` is the fully merged invocation. New codes from any side
    fail the gate. Logged with the substrate name so an overlay-only
    mask vs an env-activated hazard vs an invocation-only hazard are
    distinguishable.
    """
    commands = {current_command} | set(PERSISTENCE_HAZARD_COMMANDS)
    persisted_new = _new_codes(pre_persisted, post_persisted, commands)
    half_merged_new = _new_codes(pre_half_merged, post_half_merged, commands)
    active_new = _new_codes(pre_active, post_active, commands)
    if not persisted_new and not half_merged_new and not active_new:
        return True
    for source, new_codes in (
        ("persisted", persisted_new),
        ("half_merged", half_merged_new),
        ("active", active_new),
    ):
        for cmd, code in sorted(new_codes, key=lambda x: (x[0], x[1].value)):
            log_if_not_mcp(
                "ERROR",
                "Remote-config rejected ({}): {}: {}",
                source,
                cmd,
                code.value,
            )
    return False
