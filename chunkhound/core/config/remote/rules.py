"""Rule parsing and application for the remote-config pipeline.

Public contracts:
- ``ENCLOSING_SUB_MODEL`` — the static path→BaseModel map used for per-rule
  sub-model validation. Built once at import from ``Config.model_fields``,
  unwrapping ``X | None`` unions so paths like ``embedding.api_key`` bind
  to ``EmbeddingConfig`` rather than ``types.UnionType`` (which has no
  ``model_validate`` and would silently break validation for every
  Optional sub-config).
- ``parse_path`` — accepts dotted (``a.b.c``) or slash (``a/b/c``) forms with
  an optional single leading ``.`` or ``/``. Both forms accepted for
  copy-paste ergonomics.
- ``apply_rule`` — dispatches ``merge`` / ``set`` / ``remove``. Rules that
  omit ``op`` default to ``merge``. ``merge`` collapses to ``set`` when
  either the value or the existing leaf isn't a dict (there's nothing to
  recursively merge into a non-dict). Missing intermediates are created on
  write; ``remove`` on a missing path is a no-op (logged as such).

Multiple rules may share the same ``id`` (path). They apply in list order —
later rules overwrite earlier ones at the same leaf. Failures are attributed
by 1-based ordinal position AND path in log messages, so duplicate-id rules
remain distinguishable: e.g. ``Remote-config rule 3 on 'database.provider'
schema_error: ...``. A rule with no ``id`` field is reported as ``rule N
(missing id)``.

Unknown top-level paths log a WARNING (``schema_error``) and skip the rule
rather than being silently dropped. Silent drops would let a typo like
``embeding.provider`` land as a no-op with no operator signal — the whole
point of a rule loop is to be able to see what did and didn't take effect.
The same check runs on the second segment (e.g., ``embedding.provder``),
because sub-models ignore unknown fields (pydantic's default; some also
set ``extra="ignore"`` explicitly for local-config forward compatibility)
and would otherwise silently absorb the typo. This applies to ``remove``
as well: removing an unknown leaf now surfaces as ``schema_error`` rather
than the previous ``no-op`` audit line — the leaf can't legitimately
exist under the schema, so silence would still hide the typo. Parent-level
``merge``/``set`` payloads (e.g. ``id=embedding, op=merge,
value={provder: ...}``) are recursively checked against the target
sub-model's fields — an unknown nested key surfaces as ``schema_error``
before mutation, preventing typos from being persisted to the global JSON.

The depth-2 assumption above (Config → sub-model → leaf) is enforced by
a shape-invariant tripwire in ``test_rules.py`` — if a sub-config gains
a nested ``BaseModel`` field the test fails, listing what to re-verify.
"""

import copy
import sys
import typing
from types import NoneType
from typing import Any

from packaging.version import InvalidVersion
from packaging.version import parse as parse_version
from pydantic import BaseModel, ValidationError

from chunkhound.core.config.config import Config
from chunkhound.utils.logging_guard import log_if_not_mcp


def _build_enclosing_map() -> dict[str, type[BaseModel]]:
    """Map top-level Config field name → enclosing BaseModel subclass.

    Unwraps ``X | None`` unions (currently ``embedding``, ``llm``,
    ``remote_config``) so per-rule sub-model construction has a real model
    class to instantiate. Skips scalar/collection fields (bool, Path, etc.)
    since they cannot enclose sub-paths.
    """
    result: dict[str, type[BaseModel]] = {}
    for name, info in Config.model_fields.items():
        annotation = info.annotation
        candidates: tuple[Any, ...]
        args = typing.get_args(annotation)
        if args:
            candidates = tuple(a for a in args if a is not NoneType)
        else:
            candidates = (annotation,)
        for candidate in candidates:
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                result[name] = candidate
                break
    return result


ENCLOSING_SUB_MODEL: dict[str, type[BaseModel]] = _build_enclosing_map()


def parse_path(path: str) -> list[str]:
    """Parse a dotted or slash-separated config path into segments.

    Accepts an optional single leading ``.`` or ``/``. Empty segments are
    rejected.
    """
    if not path:
        raise ValueError("empty path")
    stripped = path
    if stripped.startswith(".") or stripped.startswith("/"):
        stripped = stripped[1:]
    # Normalize slash to dot so we accept both forms uniformly.
    stripped = stripped.replace("/", ".")
    segments = stripped.split(".")
    if any(not s for s in segments):
        raise ValueError(f"invalid path {path!r}: empty segment")
    return segments


def _walk_to_parent(
    root: dict[str, Any], segments: list[str], *, create: bool
) -> dict[str, Any] | None:
    """Walk ``root`` to the parent dict of the leaf at ``segments``.

    When ``create`` is True, missing intermediate dicts are created.
    Returns ``None`` if a non-dict is encountered mid-walk (rules can't
    write past a scalar), or if any intermediate is missing under
    ``create=False``.
    """
    node: Any = root
    for seg in segments[:-1]:
        if not isinstance(node, dict):
            return None
        if seg not in node:
            if not create:
                return None
            node[seg] = {}
        elif not isinstance(node[seg], dict):
            if not create:
                return None
            # Overwrite non-dict with dict so the caller can extend past a
            # scalar leaf. Per-rule sub-model validation catches any resulting
            # shape mismatch and rolls back.
            node[seg] = {}
        node = node[seg]
    if not isinstance(node, dict):
        return None
    return node


def _predicate_matches(
    predicates: dict[str, Any] | None,
    half_merged: Config,
) -> tuple[bool, str | None]:
    """Evaluate rule predicates.

    Returns ``(matches, schema_error)``. ``schema_error`` is non-None only
    when a predicate key is misspelled/unknown — the caller surfaces that
    to the operator as a WARNING, matching how unknown ops and top-level
    paths are treated (see module docstring).
    """
    if not predicates:
        return True, None
    for key, expected in predicates.items():
        if key == "os":
            wanted = expected if isinstance(expected, list) else [expected]
            if sys.platform not in wanted:
                return False, None
        elif key == "existing":
            wanted = expected if isinstance(expected, list) else [expected]
            for path in wanted:
                if not _existing(half_merged, path):
                    return False, None
        else:
            return False, f"unknown predicate {key!r}"
    return True, None


def _existing(half_merged: Config, path: str) -> bool:
    """True iff ``path`` resolves to a non-None value in ``half_merged``."""
    try:
        segments = parse_path(path)
    except ValueError:
        return False
    node: Any = half_merged
    for seg in segments:
        if isinstance(node, BaseModel):
            node = getattr(node, seg, None)
        elif isinstance(node, dict):
            node = node.get(seg)
        else:
            return False
        if node is None:
            return False
    return True


def _version_gate_ok(
    rule: dict[str, Any], current_version: str
) -> tuple[bool, bool]:
    """Return ``(satisfied, unparseable)`` for the rule's ``min_chunkhound_version``.

    Unparseable is treated as not-satisfied so the rule is still skipped,
    but the caller can log a distinct message rather than a misleading
    "requires >= <garbage>" line.
    """
    required = rule.get("min_chunkhound_version")
    if required is None:
        return True, False
    try:
        required_parsed = parse_version(str(required))
    except InvalidVersion:
        return False, True
    # A malformed ``current_version`` is a build-time bug (hatch-vcs tag
    # gone sideways), not a rule-authoring bug. Let ``InvalidVersion``
    # propagate: mislabelling it as "unparseable rule version" would
    # point operators at the wrong place to fix it.
    return parse_version(current_version) >= required_parsed, False


def _sub_model_for(top_segment: str) -> type[BaseModel] | None:
    return ENCLOSING_SUB_MODEL.get(top_segment)


def _unwrap_optional_basemodel(annotation: Any) -> type[BaseModel] | None:
    """Return the ``BaseModel`` inside ``X | None`` (or ``X`` directly), else None.

    Mirrors the union handling in ``_build_enclosing_map`` so nested-field
    recursion in ``_check_payload_keys`` follows the same shape rules as
    the top-level enclosing map.
    """
    candidates = typing.get_args(annotation) or (annotation,)
    for candidate in candidates:
        if candidate is NoneType:
            continue
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def _check_payload_keys(
    model: type[BaseModel],
    payload: Any,
    path_prefix: list[str],
) -> str | None:
    """Return the dotted path of the first unknown key in ``payload``, else None.

    Walks ``payload`` recursively and validates every dict key against the
    corresponding model's ``model_fields``. Sub-models set ``extra="ignore"``
    for local-config forward compatibility, so pydantic would otherwise drop
    typos silently — this pre-mutation check surfaces them as ``schema_error``.

    Only descends into a field's payload when the field annotation resolves
    to a ``BaseModel`` subclass. Non-model dict fields (``dict[str, str]``,
    etc.) may legitimately carry arbitrary keys, so recursion stops there.
    """
    if not isinstance(payload, dict):
        return None
    fields = model.model_fields
    for key, sub_payload in payload.items():
        if key not in fields:
            return ".".join([*path_prefix, key])
        nested = _unwrap_optional_basemodel(fields[key].annotation)
        if nested is not None:
            err = _check_payload_keys(nested, sub_payload, [*path_prefix, key])
            if err is not None:
                return err
    return None


def _validate_sub_model(
    top_segment: str,
    working_copy: dict[str, Any],
) -> tuple[bool, str | None]:
    """Construct the enclosing sub-model for ``top_segment`` from the current
    working-copy state. Returns (ok, error_message).
    """
    sub_model = _sub_model_for(top_segment)
    if sub_model is None:
        return False, f"unknown top-level path {top_segment!r}"
    subtree = working_copy.get(top_segment)
    if subtree is None:
        # Nothing to validate — a rule that only removed keys.
        return True, None
    if not isinstance(subtree, dict):
        return False, (
            f"expected dict at {top_segment!r}, got {type(subtree).__name__}"
        )
    try:
        sub_model(**subtree)
    except ValidationError as exc:
        return False, f"sub-model validation failed: {exc}"
    return True, None


def _format_rule_ref(rule_index: int, rule: dict[str, Any]) -> str:
    """Format a rule reference for log messages: ``rule N on 'path'``.

    Falls back to ``rule N (missing id)`` when the field is absent or an
    empty string, and ``rule N (invalid id: <repr>)`` when it exists but
    isn't a usable string — so a bogus ``"id": 123`` isn't mislabelled as
    "missing" while the ordinal still anchors the log entry.
    """
    if "id" not in rule:
        return f"rule {rule_index} (missing id)"
    path = rule["id"]
    if isinstance(path, str):
        if path:
            return f"rule {rule_index} on {path!r}"
        return f"rule {rule_index} (missing id)"
    return f"rule {rule_index} (invalid id: {path!r})"


def apply_rule(
    rule: dict[str, Any],
    working_copy: dict[str, Any],
    half_merged: Config,
    current_version: str,
    *,
    rule_index: int,
) -> None:
    """Apply a single rule to ``working_copy`` in place.

    Rules that omit ``op`` default to ``merge`` — the module contract.

    On any per-rule failure (schema error, version gate, predicate mismatch,
    sub-model validation), the rule is silently skipped after logging a
    WARNING (except for the predicate-mismatch case, which is not an error
    — it's the rule's normal skip path). Successful applies emit an INFO
    audit line with op + path only — never the value, since paths like
    ``embedding.api_key`` legitimately carry secrets. A ``remove`` targeting
    a missing path logs ``no-op`` instead of ``applied`` so operators can
    distinguish evaluated-but-inert rules from ones that mutated state.

    ``rule_index`` is the 1-based position of the rule in the envelope's
    ``rules`` list. Included in every WARNING and INFO line so that
    duplicate-id rules remain distinguishable in operator logs.
    """
    ref = _format_rule_ref(rule_index, rule)
    # Default op=merge for author convenience. Tradeoff: a typo like
    # `"opp": "set"` silently applies as merge; INFO audit line catches it.
    op = rule.get("op", "merge")
    if op not in {"merge", "set", "remove"}:
        log_if_not_mcp(
            "WARNING",
            "Remote-config {} schema_error: unknown op {!r}",
            ref,
            op,
        )
        return

    version_ok, version_unparseable = _version_gate_ok(rule, current_version)
    if not version_ok:
        if version_unparseable:
            log_if_not_mcp(
                "WARNING",
                "Remote-config {} version_gate: unparseable "
                "min_chunkhound_version {!r} — skipping rule",
                ref,
                rule.get("min_chunkhound_version"),
            )
        else:
            log_if_not_mcp(
                "WARNING",
                "Remote-config {} version_gate: requires >= {}",
                ref,
                rule.get("min_chunkhound_version"),
            )
        return

    matches, pred_err = _predicate_matches(rule.get("when"), half_merged)
    if pred_err is not None:
        log_if_not_mcp(
            "WARNING",
            "Remote-config {} schema_error: {}",
            ref,
            pred_err,
        )
        return
    if not matches:
        return  # normal predicate miss — no log, by design

    try:
        segments = parse_path(rule["id"])
    except (KeyError, ValueError) as exc:
        log_if_not_mcp(
            "WARNING",
            "Remote-config {} schema_error: {}",
            ref,
            exc,
        )
        return

    top_segment = segments[0]
    if top_segment not in ENCLOSING_SUB_MODEL:
        # Log rather than silently drop — see module docstring rationale.
        log_if_not_mcp(
            "WARNING",
            "Remote-config {} schema_error: unknown top-level path {!r}",
            ref,
            top_segment,
        )
        return

    # Reject depth-2 typos up front; sub-models set ``extra="ignore"`` and
    # would silently absorb them (see module docstring for full rationale).
    sub_model = ENCLOSING_SUB_MODEL[top_segment]
    if len(segments) >= 2 and segments[1] not in sub_model.model_fields:
        log_if_not_mcp(
            "WARNING",
            "Remote-config {} schema_error: unknown path {!r}",
            ref,
            ".".join(segments),
        )
        return

    # Reject unknown keys inside a parent-level merge/set payload for the
    # same reason (extra="ignore" would drop them silently, then persist
    # the raw payload dict to global JSON via deep_merge). Only runs when
    # the rule addresses the sub-model itself (``id=<top>``) — deeper
    # paths target a scalar/leaf whose value-shape is validated by
    # ``_validate_sub_model``; running the payload check against
    # ``sub_model`` there would misread a dict value for a scalar leaf
    # as an unknown top-level field, and would reject legitimate rules
    # targeting a future dict-typed leaf (e.g. ``dict[str, str]``).
    if op in {"merge", "set"} and len(segments) == 1:
        value = rule.get("value")
        if isinstance(value, dict):
            unknown = _check_payload_keys(sub_model, value, segments)
            if unknown is not None:
                log_if_not_mcp(
                    "WARNING",
                    "Remote-config {} schema_error: unknown path {!r}",
                    ref,
                    unknown,
                )
                return

    # Snapshot the top-level subtree so we can roll back a rule that fails
    # per-rule sub-model validation without contaminating later rules.
    snapshot = copy.deepcopy(working_copy.get(top_segment))

    # Pre-inspect state so the audit line can reflect what actually happens
    # (no-op remove vs. merge that falls through to set), not just the
    # requested op.
    parent = _walk_to_parent(working_copy, segments, create=False)
    leaf = segments[-1]
    existing = parent.get(leaf) if parent is not None else None
    collapsed_merge_to_set = False
    remove_no_op = False
    if op == "remove":
        remove_no_op = parent is None or leaf not in parent
        _apply_remove(working_copy, segments)
    else:
        value = rule.get("value")
        if op == "merge" and isinstance(value, dict) and isinstance(existing, dict):
            _apply_merge(working_copy, segments, value)
        else:
            _apply_set(working_copy, segments, value)
            collapsed_merge_to_set = op == "merge"

    ok, err = _validate_sub_model(top_segment, working_copy)
    if not ok:
        # Roll back this rule's mutation to keep later rules working on a
        # clean subtree.
        if snapshot is None:
            working_copy.pop(top_segment, None)
        else:
            working_copy[top_segment] = snapshot
        log_if_not_mcp(
            "WARNING",
            "Remote-config {} schema_error: {}",
            ref,
            err,
        )
        return

    if remove_no_op:
        log_if_not_mcp(
            "INFO", "Remote-config {} no-op: op=remove (path absent)", ref
        )
        return

    suffix = " (collapsed→set)" if collapsed_merge_to_set else ""
    log_if_not_mcp("INFO", "Remote-config {} applied: op={}{}", ref, op, suffix)


def _apply_set(root: dict[str, Any], segments: list[str], value: Any) -> None:
    parent = _walk_to_parent(root, segments, create=True)
    if parent is None:
        return
    parent[segments[-1]] = value


def _apply_merge(
    root: dict[str, Any], segments: list[str], value: dict[str, Any]
) -> None:
    # apply_rule guarantees the leaf is a dict; collapse-to-set is handled
    # there so the audit line can annotate it.
    parent = _walk_to_parent(root, segments, create=True)
    if parent is None:
        return
    Config.deep_merge(parent[segments[-1]], value)


def _apply_remove(root: dict[str, Any], segments: list[str]) -> None:
    parent = _walk_to_parent(root, segments, create=False)
    if parent is None:
        return
    parent.pop(segments[-1], None)
