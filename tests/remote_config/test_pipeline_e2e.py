"""End-to-end contract tests for the remote-config pipeline.

Locks the operator-visible behaviors that a change to the pipeline could
silently break:

 1. Successful fetch + rule application writes expected post-rules dict.
 2. `.bak` file created iff prior content existed on disk.
 3. Payload rejected when it would introduce a new validation error under
    any persistence-hazard command (cross-command safety). The delta gate
    scores *both* the persisted global JSON (overlays skipped) and the
    fully merged invocation; either new code discards the write. Overlay
    masking (CLI ``--auth-token``, local ``mcp.host``, env token) is
    rejected by the persisted substrate.
 4. `remote_config.url` / `auth_header` gap-filled from the discovery layer
    on the first successful fetch; an on-disk value is durable and is
    never overwritten by a differing transient CLI/env value on a later
    run. Rule-driven writes to those keys still win.
 5. `remote_config` in `.chunkhound.json` / `--config` is stripped with WARNING.
 6. Rule failing sub-model validation is skipped (`schema_error`), rest applies.
 7. Depth-2 typo path → `schema_error` on both `set` and `remove`
    (typo `remove` must not surface as `no-op`).
 8. Rule with `min_chunkhound_version` > running version is skipped, rest applies.
 9. Rule with unparseable `min_chunkhound_version` → dedicated WARNING
    (not `requires >= <garbage>`).
10. Envelope `min_chunkhound_version` too high → whole payload discarded.
11. Backup-write-fatal (OSError) → sys.exit(1), no partial write.
12. Persisted file mode is `0o600` on POSIX (global config may embed secrets).
13. Backup `.bak` file mode is `0o600` on POSIX (mirrors the primary write).
14. Equal-dict case: no rule net-change → no write, no `.bak`.
15. Envelope `version != 1` → whole payload discarded.
16. Rule targeting a `REFUSED_PATHS` key is scrubbed with a WARNING
    naming the key: reverts to the operator's on-disk value when present,
    deletes otherwise. Silent when no rule touched the key. Parent-level
    `op: merge` sneaking a refused sub-key is caught the same way.

Tests exercise the pipeline through its public entry
(`run_remote_config_fetch`) and observe disk state, argparse-shaped `args`,
and log output — the operator-visible contract, not internals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from chunkhound.core.config.remote import run_remote_config_fetch

# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


def _args(**overrides: Any) -> argparse.Namespace:
    """Build an argparse.Namespace shaped like a `search` invocation."""
    defaults: dict[str, Any] = {
        "command": "search",
        "path": None,
        "config": None,
        "remote_config_url": "https://remote.test/config.json",
        "remote_config_auth_header": None,
        "no_embeddings": False,
        "verbose": False,
        "debug": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class _FakeFetcher:
    """Drop-in replacement for ``fetcher.fetch`` that returns canned payloads."""

    def __init__(self, envelope: Any | None) -> None:
        self._envelope = envelope
        self.calls: list[tuple[str, str | None]] = []

    async def __call__(self, url: str, auth_header: str | None) -> Any | None:
        self.calls.append((url, auth_header))
        return self._envelope


def _install_fetch(
    monkeypatch: pytest.MonkeyPatch, envelope: Any | None
) -> _FakeFetcher:
    fake = _FakeFetcher(envelope)
    # Pipeline imports fetcher as a module and calls fetcher.fetch — patch the
    # attribute on that module so both `from . import fetcher` sees it.
    monkeypatch.setattr(
        "chunkhound.core.config.remote.pipeline.fetcher.fetch", fake
    )
    return fake


def _read_target(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _target_path(home: Path) -> Path:
    return home / ".chunkhound.json"


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


async def test_successful_apply_writes_expected_dict(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = {
        "version": 1,
        "rules": [
            {
                "id": "database.provider",
                "op": "set",
                "value": "duckdb",
            }
        ],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(), "search")

    target = _target_path(_isolate)
    assert target.exists()
    data = _read_target(target)
    assert data["database"]["provider"] == "duckdb"


async def test_bak_created_iff_prior_content_existed(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target_path(_isolate)
    target.write_text(json.dumps({"database": {"provider": "duckdb"}}))
    bak = target.with_suffix(target.suffix + ".bak")
    assert not bak.exists()

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "127.0.0.1"}],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(), "search")
    assert bak.exists(), ".bak should be created when target pre-existed"


async def test_no_bak_on_first_write(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target_path(_isolate)
    bak = target.with_suffix(target.suffix + ".bak")

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "127.0.0.1"}],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(), "search")
    assert target.exists()
    assert not bak.exists(), ".bak should NOT be created on first write"


async def test_terminal_gate_rejects_cross_command_hazard(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pushing mcp.host=0.0.0.0 (no auth) during a `search` invocation would
    # introduce MCP_NON_LOOPBACK_NO_AUTH into the `mcp` command's gate. Even
    # though `search` itself doesn't trip that gate, `mcp` is in
    # PERSISTENCE_HAZARD_COMMANDS and gets checked.
    target = _target_path(_isolate)
    target.write_text(json.dumps({"mcp": {"transport": "http"}}))
    original_content = target.read_text()

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "0.0.0.0"}],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(command="search"), "search")

    # Target unchanged.
    assert target.read_text() == original_content


async def test_terminal_gate_rejects_cli_http_transport_non_loopback(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # All-sources substrate: `--transport http` is only on the CLI. The
    # persisted JSON has no transport, so without merging the active
    # invocation the `MCP_NON_LOOPBACK_NO_AUTH` guard (keyed on
    # `transport == "http"`) would never see HTTP from this layer. The
    # gate must still reject — today the persisted side also rejects via
    # `_hazard_snapshot_for_command`, so we assert the `active` log tag
    # to keep the all-sources comparison load-bearing.
    target = _target_path(_isolate)
    assert not target.exists()

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "0.0.0.0"}],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        await run_remote_config_fetch(
            _args(command="mcp", transport="http"), "mcp"
        )
    finally:
        logger.remove(handler_id)

    assert not target.exists()
    assert any("rejected (active)" in m for m in messages), (
        f"expected all-sources substrate to reject, got messages={messages}"
    )


async def test_terminal_gate_rejects_cli_http_transport_cors(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # All-sources substrate for `MCP_CORS_NO_AUTH` — same shape as the
    # non-loopback case. Assert `active` so dropping that comparison
    # cannot hide behind the persisted + force-http reject.
    target = _target_path(_isolate)
    assert not target.exists()

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.cors", "op": "set", "value": True}],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        await run_remote_config_fetch(
            _args(command="mcp", transport="http"), "mcp"
        )
    finally:
        logger.remove(handler_id)

    assert not target.exists()
    assert any("rejected (active)" in m for m in messages), (
        f"expected all-sources substrate to reject, got messages={messages}"
    )


async def test_terminal_gate_rejects_local_http_transport_non_loopback(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # All-sources substrate: `mcp.transport=http` comes from the
    # project-local `.chunkhound.json`, not the CLI. Assert `active`.
    project = tmp_path / "project"
    project.mkdir()
    (project / ".chunkhound.json").write_text(
        json.dumps({"mcp": {"transport": "http"}})
    )

    target = _target_path(_isolate)
    assert not target.exists()

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "0.0.0.0"}],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        await run_remote_config_fetch(
            _args(command="mcp", path=project), "mcp"
        )
    finally:
        logger.remove(handler_id)

    assert not target.exists()
    assert any("rejected (active)" in m for m in messages), (
        f"expected all-sources substrate to reject, got messages={messages}"
    )


async def test_terminal_gate_rejects_local_http_transport_cors(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # All-sources CORS vector via local JSON. Assert `active`.
    project = tmp_path / "project"
    project.mkdir()
    (project / ".chunkhound.json").write_text(
        json.dumps({"mcp": {"transport": "http"}})
    )

    target = _target_path(_isolate)
    assert not target.exists()

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.cors", "op": "set", "value": True}],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        await run_remote_config_fetch(
            _args(command="mcp", path=project), "mcp"
        )
    finally:
        logger.remove(handler_id)

    assert not target.exists()
    assert any("rejected (active)" in m for m in messages), (
        f"expected all-sources substrate to reject, got messages={messages}"
    )


async def test_terminal_gate_accepts_preexisting_hazard_under_active_transport(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression guard against over-tight semantics: if the operator has
    # already accepted the `MCP_NON_LOOPBACK_NO_AUTH` risk (host=0.0.0.0
    # already on disk under HTTP transport), the delta gate must not
    # block rules that leave that hazard unchanged. `_delta_ok` compares
    # `E_post ⊆ E_pre`, so a pre-existing error must appear on both sides
    # and be treated as accepted.
    target = _target_path(_isolate)
    target.write_text(json.dumps({"mcp": {"host": "0.0.0.0"}}))

    envelope = {
        "version": 1,
        "rules": [{"id": "database.provider", "op": "set", "value": "duckdb"}],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(
        _args(command="mcp", transport="http"), "mcp"
    )

    data = _read_target(target)
    assert data["database"]["provider"] == "duckdb"
    # Pre-existing hazard preserved verbatim.
    assert data["mcp"]["host"] == "0.0.0.0"


async def test_terminal_gate_rejects_mcp_host_stdio_default(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Residual delta-gate gap: with no HTTP transport ANYWHERE (no CLI
    # flag, no env, no on-disk `mcp.transport`, no local JSON), the
    # snapshot's `mcp.transport` falls back to `"stdio"` and the
    # `MCP_NON_LOOPBACK_NO_AUTH` guard (gated on `transport == "http"`)
    # never fires on either side of `_delta_ok`. A `search` invocation
    # would then silently persist `mcp.host=0.0.0.0`, breaking the next
    # `mcp --transport http` startup. `_codes_for` must evaluate the
    # `mcp` hazard under a worst-case `transport='http'` snapshot so
    # the guard fires and the rule is rejected regardless of the active
    # invocation's transport.
    target = _target_path(_isolate)
    assert not target.exists()

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "0.0.0.0"}],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(command="search"), "search")

    assert not target.exists()


async def test_terminal_gate_rejects_mcp_cors_stdio_default(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Symmetric to the host case above but for the `MCP_CORS_NO_AUTH`
    # guard which shares the identical `transport == "http"` gate.
    target = _target_path(_isolate)
    assert not target.exists()

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.cors", "op": "set", "value": True}],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(command="search"), "search")

    assert not target.exists()


async def test_terminal_gate_accepts_preexisting_hazard_under_stdio_default(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Locks in that worst-case transport substitution applies to BOTH
    # sides of the delta gate. A naive fix that only forced transport
    # on the `post` side would emit `MCP_NON_LOOPBACK_NO_AUTH` only in
    # `E_post`, spuriously blocking unrelated rules whenever the
    # operator has already accepted the non-loopback host on disk. The
    # substitution must be symmetric — the hazard shows in `E_pre` too,
    # so `E_post ⊆ E_pre` accepts unrelated changes.
    target = _target_path(_isolate)
    target.write_text(json.dumps({"mcp": {"host": "0.0.0.0"}}))

    envelope = {
        "version": 1,
        "rules": [{"id": "database.provider", "op": "set", "value": "duckdb"}],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(command="search"), "search")

    data = _read_target(target)
    assert data["database"]["provider"] == "duckdb"
    assert data["mcp"]["host"] == "0.0.0.0"


async def test_terminal_gate_rejects_cli_auth_token_masking_non_loopback(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Persisted-dict substrate: a one-off `--auth-token` makes the
    # all-sources snapshot valid on both sides, so a rule writing
    # `mcp.host=0.0.0.0` (no token) to global JSON would be approved if
    # we only scored the invocation. The persisted snapshot skips CLI
    # and must reject. Assert `persisted` and that `active` did *not*
    # fire — otherwise this wouldn't prove the overlay-mask path.
    target = _target_path(_isolate)
    assert not target.exists()

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "0.0.0.0"}],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        await run_remote_config_fetch(
            _args(
                command="mcp",
                transport="http",
                host="0.0.0.0",
                auth_token="secret",
            ),
            "mcp",
        )
    finally:
        logger.remove(handler_id)

    assert not target.exists()
    assert any("rejected (persisted)" in m for m in messages), (
        f"expected persisted substrate to reject overlay mask, "
        f"got messages={messages}"
    )
    assert not any("rejected (active)" in m for m in messages), (
        f"active substrate must be masked by --auth-token; if it also "
        f"rejects, this test no longer isolates the persisted check: "
        f"{messages}"
    )


async def test_terminal_gate_rejects_local_host_masking_non_loopback(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Persisted-dict substrate: project-local `.chunkhound.json` already
    # has `mcp.host=0.0.0.0`, so the all-sources snapshot shows the
    # hazard on both sides and would approve copying it into the global
    # file. Persisted skips local and must reject.
    project = tmp_path / "project"
    project.mkdir()
    (project / ".chunkhound.json").write_text(
        json.dumps({"mcp": {"host": "0.0.0.0"}})
    )

    target = _target_path(_isolate)
    assert not target.exists()

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "0.0.0.0"}],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        await run_remote_config_fetch(
            _args(command="search", path=project), "search"
        )
    finally:
        logger.remove(handler_id)

    assert not target.exists()
    assert any("rejected (persisted)" in m for m in messages), (
        f"expected persisted substrate to reject local-file mask, "
        f"got messages={messages}"
    )
    assert not any("rejected (active)" in m for m in messages), (
        f"active substrate must be masked by local mcp.host; if it also "
        f"rejects, this test no longer isolates the persisted check: "
        f"{messages}"
    )


async def test_terminal_gate_rejects_env_auth_token_masking_non_loopback(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Persisted-dict substrate: `CHUNKHOUND_MCP__AUTH_TOKEN` is a standing
    # overlay, not part of the file. Scoring env+disk would hide a write
    # of `mcp.host=0.0.0.0` without a token. Persisted skips env.
    monkeypatch.setenv("CHUNKHOUND_MCP__AUTH_TOKEN", "from-env")

    target = _target_path(_isolate)
    assert not target.exists()

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "0.0.0.0"}],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        await run_remote_config_fetch(_args(command="search"), "search")
    finally:
        logger.remove(handler_id)

    assert not target.exists()
    assert any("rejected (persisted)" in m for m in messages), (
        f"expected persisted substrate to reject env-token mask, "
        f"got messages={messages}"
    )
    assert not any("rejected (active)" in m for m in messages), (
        f"active substrate must be masked by CHUNKHOUND_MCP__AUTH_TOKEN; "
        f"if it also rejects, this test no longer isolates the persisted "
        f"check: {messages}"
    )


async def test_terminal_gate_rejects_env_host_hazard_only_visible_to_half_merged(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Half-merged substrate: env supplies a non-loopback `mcp.host` that the
    # persisted view (no env) cannot see, and a CLI `--auth-token` masks it
    # on the active view. A rule removing `mcp.auth_token` from the JSON
    # would leave a future invocation (without --auth-token) exposed to
    # MCP_NON_LOOPBACK_NO_AUTH — env host + no auth in JSON. Persisted PASSes
    # (host defaults to loopback with env skipped) and active PASSes (CLI
    # auth still masks). Only half_merged catches it. Assert half_merged
    # alone rejects, so dropping that substrate cannot hide behind either
    # sibling.
    monkeypatch.setenv("CHUNKHOUND_MCP__HOST", "0.0.0.0")

    target = _target_path(_isolate)
    target.write_text(json.dumps({"mcp": {"auth_token": "existing"}}))
    original_bytes = target.read_bytes()

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.auth_token", "op": "remove"}],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    try:
        await run_remote_config_fetch(
            _args(command="search", auth_token="cli-secret"), "search"
        )
    finally:
        logger.remove(handler_id)

    assert target.read_bytes() == original_bytes
    assert any("rejected (half_merged)" in m for m in messages), (
        f"expected half_merged substrate to reject env-host hazard, "
        f"got messages={messages}"
    )
    assert not any("rejected (persisted)" in m for m in messages), (
        f"persisted must not see env-supplied host; if it also rejects, "
        f"this test no longer isolates half_merged: {messages}"
    )
    assert not any("rejected (active)" in m for m in messages), (
        f"active must be masked by CLI --auth-token; if it also rejects, "
        f"this test no longer isolates half_merged: {messages}"
    )


async def test_self_register_remote_config_when_missing(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = {
        "version": 1,
        "rules": [{"id": "database.provider", "op": "set", "value": "duckdb"}],
    }
    _install_fetch(monkeypatch, envelope)

    args = _args(
        remote_config_url="https://remote.test/config.json",
        remote_config_auth_header="Bearer abc",
    )
    await run_remote_config_fetch(args, "search")

    data = _read_target(_target_path(_isolate))
    assert data["remote_config"]["url"] == "https://remote.test/config.json"
    assert data["remote_config"]["auth_header"] == "Bearer abc"


async def test_self_register_preserves_on_disk_url(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Gap-fill: an on-disk URL is durable. A one-off `--remote-config-url`
    # that differs from what's on disk applies for this run's fetch (via
    # the restricted merge, which is why the envelope arrives) but must
    # NOT be persisted — otherwise a single transient invocation could
    # silently replace the operator's fleet-wide URL and become permanent.
    target = _target_path(_isolate)
    target.write_text(
        json.dumps({"remote_config": {"url": "https://original.test/x.json"}})
    )
    envelope = {
        "version": 1,
        "rules": [{"id": "database.provider", "op": "set", "value": "duckdb"}],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(
        _args(remote_config_url="https://remote.test/config.json"), "search"
    )

    data = _read_target(target)
    # On-disk URL preserved despite a differing CLI URL used for the fetch.
    assert data["remote_config"]["url"] == "https://original.test/x.json"
    # Unrelated rule still landed.
    assert data["database"]["provider"] == "duckdb"


async def test_self_register_preserves_rule_set_url(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Rule-driven URL rotation must win over both the on-disk value and the
    # discovery-layer (CLI/env) value — the envelope is authoritative for
    # server-driven migrations.
    target = _target_path(_isolate)
    target.write_text(
        json.dumps({"remote_config": {"url": "https://old.test/x.json"}})
    )
    envelope = {
        "version": 1,
        "rules": [
            {
                "id": "remote_config.url",
                "op": "set",
                "value": "https://rule.test/x.json",
            }
        ],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(
        _args(remote_config_url="https://cli.test/config.json"), "search"
    )

    data = _read_target(target)
    assert data["remote_config"]["url"] == "https://rule.test/x.json"


async def test_self_register_noop_when_discovery_matches_disk(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Idempotent path: discovery URL equals on-disk URL and no rule touches
    # remote_config → no write, no .bak. Guards against needless mtime and
    # backup churn on every invocation once a URL is registered.
    target = _target_path(_isolate)
    url = "https://remote.test/config.json"
    target.write_text(json.dumps({"remote_config": {"url": url}}))
    bak = target.with_suffix(target.suffix + ".bak")
    original_bytes = target.read_bytes()

    envelope = {"version": 1, "rules": []}
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(remote_config_url=url), "search")

    assert not bak.exists()
    assert target.read_bytes() == original_bytes


async def test_rule_set_url_reverted_when_scheme_invalid(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A rule persisting `http://attacker/x.json` (non-loopback http) would
    # brick the pipeline on the next fetch and — if the attacker chose https —
    # silently retarget the fleet. Same policy the fetcher enforces on the
    # initial URL and every redirect hop; the persist path must match.
    target = _target_path(_isolate)
    original = "https://original.test/x.json"
    target.write_text(json.dumps({"remote_config": {"url": original}}))

    envelope = {
        "version": 1,
        "rules": [
            {
                "id": "remote_config.url",
                "op": "set",
                "value": "http://evil.test/x.json",
            }
        ],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await run_remote_config_fetch(_args(remote_config_url=original), "search")
    finally:
        logger.remove(handler_id)

    data = _read_target(target)
    assert data["remote_config"]["url"] == original
    assert any("remote_config.url" in m for m in messages), (
        f"expected refused-scheme WARNING, got messages={messages}"
    )


async def test_rule_set_url_accepted_when_loopback_http(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Loopback http is inside the fetcher's allowed set — local dev
    # against a control plane on 127.0.0.1 must remain a supported rotation.
    target = _target_path(_isolate)
    target.write_text(
        json.dumps({"remote_config": {"url": "https://original.test/x.json"}})
    )
    loopback = "http://127.0.0.1:8080/config.json"
    envelope = {
        "version": 1,
        "rules": [
            {"id": "remote_config.url", "op": "set", "value": loopback}
        ],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(
        _args(remote_config_url="https://original.test/x.json"), "search"
    )

    data = _read_target(target)
    assert data["remote_config"]["url"] == loopback


async def test_rule_set_unsafe_url_reverted_and_cli_url_seeded_when_no_on_disk_value(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # First-run contract: one successful `--remote-config-url` should make
    # later runs self-sufficient. An unsafe URL rule that arrives in the
    # same envelope must not defeat that — the scheme guard reverts the
    # unsafe rule value, then self-registration still seeds the
    # fetcher-approved discovery URL into the empty slot. Without this
    # ordering (validate → self-register), a hostile or typo'd
    # `remote_config.url` rule in the very first envelope would leave the
    # global file with no URL and the fleet would self-disable.
    envelope = {
        "version": 1,
        "rules": [
            {
                "id": "remote_config.url",
                "op": "set",
                "value": "http://evil.test/x.json",
            }
        ],
    }
    _install_fetch(monkeypatch, envelope)

    cli_url = "https://cli.test/config.json"
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await run_remote_config_fetch(_args(remote_config_url=cli_url), "search")
    finally:
        logger.remove(handler_id)

    target = _target_path(_isolate)
    assert target.exists(), "self-register should have written the seeded URL"
    data = _read_target(target)
    assert data["remote_config"]["url"] == cli_url, (
        f"discovery URL should seed after unsafe rule revert; got {data!r}"
    )
    assert any("remote_config.url" in m for m in messages), (
        f"expected refused-scheme WARNING, got messages={messages}"
    )


async def test_local_json_remote_config_stripped_and_ignored_by_url_discovery(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `.chunkhound.json` in the target dir carrying `remote_config` must
    # (a) never influence the pipeline's step-1 URL discovery (restricted
    # merge skips the local_config layer) and (b) be scrubbed with a
    # WARNING during any full Config load.
    from loguru import logger

    from chunkhound.core.config.config import Config

    project = tmp_path / "project"
    project.mkdir()
    malicious_url = "http://malicious.test/x.json"
    (project / ".chunkhound.json").write_text(
        json.dumps({"remote_config": {"url": malicious_url}})
    )

    benign_url = "https://remote.test/config.json"
    args = _args(command="search", path=project, remote_config_url=benign_url)

    # (a) Pipeline URL discovery must fetch the CLI URL, not the file URL.
    #     envelope=None → pipeline exits after step 2, no disk writes.
    fake = _install_fetch(monkeypatch, None)
    await run_remote_config_fetch(args, "search")
    assert [c[0] for c in fake.calls] == [benign_url], (
        f"expected single fetch to {benign_url!r}; got {fake.calls!r}"
    )

    # (b) Full Config load must strip the subtree and emit the WARNING.
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        cfg = Config(args=args)
    finally:
        logger.remove(handler_id)

    assert cfg.remote_config is not None
    assert cfg.remote_config.url == benign_url
    assert any("remote_config" in m for m in messages), (
        f"expected trust-boundary WARNING, got messages={messages}"
    )


async def test_explicit_config_remote_config_stripped_and_ignored_by_url_discovery(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An explicit `--config` file carrying `remote_config` must
    # (a) never influence the pipeline's step-1 URL discovery (restricted
    # merge skips the config_file layer) and (b) be scrubbed with a
    # WARNING during any full Config load.
    from loguru import logger

    from chunkhound.core.config.config import Config

    # Use an empty project dir so no `.chunkhound.json` sneaks in via
    # find_project_root and confuses the "which layer stripped it" signal.
    project = tmp_path / "empty-project"
    project.mkdir()

    explicit_cfg = tmp_path / "explicit-config.json"
    malicious_url = "http://malicious.test/x.json"
    explicit_cfg.write_text(
        json.dumps({"remote_config": {"url": malicious_url}})
    )

    benign_url = "https://remote.test/config.json"
    args = _args(
        command="search",
        path=project,
        config=str(explicit_cfg),
        remote_config_url=benign_url,
    )

    fake = _install_fetch(monkeypatch, None)
    await run_remote_config_fetch(args, "search")
    assert [c[0] for c in fake.calls] == [benign_url], (
        f"expected single fetch to {benign_url!r}; got {fake.calls!r}"
    )

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        cfg = Config(args=args)
    finally:
        logger.remove(handler_id)

    assert cfg.remote_config is not None
    assert cfg.remote_config.url == benign_url
    assert any("remote_config" in m for m in messages), (
        f"expected trust-boundary WARNING, got messages={messages}"
    )


async def test_rule_failing_sub_model_validation_is_skipped(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Craft a rule that fails EmbeddingConfig validation (invalid provider)
    # alongside a valid rule. The bad rule is dropped; the good rule applies.
    envelope = {
        "version": 1,
        "rules": [
            {
                "id": "embedding.provider",
                "op": "set",
                "value": "definitely-not-a-real-provider",
            },
            {"id": "database.provider", "op": "set", "value": "duckdb"},
        ],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(), "search")

    data = _read_target(_target_path(_isolate))
    assert data.get("database", {}).get("provider") == "duckdb"
    # Bad embedding rule was rolled back — provider not persisted.
    assert data.get("embedding", {}).get("provider") != (
        "definitely-not-a-real-provider"
    )


async def test_leaf_segment_typo_is_skipped_with_warning(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sub-models set ``extra="ignore"`` for local-config forward compatibility,
    # so a leaf typo like ``embedding.provder`` would be silently discarded
    # by pydantic and the audit line would still say "applied". The pipeline
    # rejects unknown leaf segments up front with a WARNING, mirroring the
    # depth-1 unknown-top-level-path check.
    from loguru import logger

    envelope = {
        "version": 1,
        "rules": [
            {"id": "embedding.provder", "op": "set", "value": "openai"},
            {"id": "database.provider", "op": "set", "value": "duckdb"},
        ],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    data = _read_target(_target_path(_isolate))
    # Good rule still applies.
    assert data.get("database", {}).get("provider") == "duckdb"
    # Typo never lands — neither under the misspelling nor the real key.
    assert "provder" not in data.get("embedding", {})
    assert data.get("embedding", {}).get("provider") != "openai"
    # Operator sees a WARNING naming the bad path.
    assert any(
        "schema_error" in m and "embedding.provder" in m for m in messages
    ), f"expected WARNING naming the typo path, got messages={messages}"


async def test_remove_on_leaf_segment_typo_is_schema_error_not_no_op(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Symmetric with test_leaf_segment_typo_is_skipped_with_warning: a
    # ``remove`` targeting a typo path must surface as ``schema_error``, not
    # the ``no-op`` audit line — silence would hide the typo just as it
    # would on ``set``.
    from loguru import logger

    envelope = {
        "version": 1,
        "rules": [{"id": "embedding.provder", "op": "remove"}],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    assert any(
        "schema_error" in m and "embedding.provder" in m for m in messages
    ), f"expected schema_error WARNING on typo remove, got messages={messages}"
    assert not any(
        "no-op" in m and "embedding.provder" in m for m in messages
    ), f"typo remove must not log 'no-op', got messages={messages}"


async def test_parent_merge_with_unknown_nested_key_is_schema_error(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Symmetric with test_leaf_segment_typo_is_skipped_with_warning, but at
    # the parent level: `{"id":"embedding","op":"merge","value":{"provder":...}}`
    # slips past the path-segment check (path has depth 1). Without the
    # payload-key pre-check, extra="ignore" drops `provder` from the
    # sub-model but the raw payload dict is still deep-merged into
    # working_copy and persisted verbatim, and the audit line says
    # "applied" — a silent typo landing in durable config.
    from loguru import logger

    envelope = {
        "version": 1,
        "rules": [
            {"id": "embedding", "op": "merge", "value": {"provder": "openai"}},
            {"id": "database.provider", "op": "set", "value": "duckdb"},
        ],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    data = _read_target(_target_path(_isolate))
    # Good rule still applies.
    assert data.get("database", {}).get("provider") == "duckdb"
    # Typo never lands — neither under the misspelling nor the real key.
    embedding = data.get("embedding", {})
    assert "provder" not in embedding
    assert embedding.get("provider") != "openai"
    # Operator sees a WARNING naming the bad path.
    assert any(
        "schema_error" in m and "embedding.provder" in m for m in messages
    ), f"expected WARNING naming the typo path, got messages={messages}"
    # And crucially, no `applied` line for the embedding rule.
    assert not any(
        "applied" in m and "op=merge" in m and "'embedding'" in m
        for m in messages
    ), f"typo merge must not log 'applied', got messages={messages}"


async def test_parent_set_with_unknown_nested_key_is_schema_error(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same shape as the parent-merge case but with `op: set`, where the
    # raw payload dict wholesale replaces the sub-tree via _apply_set —
    # extra="ignore" hides the typo, persistence writes it verbatim.
    from loguru import logger

    envelope = {
        "version": 1,
        "rules": [
            {"id": "embedding", "op": "set", "value": {"provder": "openai"}},
            {"id": "database.provider", "op": "set", "value": "duckdb"},
        ],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    data = _read_target(_target_path(_isolate))
    assert data.get("database", {}).get("provider") == "duckdb"
    embedding = data.get("embedding", {})
    assert "provder" not in embedding
    assert embedding.get("provider") != "openai"
    assert any(
        "schema_error" in m and "embedding.provder" in m for m in messages
    ), f"expected WARNING naming the typo path, got messages={messages}"
    assert not any(
        "applied" in m and "op=set" in m and "'embedding'" in m
        for m in messages
    ), f"typo set must not log 'applied', got messages={messages}"


async def test_deep_path_with_dict_value_defers_to_sub_model_validation(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Locks in the ``len(segments) == 1`` guard on the payload-key pre-check
    # in ``rules.py`` (see comment at the guard site for the full rationale).
    from loguru import logger

    envelope = {
        "version": 1,
        "rules": [
            {"id": "embedding.api_key", "op": "set", "value": {"nested": "x"}},
            {"id": "database.provider", "op": "set", "value": "duckdb"},
        ],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    data = _read_target(_target_path(_isolate))
    # Good rule still applies (bad rule rolled back cleanly).
    assert data.get("database", {}).get("provider") == "duckdb"
    # Bad rule never landed — either ``embedding`` is absent, or ``api_key``
    # is not a dict.
    assert "embedding" not in data or not isinstance(
        data["embedding"].get("api_key"), dict
    )
    # Operator sees a WARNING from sub-model validation — proves the fix
    # deferred to pydantic instead of the (wrong) parent-payload key check.
    assert any(
        "schema_error" in m and "sub-model validation failed" in m
        for m in messages
    ), (
        "expected schema_error WARNING from sub-model validation, "
        f"got messages={messages}"
    )
    # Regression guard: substring-match on ``unknown path`` + the target path,
    # so cosmetic changes to the log wording can't silently disable the check.
    assert not any(
        "unknown path" in m and "embedding.api_key" in m for m in messages
    ), (
        "regression: parent-payload key check misfired on a depth-2 path — "
        f"got messages={messages}"
    )


async def test_rule_version_gated_is_skipped_rest_applies(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = {
        "version": 1,
        "rules": [
            {
                "id": "database.provider",
                "op": "set",
                "value": "lancedb",
                "min_chunkhound_version": "999.999.999",
            },
            {"id": "mcp.host", "op": "set", "value": "127.0.0.1"},
        ],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(), "search")

    data = _read_target(_target_path(_isolate))
    # Version-gated rule dropped.
    assert data.get("database", {}).get("provider") != "lancedb"
    # Ungated rule applied.
    assert data["mcp"]["host"] == "127.0.0.1"


async def test_rule_unparseable_version_is_skipped_with_warning(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A garbage min_chunkhound_version must not be logged as
    # "requires >= <garbage>"; the pipeline distinguishes unparseable from
    # too-high and emits a dedicated WARNING.
    from loguru import logger

    envelope = {
        "version": 1,
        "rules": [
            {
                "id": "database.provider",
                "op": "set",
                "value": "lancedb",
                "min_chunkhound_version": "not-a-version",
            },
            {"id": "mcp.host", "op": "set", "value": "127.0.0.1"},
        ],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    data = _read_target(_target_path(_isolate))
    assert data.get("database", {}).get("provider") != "lancedb"
    assert data["mcp"]["host"] == "127.0.0.1"
    assert any(
        "unparseable" in m and "not-a-version" in m for m in messages
    ), f"expected unparseable-version WARNING, got messages={messages}"


async def test_envelope_min_version_too_high_discards_all(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = {
        "version": 1,
        "min_chunkhound_version": "999.999.999",
        "rules": [
            {"id": "mcp.host", "op": "set", "value": "127.0.0.1"},
        ],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(), "search")

    assert not _target_path(_isolate).exists(), "no write on version-gate reject"


async def test_backup_osfailure_is_fatal(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target_path(_isolate)
    target.write_text(json.dumps({"mcp": {"transport": "stdio"}}))

    def _raise(*_a, **_kw):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(
        "chunkhound.core.config.remote.persistence.shutil.copy2", _raise
    )

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "127.0.0.1"}],
    }
    _install_fetch(monkeypatch, envelope)

    with pytest.raises(SystemExit) as excinfo:
        await run_remote_config_fetch(_args(), "search")
    assert excinfo.value.code == 1


async def test_persisted_file_is_mode_0600_on_posix(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Global config may embed secrets (embedding.api_key, etc.); persistence
    # chmods to 0o600. atomic_write silently swallows chmod errors, so lock
    # this via an explicit assertion.
    if sys.platform == "win32":
        pytest.skip("mode bits are POSIX-only")

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "127.0.0.1"}],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(), "search")

    target = _target_path(_isolate)
    assert target.exists()
    assert target.stat().st_mode & 0o777 == 0o600


async def test_backup_file_is_mode_0600_on_posix(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ``shutil.copy2`` preserves the source mode, so a pre-existing 0o644
    # config would leave a world-readable ``.bak`` next to the 0o600 primary.
    # Persistence chmods the backup to match; lock that via assertion.
    if sys.platform == "win32":
        pytest.skip("mode bits are POSIX-only")

    target = _target_path(_isolate)
    target.write_text(json.dumps({"mcp": {"transport": "stdio"}}))
    target.chmod(0o644)

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "127.0.0.1"}],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(), "search")

    backup = target.with_suffix(target.suffix + ".bak")
    assert backup.exists()
    assert backup.stat().st_mode & 0o777 == 0o600


async def test_no_write_when_dict_unchanged(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pre-populate the target with a value AND the remote_config already
    # written (so self-registration is a no-op); craft a `set` that lands on
    # the same value → deep-equality passes → no write, no .bak.
    target = _target_path(_isolate)
    url = "https://remote.test/config.json"
    target.write_text(
        json.dumps(
            {"mcp": {"host": "127.0.0.1"}, "remote_config": {"url": url}}
        )
    )
    bak = target.with_suffix(target.suffix + ".bak")

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "127.0.0.1"}],
    }
    _install_fetch(monkeypatch, envelope)

    original_bytes = target.read_bytes()
    await run_remote_config_fetch(_args(remote_config_url=url), "search")

    assert not bak.exists()
    # Content byte-identical (proves neither content nor mtime touched).
    assert target.read_bytes() == original_bytes


async def test_envelope_version_not_1_discards_all(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = {
        "version": 2,
        "rules": [{"id": "mcp.host", "op": "set", "value": "127.0.0.1"}],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(), "search")

    assert not _target_path(_isolate).exists()


async def test_rule_without_op_defaults_to_merge(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `op` defaults to `merge` per the module contract — omitting it must
    # apply the rule, not schema_error out. `merge` on a scalar leaf
    # collapses to `set`, so the value still lands.
    envelope = {
        "version": 1,
        "rules": [{"id": "database.provider", "value": "duckdb"}],
    }
    _install_fetch(monkeypatch, envelope)

    await run_remote_config_fetch(_args(), "search")

    data = _read_target(_target_path(_isolate))
    assert data["database"]["provider"] == "duckdb"


async def test_successful_apply_emits_info_audit_log(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Successful applies must be visible to operators — the module's own
    # docstring rationale ("see what did and didn't take effect") applies
    # to successes too, not only failures. The value is intentionally
    # omitted because paths like `embedding.api_key` carry secrets; op +
    # path is enough for the operator to correlate with the envelope.
    from loguru import logger

    envelope = {
        "version": 1,
        "rules": [
            {"id": "embedding.api_key", "op": "set", "value": "sk-secret"},
        ],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    assert any(
        "applied" in m and "embedding.api_key" in m and "op=set" in m
        for m in messages
    ), f"expected INFO audit line with op + path, got messages={messages}"
    assert not any("sk-secret" in m for m in messages), (
        f"rule value must not appear in logs, got messages={messages}"
    )


async def test_merge_collapsed_to_set_annotated_in_audit_log(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A `merge` rule on a scalar leaf actually calls _apply_set — audit
    # line must reflect that so operators aren't misled into thinking a
    # merge happened when the value was simply replaced.
    from loguru import logger

    envelope = {
        "version": 1,
        "rules": [
            {"id": "database.provider", "op": "merge", "value": "duckdb"},
        ],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    assert any(
        "applied" in m and "database.provider" in m and "collapsed" in m
        for m in messages
    ), f"expected collapse annotation in audit line, got messages={messages}"


async def test_remove_on_missing_path_logs_no_op(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A `remove` rule targeting a path that isn't set must log `no-op`, not
    # `applied` — the audit line drives operators' "did this rule actually
    # do anything?" investigation, so it has to reflect state changes, not
    # just the requested op.
    from loguru import logger

    envelope = {
        "version": 1,
        "rules": [{"id": "database.provider", "op": "remove"}],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    assert any(
        "no-op" in m and "database.provider" in m for m in messages
    ), f"expected no-op INFO line, got messages={messages}"
    assert not any(
        "applied" in m and "database.provider" in m for m in messages
    ), f"remove on missing path must not log 'applied', got messages={messages}"


async def test_rule_setting_refused_path_emits_scrub_warning(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Rule mutating a refused path logs INFO `applied` in apply_rule,
    # then step 4b silently reverts it — audit line would lie. Scrub
    # must emit WARNING naming the key. Value must not leak (refused
    # paths reveal install topology).
    from loguru import logger

    envelope = {
        "version": 1,
        "rules": [
            {"id": "database.path", "op": "set", "value": "/hijacked"},
        ],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    assert any(
        "refused_path" in m and "database.path" in m for m in messages
    ), f"expected refused_path WARNING naming the key, got messages={messages}"
    assert not any("/hijacked" in m for m in messages), (
        f"rule value must not appear in logs, got messages={messages}"
    )
    # No prior on-disk value and scrub deletes → database.path absent.
    target = _target_path(_isolate)
    if target.exists():
        data = _read_target(target)
        assert "path" not in data.get("database", {}), (
            f"database.path must not persist after scrub, got {data}"
        )


async def test_rule_setting_refused_path_over_existing_disk_value_emits_warning(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When on-disk holds an operator value, scrub restores it. Still a
    # rule effect reverted → WARNING fires. Neither value leaks.
    from loguru import logger

    target = _target_path(_isolate)
    target.write_text(json.dumps({"database": {"path": "/original"}}))

    envelope = {
        "version": 1,
        "rules": [
            {"id": "database.path", "op": "set", "value": "/hijacked"},
            # Non-refused rule forces a write — proves scrub reverts the
            # refused change while allowed changes still land on disk.
            {"id": "mcp.host", "op": "set", "value": "127.0.0.1"},
        ],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    assert any(
        "refused_path" in m and "database.path" in m for m in messages
    ), f"expected refused_path WARNING, got messages={messages}"
    assert not any("/hijacked" in m for m in messages), (
        f"rule value must not leak, got messages={messages}"
    )
    assert not any("/original" in m for m in messages), (
        f"on-disk value must not leak, got messages={messages}"
    )
    # Operator's on-disk value wins.
    data = _read_target(target)
    assert data["database"]["path"] == "/original"


async def test_operator_owned_disk_value_does_not_emit_scrub_warning(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No rule touched the refused key → no WARNING. Otherwise the signal
    # turns into background noise on every invocation.
    from loguru import logger

    target = _target_path(_isolate)
    target.write_text(json.dumps({"database": {"path": "/operator-set"}}))

    envelope = {
        "version": 1,
        "rules": [{"id": "mcp.host", "op": "set", "value": "127.0.0.1"}],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    assert not any(
        "refused_path" in m or "database.path" in m for m in messages
    ), (
        "scrub must be silent when no rule touched the refused key, got "
        f"messages={messages}"
    )
    # Operator's on-disk value intact.
    data = _read_target(target)
    assert data["database"]["path"] == "/operator-set"


async def test_parent_merge_sneaking_refused_subkey_emits_scrub_warning(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A rule that names the parent (`database`) and merges a refused
    # sub-key (`path`) is the interesting attack: apply_rule's audit
    # line says "database applied: op=merge" — it never names
    # `database.path` at all. Without the scrub WARNING, an operator
    # grepping for the refused key sees nothing. The WARNING must fire
    # naming the refused key, and disk must not carry the hijack.
    from loguru import logger

    envelope = {
        "version": 1,
        "rules": [
            {
                "id": "database",
                "op": "merge",
                "value": {"path": "/hijacked", "provider": "duckdb"},
            },
        ],
    }
    _install_fetch(monkeypatch, envelope)

    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    assert any(
        "refused_path" in m and "database.path" in m for m in messages
    ), f"expected refused_path WARNING naming the key, got messages={messages}"
    assert not any("/hijacked" in m for m in messages), (
        f"rule value must not leak, got messages={messages}"
    )
    # Non-refused sibling landed; refused sub-key did not.
    data = _read_target(_target_path(_isolate))
    assert data["database"]["provider"] == "duckdb"
    assert "path" not in data["database"], (
        f"refused sub-key must not persist, got {data}"
    )


async def test_write_atomicity_preserves_prior_content_on_mid_write_failure(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If json.dump raises mid-write (ENOSPC, SIGKILL-simulated, etc.), the
    # atomic-write helper must leave the pre-existing target untouched and
    # clean up its .tmp sibling. Without atomicity, `open(target, "w")` would
    # have already truncated the file — corrupting on-disk state and making
    # every subsequent chunkhound invocation fail config load.
    target = _target_path(_isolate)
    original_content = json.dumps(
        {"mcp": {"host": "127.0.0.1"}}, indent=2, sort_keys=True
    ) + "\n"
    target.write_text(original_content)
    original_bytes = target.read_bytes()

    envelope = {
        "version": 1,
        "rules": [{"id": "database.provider", "op": "set", "value": "duckdb"}],
    }
    _install_fetch(monkeypatch, envelope)

    def _raise_enospc(*_a, **_kw):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(
        "chunkhound.utils.atomic_write.json.dump", _raise_enospc
    )

    with pytest.raises(SystemExit) as excinfo:
        await run_remote_config_fetch(_args(), "search")
    assert excinfo.value.code == 1

    # Original file is byte-identical — no truncation, no partial write.
    assert target.read_bytes() == original_bytes
    # No orphan .tmp sibling left behind by the aborted write.
    orphans = [p for p in target.parent.iterdir() if p.suffix == ".tmp"]
    assert orphans == [], f"expected no .tmp orphans, got {orphans}"


async def test_unexpected_bug_in_pipeline_logs_traceback_and_returns(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The outer wrapper in `pipeline.run` exists exclusively to catch bugs
    # in code paths whose *known* failure modes are already handled inside
    # `_run`. When it fires it must (a) not propagate — every `chunkhound`
    # invocation calls this pipeline, so an unhandled exception here would
    # take down unrelated commands — (b) log at ERROR with a full traceback,
    # since without one an in-field report is nearly undebuggable, and
    # (c) leave the on-disk target untouched (no partial write from the
    # aborted pipeline).
    from loguru import logger

    from chunkhound.core.config.remote import pipeline as pipeline_mod

    target = _target_path(_isolate)
    original_content = json.dumps(
        {"mcp": {"host": "127.0.0.1"}}, indent=2, sort_keys=True
    ) + "\n"
    target.write_text(original_content)
    original_bytes = target.read_bytes()

    envelope = {
        "version": 1,
        "rules": [{"id": "database.provider", "op": "set", "value": "duckdb"}],
    }
    _install_fetch(monkeypatch, envelope)

    def _boom(*_a: Any, **_kw: Any) -> None:
        raise AttributeError("simulated rule-engine bug")

    monkeypatch.setattr(pipeline_mod.rules, "apply_rule", _boom)

    messages: list[str] = []
    handler_id = logger.add(
        lambda m: messages.append(str(m)),
        level="ERROR",
        backtrace=False,
        diagnose=False,
    )
    try:
        # Must not raise: the wrapper degrades to a log.
        await run_remote_config_fetch(_args(), "search")
    finally:
        logger.remove(handler_id)

    joined = "\n".join(messages)
    assert "AttributeError" in joined, joined
    assert "pipeline aborted" in joined, joined
    assert "Traceback" in joined, joined
    assert "simulated rule-engine bug" in joined, joined

    # No partial write to the on-disk global config.
    assert target.read_bytes() == original_bytes


async def test_pipeline_skips_fetch_for_internal_subprocess_commands(
    _isolate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `_daemon` and `_quickresearch` are spawned by parents that already ran
    # the pipeline; the child reads the persisted result from disk. Without
    # the skip, `websearch → _quickresearch` and MCP proxy → `_daemon` each
    # pay the 10s fetch tax twice. Gate is `_SUBPROCESS_SKIP` on the pipeline
    # entry — add any new child that inherits its parent's fetched config there.
    fake = _install_fetch(monkeypatch, {"version": 1, "rules": []})

    # Positive control: top-level `search` invocation does fetch.
    await run_remote_config_fetch(_args(command="search"), "search")
    assert len(fake.calls) == 1, (
        f"top-level command must run fetch; got calls={fake.calls}"
    )

    # Internal subprocess commands must not add any calls.
    for command in ("_daemon", "_quickresearch"):
        prior = len(fake.calls)
        await run_remote_config_fetch(_args(command=command), command)
        assert len(fake.calls) == prior, (
            f"{command!r} must skip fetch; got calls={fake.calls}"
        )
