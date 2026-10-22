"""Contract tests for ``Config(skip_layers=...)``.

Locks the public per-layer skip contract before the remote-config pipeline
begins consuming it to construct restricted-merge / half-merged / post-rules
snapshots. Regressions in layer gating would silently corrupt the terminal
delta gate, so the contract lives in tests rather than in comments.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import Field, ValidationError, model_validator
from typing_extensions import Self

from chunkhound.core.config._settings import (
    env_sources_disabled,
    suppressed_env_sources,
)
from chunkhound.core.config.config import Config
from chunkhound.core.config.remote_config import RemoteConfig
from chunkhound.core.config.research_config import ResearchConfig


@pytest.fixture
def proj(tmp_path: Path) -> Path:
    """A concrete project directory. `Config.validate_config` sys.exits when
    target_dir does not exist, so every test needs a real path.
    """
    path = tmp_path / "proj"
    path.mkdir()
    return path


def test_default_skip_layers_none_runs_env_layer(
    proj: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHUNKHOUND_DEBUG", "true")
    config = Config(target_dir=proj)
    assert config.debug is True


def test_skip_env(proj: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNKHOUND_DEBUG", "true")
    config = Config(target_dir=proj, skip_layers={"env"})
    assert config.debug is False


def test_skip_cli(proj: Path) -> None:
    args = SimpleNamespace(debug=True, verbose=False)
    without_skip = Config(args=args, target_dir=proj)
    with_skip = Config(args=args, target_dir=proj, skip_layers={"cli"})
    assert without_skip.debug is True
    assert with_skip.debug is False


def test_skip_local_config(proj: Path) -> None:
    (proj / ".chunkhound.json").write_text(
        json.dumps({"debug": True}), encoding="utf-8"
    )
    without_skip = Config(target_dir=proj)
    with_skip = Config(target_dir=proj, skip_layers={"local_config"})
    assert without_skip.debug is True
    assert with_skip.debug is False


def test_skip_global(hermetic_environment: Path, proj: Path) -> None:
    (hermetic_environment / ".chunkhound.json").write_text(
        json.dumps({"debug": True}), encoding="utf-8"
    )
    without_skip = Config(target_dir=proj)
    with_skip = Config(target_dir=proj, skip_layers={"global"})
    assert without_skip.debug is True
    assert with_skip.debug is False


def test_skip_config_file(tmp_path: Path, proj: Path) -> None:
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"debug": True}), encoding="utf-8")
    args = SimpleNamespace(config=str(explicit), debug=False, verbose=False)
    without_skip = Config(args=args, target_dir=proj)
    with_skip = Config(args=args, target_dir=proj, skip_layers={"config_file"})
    assert without_skip.debug is True
    assert with_skip.debug is False


def test_skip_all_layers_yields_defaults(
    tmp_path: Path,
    proj: Path,
    monkeypatch: pytest.MonkeyPatch,
    hermetic_environment: Path,
) -> None:
    """Every configurable layer set to True, then all layers skipped. The
    resulting Config must equal the model defaults — proves the kwarg gates
    every layer and nothing sneaks through.
    """
    monkeypatch.setenv("CHUNKHOUND_DEBUG", "true")
    (hermetic_environment / ".chunkhound.json").write_text(
        json.dumps({"debug": True}), encoding="utf-8"
    )
    (proj / ".chunkhound.json").write_text(
        json.dumps({"debug": True}), encoding="utf-8"
    )
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"debug": True}), encoding="utf-8")
    args = SimpleNamespace(config=str(explicit), debug=True, verbose=False)

    config = Config(
        args=args,
        target_dir=proj,
        skip_layers={"env", "global", "local_config", "config_file", "cli"},
    )
    assert config.debug is False


def test_target_dir_from_args_path_survives_layer_skip(proj: Path) -> None:
    """``target_dir`` is extracted from ``args.path`` outside the layer gates,
    so skipping ``local_config`` and ``config_file`` (the exact shape the
    remote-config pipeline's URL-discovery step uses) must still route
    ``args.path`` to ``target_dir``. A future refactor that folded target_dir
    extraction into a layer-application helper would silently break that step.
    """
    args = SimpleNamespace(path=str(proj), config=None, debug=False, verbose=False)
    config = Config(args=args, skip_layers={"local_config", "config_file"})
    assert config.target_dir == proj.resolve()


def test_skip_all_layers_still_applies_direct_kwargs(proj: Path) -> None:
    """Direct kwargs bypass the skip gate — they are merged unconditionally
    after every layer, so callers using ``Config(**overrides)`` for tests or
    programmatic construction retain that path even when the layered inputs
    are fully suppressed.
    """
    config = Config(
        target_dir=proj,
        skip_layers={"env", "global", "local_config", "config_file", "cli"},
        debug=True,
    )
    assert config.debug is True


def test_remote_config_dict_coerces_to_model(proj: Path) -> None:
    """A ``remote_config`` kwarg passed as a dict must validate into a
    ``RemoteConfig`` instance. The explicit coercion block was removed in
    favor of pydantic v2 automatic validation; this test locks that contract.
    """
    config = Config(
        target_dir=proj,
        remote_config={
            "url": "https://example.com/config",
            "auth_header": "Bearer ${TOKEN}",
        },
    )
    assert isinstance(config.remote_config, RemoteConfig)
    assert config.remote_config.url == "https://example.com/config"
    assert config.remote_config.auth_header == "Bearer ${TOKEN}"


def test_snapshot_for_persisted_gate_skips_env_and_cli(
    proj: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three gate substrates apply distinct layer selections.

    Persisted = JSON alone (overlays must not leak in — otherwise
    ``--auth-token`` / ``CHUNKHOUND_MCP__AUTH_TOKEN`` mask a newly
    persisted ``mcp.host``).
    Half-merged = env + JSON (CLI / local / --config skipped) — catches
    env-activated hazards that CLI overlays mask on the active side.
    Active = every layer (env + JSON + local + --config + CLI).
    """
    monkeypatch.setenv("CHUNKHOUND_MCP__AUTH_TOKEN", "from-env")
    args = SimpleNamespace(
        path=str(proj),
        config=None,
        debug=False,
        verbose=False,
        transport="http",
        host="1.2.3.4",
        auth_token="from-cli",
    )
    global_dict = {"mcp": {"host": "0.0.0.0"}}

    persisted = Config.snapshot_for_persisted_gate(global_dict)
    assert persisted.mcp.host == "0.0.0.0"
    assert persisted.mcp.auth_token is None
    assert persisted.mcp.transport == "stdio"

    half_merged = Config.snapshot_from_global_dict(global_dict)
    assert half_merged.mcp.host == "0.0.0.0"
    assert half_merged.mcp.auth_token == "from-env"
    assert half_merged.mcp.transport == "stdio"

    active = Config.snapshot_for_delta_gate(global_dict, args)
    assert active.mcp.host == "1.2.3.4"
    assert active.mcp.auth_token == "from-cli"
    assert active.mcp.transport == "http"


def test_snapshot_for_persisted_gate_hides_nested_basesettings_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested BaseSettings must not re-read CHUNKHOUND_* env inside
    ``snapshot_for_persisted_gate`` — otherwise a standing env var masks
    the persist-side delta of a hazard rule and the write commits an
    unusable file that breaks downstream machines without the env.

    Two leak paths, both covered:
    - nested model validation of embedding / LLM dicts that omit ``api_key``;
    - the ``default_factory=ResearchConfig`` / ``FetchUrlConfig`` path
      when the field is absent from ``config_data`` entirely.
    """
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING_API_KEY", "env-key")
    monkeypatch.setenv("CHUNKHOUND_LLM_API_KEY", "env-key")
    monkeypatch.setenv("CHUNKHOUND_RESEARCH_TARGET_TOKENS", "50000")
    monkeypatch.setenv("CHUNKHOUND_FETCHURL_MAX_RETRIES", "9")

    # Partial JSON: api_key omitted. Env must not fill it in.
    snap = Config.snapshot_for_persisted_gate(
        {"embedding": {"provider": "openai"}, "llm": {"provider": "openai"}}
    )
    assert snap.embedding is not None and snap.embedding.api_key is None
    assert snap.llm is not None and snap.llm.api_key is None

    # default_factory path: research absent from dict must yield the
    # model default, not the env override.
    snap = Config.snapshot_for_persisted_gate({})
    assert snap.research.target_tokens == 20000
    assert snap.fetchurl.max_retries == 3
    assert snap.embedding is None
    assert snap.llm is None

    # os.environ restored after the snapshot completes.
    assert os.environ["CHUNKHOUND_EMBEDDING_API_KEY"] == "env-key"
    assert os.environ["CHUNKHOUND_LLM_API_KEY"] == "env-key"
    assert os.environ["CHUNKHOUND_RESEARCH_TARGET_TOKENS"] == "50000"


def test_snapshot_for_persisted_gate_hides_lowercase_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested BaseSettings use ``case_sensitive=False``, so a lowercase
    env name is still honored by pydantic and must also be hidden.
    """
    monkeypatch.setenv("chunkhound_llm_api_key", "env-key")

    snap = Config.snapshot_for_persisted_gate({"llm": {"provider": "openai"}})
    assert snap.llm is not None and snap.llm.api_key is None

    assert os.environ["chunkhound_llm_api_key"] == "env-key"


def test_snapshot_from_global_dict_does_not_mutate_caller_dict(proj: Path) -> None:
    """Snapshot construction must leave the caller's dict untouched — otherwise
    the remote-config pipeline leaks ``indexing.exclude_user_supplied`` into
    the persisted global JSON on its next write.
    """
    original: dict[str, object] = {"indexing": {"exclude": ["*.tmp"]}}
    snapshot_input = {"indexing": {"exclude": ["*.tmp"]}}
    Config.snapshot_from_global_dict(snapshot_input)
    assert snapshot_input == original


# --- nested BaseSettings source-policy regression contracts ---
#
# These lock the env-skip contract for nested BaseSettings. The env layer is
# skipped by suppressing model env sources via the context-local policy
# (config._settings) — os.environ is never mutated, so concurrent/standalone
# readers are always safe.


def test_snapshot_for_persisted_gate_ignores_invalid_env(
    proj: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A standing invalid env var must not break the persisted gate.

    With the env layer skipped the value is never parsed, so defaults apply
    without error. The contrast proves the skip — not a coincidence — is what
    neutralizes it: an env-active build raises on the same invalid value.
    """
    monkeypatch.setenv("CHUNKHOUND_RESEARCH_TARGET_TOKENS", "not-a-number")

    snap = Config.snapshot_for_persisted_gate({})
    assert snap.research.target_tokens == 20_000

    with pytest.raises(ValueError):
        Config(target_dir=proj)


def test_snapshot_for_persisted_gate_ignores_invalid_lowercase_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``case_sensitive=False`` means lowercase env is honored by pydantic;
    under the env-layer skip it must be ignored just like the canonical form,
    and the env var must not be mutated."""
    monkeypatch.setenv("chunkhound_research_target_tokens", "not-a-number")

    snap = Config.snapshot_for_persisted_gate({})
    assert snap.research.target_tokens == 20_000
    assert os.environ["chunkhound_research_target_tokens"] == "not-a-number"


@pytest.mark.parametrize("source", ["dotenv", "secrets"])
def test_snapshot_for_persisted_gate_ignores_dotenv_and_secrets(
    tmp_path: Path, proj: Path, source: str
) -> None:
    """Each file source loads normally but cannot enter a persisted snapshot."""
    path = tmp_path / source
    if source == "dotenv":
        path.write_text("CHUNKHOUND_RESEARCH_TARGET_TOKENS=15000\n")
        options: dict[str, Any] = {"_env_file": str(path)}
    else:
        path.mkdir()
        (path / "CHUNKHOUND_RESEARCH_TARGET_TOKENS").write_text("15000")
        options = {"_secrets_dir": str(path)}
    seeded: dict[str, Any] = {"research": options}

    assert Config(target_dir=proj, **seeded).research.target_tokens == 15_000
    snap = Config.snapshot_for_persisted_gate(seeded)
    assert snap.research.target_tokens == 20_000


def test_snapshot_for_persisted_gate_never_mutates_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source policy never pops env vars from ``os.environ`` — the process
    global stays intact so concurrent and standalone readers always observe
    real values (the property the old pop-and-restore hack violated)."""
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING_API_KEY", "env-key")
    monkeypatch.setenv("CHUNKHOUND_LLM_API_KEY", "env-key")
    monkeypatch.setenv("CHUNKHOUND_RESEARCH_TARGET_TOKENS", "50000")
    monkeypatch.setenv("CHUNKHOUND_FETCHURL_MAX_RETRIES", "9")
    env_before = dict(os.environ)

    Config.snapshot_for_persisted_gate(
        {"embedding": {"provider": "openai"}, "llm": {"provider": "openai"}}
    )

    assert os.environ["CHUNKHOUND_EMBEDDING_API_KEY"] == "env-key"
    assert os.environ["CHUNKHOUND_LLM_API_KEY"] == "env-key"
    assert os.environ["CHUNKHOUND_RESEARCH_TARGET_TOKENS"] == "50000"
    assert os.environ["CHUNKHOUND_FETCHURL_MAX_RETRIES"] == "9"
    assert dict(os.environ) == env_before


def test_snapshot_for_persisted_gate_env_intact_after_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If nested construction raises (e.g. invalid provider), env must remain
    intact — there is no ``os.environ`` pop that could leak or fail to
    restore across the exception."""
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING_API_KEY", "env-key")
    monkeypatch.setenv("CHUNKHOUND_RESEARCH_TARGET_TOKENS", "50000")

    with pytest.raises(ValidationError):
        Config.snapshot_for_persisted_gate({"embedding": {"provider": "bogus"}})

    assert os.environ["CHUNKHOUND_EMBEDDING_API_KEY"] == "env-key"
    assert os.environ["CHUNKHOUND_RESEARCH_TARGET_TOKENS"] == "50000"
    # The failed snapshot must not leave stale suppression in this context.
    assert env_sources_disabled() is False


def test_policy_reset_after_snapshot(
    proj: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The source-policy scope is local to each snapshot: after
    ``snapshot_for_persisted_gate`` returns, a normal same-thread construction
    must read env again (no stale suppression leaking across calls)."""
    monkeypatch.setenv("CHUNKHOUND_RESEARCH_TARGET_TOKENS", "55555")

    # Snapshot suppresses env for its nested construction only.
    snap = Config.snapshot_for_persisted_gate({})
    assert snap.research.target_tokens == 20_000

    # Same thread, afterwards: env sources are active again.
    assert Config(target_dir=proj).research.target_tokens == 55555


def test_normal_construction_preserves_nested_env_loading(
    proj: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Standalone env loading for nested BaseSettings is preserved when the
    env layer is NOT skipped — the fix must not regress the normal contract."""
    monkeypatch.setenv("CHUNKHOUND_RESEARCH_TARGET_TOKENS", "55555")

    cfg = Config(target_dir=proj)
    assert cfg.research.target_tokens == 55555


@pytest.mark.parametrize(
    ("section", "field", "default", "env_value", "explicit"),
    [
        ("embedding", "max_retries", 3, 8, 5),
        ("llm", "max_retries", 3, 8, 5),
        ("research", "target_tokens", 20_000, 50_000, 30_000),
        ("fetchurl", "max_retries", 3, 8, 5),
    ],
)
def test_nested_settings_respect_env_skip(
    proj: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    default: int,
    env_value: int,
    explicit: int,
) -> None:
    """Skip affects nested sources, not explicit values or later Config loads."""
    monkeypatch.setenv(f"CHUNKHOUND_{section}_{field}".upper(), str(env_value))
    # Optional settings need input; required settings exercise default factories.
    seed: dict[str, Any] = {section: {}} if section in {"embedding", "llm"} else {}

    def value(**kwargs: Any) -> int:
        config = Config(target_dir=proj, **kwargs)
        return cast(int, getattr(getattr(config, section), field))

    assert value(**seed) == env_value
    assert value(skip_layers={"env"}, **seed) == default
    assert value(skip_layers={"env"}, **{section: {field: explicit}}) == explicit
    assert value(**{section: {field: explicit}}) == explicit
    assert value(**seed) == env_value


def test_concurrent_standalone_reader_unaffected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hold real snapshot validation open until a standalone env reader finishes."""
    monkeypatch.setenv("CHUNKHOUND_RESEARCH_TARGET_TOKENS", "99999")
    barrier = threading.Barrier(2, timeout=10)

    class BlockingResearch(ResearchConfig):
        @model_validator(mode="after")
        def hold_snapshot_scope(self) -> Self:
            barrier.wait()
            barrier.wait()
            return self

    class BlockingSnapshot(Config):
        research: BlockingResearch = Field(default_factory=BlockingResearch)

    def reader() -> int:
        barrier.wait()
        try:
            return ResearchConfig().target_tokens
        finally:
            barrier.wait()

    with ThreadPoolExecutor(max_workers=1) as executor:
        standalone = executor.submit(reader)
        snapshot = BlockingSnapshot.snapshot_for_persisted_gate({})
        assert standalone.result() == 99999

    assert snapshot.research.target_tokens == 20_000


def test_global_config_file_selector_gated_by_global_skip(
    tmp_path: Path, proj: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``CHUNKHOUND_GLOBAL_CONFIG_FILE`` selects the global-JSON layer; the
    ``global`` skip must gate its contents (the env selector is distinct from
    the env-value layer and is controlled by its own gate)."""
    global_file = tmp_path / "global.json"
    global_file.write_text(json.dumps({"debug": True}), encoding="utf-8")
    monkeypatch.setenv("CHUNKHOUND_GLOBAL_CONFIG_FILE", str(global_file))

    assert Config(target_dir=proj).debug is True
    assert Config(target_dir=proj, skip_layers={"global"}).debug is False


def test_config_file_selector_gated_by_config_file_skip(
    tmp_path: Path, proj: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``CHUNKHOUND_CONFIG_FILE`` selects the explicit-config layer; the
    ``config_file`` skip must gate its contents."""
    cfg_file = tmp_path / "explicit.json"
    cfg_file.write_text(json.dumps({"debug": True}), encoding="utf-8")
    monkeypatch.setenv("CHUNKHOUND_CONFIG_FILE", str(cfg_file))

    assert Config(target_dir=proj).debug is True
    assert Config(target_dir=proj, skip_layers={"config_file"}).debug is False


def test_asyncio_task_created_inside_scope_inherits_suppression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``asyncio.create_task`` copies the current context, so a task created
    inside ``suppressed_env_sources`` inherits the suppression."""
    monkeypatch.setenv("CHUNKHOUND_RESEARCH_TARGET_TOKENS", "99999")

    async def build_research() -> int:
        return ResearchConfig().target_tokens

    async def runner() -> int:
        with suppressed_env_sources():
            task = asyncio.create_task(build_research())
            return await task

    inherited = asyncio.run(runner())

    assert inherited == 20_000
    assert env_sources_disabled() is False
    assert ResearchConfig().target_tokens == 99999


def test_thread_started_inside_scope_gets_fresh_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New threads never inherit suppression; they read live env."""
    monkeypatch.setenv("CHUNKHOUND_RESEARCH_TARGET_TOKENS", "99999")
    seen: list[int] = []
    with suppressed_env_sources():
        worker = threading.Thread(
            target=lambda: seen.append(ResearchConfig().target_tokens)
        )
        worker.start()
        worker.join()
        assert ResearchConfig().target_tokens == 20_000
    assert seen == [99999]
    assert env_sources_disabled() is False


def test_invalid_llm_env_ignored_under_suppression(
    proj: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid LLM env breaks normal builds, not persisted snapshots."""
    monkeypatch.setenv("CHUNKHOUND_LLM_OUTPUT_LIMIT_FALLBACK", "not-a-number")
    snap = Config.snapshot_for_persisted_gate({})
    assert snap.llm is None
    with pytest.raises(ValueError):
        Config(target_dir=proj)


# --- BaseSettings registration guard ---
#
# Locks the pop-and-restore removal: every BaseSettings under
# chunkhound/core/config/ must extend _ChunkhoundBaseSettings (the
# context-local source policy), and no config module may mutate
# os.environ (getenv reads only). AST-based so comments/docstrings
# mentioning os.environ never trip the guard.


def _settings_leaks() -> list[str]:
    root = Path(__file__).resolve().parents[2] / "chunkhound" / "core" / "config"
    leaks = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name == "_ChunkhoundBaseSettings":
                continue
            bases = {ast.unparse(b).split(".")[-1] for b in node.bases}
            if "BaseSettings" in bases and "_ChunkhoundBaseSettings" not in bases:
                leaks.append(f"{path.name}:{node.name}")
    return leaks


_ENV_MUTATING_CALLS = frozenset({"pop", "clear", "update", "setdefault"})
_ENV_MUTATING_MODULE_CALLS = frozenset({"putenv", "unsetenv"})
_ENV_MAPPINGS = frozenset({"environ", "environb"})


def _environ_mutations() -> list[str]:
    root = Path(__file__).resolve().parents[2] / "chunkhound" / "core" / "config"
    hits = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                recv = node.func.value
                if node.func.attr in _ENV_MUTATING_MODULE_CALLS:
                    hits.append(f"{path.name}:{node.lineno}")
                is_env = isinstance(recv, ast.Attribute) and recv.attr in _ENV_MAPPINGS
                if is_env and node.func.attr in _ENV_MUTATING_CALLS:
                    hits.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.Subscript) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                recv = node.value
                if isinstance(recv, ast.Attribute) and recv.attr in _ENV_MAPPINGS:
                    hits.append(f"{path.name}:{node.lineno}")
    return hits


def test_all_basesettings_extend_chunkhound_base() -> None:
    assert _settings_leaks() == []


def test_no_environ_mutation_in_config() -> None:
    assert _environ_mutations() == []
