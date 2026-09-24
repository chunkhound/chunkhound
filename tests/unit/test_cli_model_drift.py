"""Contract tests for how ``chunkhound index`` handles embedding drift.

Driven through ``run_command`` with the registry and coordinator stubbed, the
same way ``test_cli_timeout_prompt.py`` does. What matters is what the operator
is asked, what they are told, and which decision reaches the registry.
"""

import io
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chunkhound.api.cli.commands import autodoc_prompts
from chunkhound.api.cli.commands import run as run_mod
from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.embedding_model_drift import IndexedEmbeddingModel, ModelDrift

pytestmark = pytest.mark.asyncio

_INDEXED = IndexedEmbeddingModel(
    provider="voyageai", model="voyage-code-3", dims=1024, embedding_count=12431
)
_MODEL_SWITCH = ModelDrift(
    indexed=_INDEXED,
    configured_provider="voyageai",
    configured_model="voyage-code-4",
    configured_dims=2048,
)
_DIMS_ONLY = ModelDrift(
    indexed=_INDEXED,
    configured_provider="voyageai",
    configured_model="voyage-code-3",
    configured_dims=512,
)
_HINT = "voyage-code-4 supersedes voyage-code-3"


def _coordinator() -> AsyncMock:
    coord = AsyncMock()
    coord._db = SimpleNamespace(
        drop_all_hnsw_indexes=lambda: None,
        ensure_all_hnsw_indexes=lambda: None,
    )
    coord.get_stats.return_value = {"files": 0, "chunks": 0, "embeddings": 0}
    coord.process_directory.return_value = {
        "status": "success",
        "files_processed": 0,
        "total_chunks": 0,
        "skipped": 0,
        "skipped_due_to_timeout": [],
    }
    coord.compact_database_with_metrics.return_value = {
        "status": "skipped",
        "reason": "unsupported",
    }
    return coord


class _Run:
    """What one ``chunkhound index`` invocation did."""

    hook: Any = "never configured"
    decision: bool | None = None
    output: str = ""


async def _index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    drift: ModelDrift | None = None,
    interactive: bool = False,
    answer: str | None = None,
    env: dict[str, str] | None = None,
    embedding: tuple[str, str] = ("voyageai", "voyage-code-3"),
) -> _Run:
    """Run ``run_command`` once, feeding ``drift`` to whatever hook it builds."""
    run = _Run()

    def configure_registry(config: Config, on_model_drift: Any = None) -> None:
        run.hook = on_model_drift
        if drift is not None and on_model_drift is not None:
            run.decision = on_model_drift(drift)

    provider_name, model = embedding
    registry = SimpleNamespace(
        get_provider=lambda name: SimpleNamespace(name=provider_name, model=model)
    )

    def read_answer(prompt: str = "") -> str:
        if answer is None:
            raise AssertionError(f"operator was prompted: {prompt!r}")
        return answer

    monkeypatch.setattr(run_mod, "configure_registry", configure_registry)
    monkeypatch.setattr(run_mod, "create_indexing_coordinator", _coordinator)
    monkeypatch.setattr(run_mod, "get_registry", lambda: registry)
    monkeypatch.setattr(autodoc_prompts, "is_interactive", lambda: interactive)
    monkeypatch.setattr("builtins.input", read_answer)
    for key in (
        "CHUNKHOUND_MCP_MODE",
        "CHUNKHOUND_NO_PROMPTS",
        "CHUNKHOUND_NO_MODEL_SUGGESTIONS",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)

    args = Namespace(
        path=tmp_path,
        verbose=False,
        no_embeddings=True,
        include=None,
        exclude=None,
        db=None,
    )
    config = Config(target_dir=tmp_path)
    config.database = DatabaseConfig(
        provider="duckdb", path=tmp_path / ".chunkhound" / "db"
    )

    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    await run_mod.run_command(args, config)
    run.output = buffer.getvalue()
    return run


# ---------------------------------------------------------------------------
# Drift prompt
# ---------------------------------------------------------------------------


async def test_interactive_accept_switches_model(tmp_path, monkeypatch):
    run = await _index(
        tmp_path, monkeypatch, drift=_MODEL_SWITCH, interactive=True, answer="y"
    )

    assert run.decision is True
    assert "voyage-code-3" in run.output
    assert "voyage-code-4" in run.output


async def test_interactive_decline_keeps_indexed_model(tmp_path, monkeypatch):
    run = await _index(
        tmp_path, monkeypatch, drift=_MODEL_SWITCH, interactive=True, answer="n"
    )

    assert run.decision is False
    assert (
        "Set embedding.model to voyage-code-3 and embedding.output_dims to 1024"
        in run.output
    )


async def test_no_prompts_env_keeps_indexed_model(tmp_path, monkeypatch):
    run = await _index(
        tmp_path,
        monkeypatch,
        drift=_MODEL_SWITCH,
        interactive=True,
        env={"CHUNKHOUND_NO_PROMPTS": "1"},
    )

    assert run.decision is False
    assert "prompts disabled" in run.output


async def test_non_interactive_run_keeps_indexed_model_without_prompting(
    tmp_path, monkeypatch
):
    run = await _index(tmp_path, monkeypatch, drift=_MODEL_SWITCH)

    assert run.decision is False
    assert "no interactive terminal" in run.output


async def test_dims_only_change_is_never_offered(tmp_path, monkeypatch):
    """It cannot be re-embedded in place, so asking would be a false promise."""
    run = await _index(tmp_path, monkeypatch, drift=_DIMS_ONLY, interactive=True)

    assert run.decision is False
    assert "delete the database directory" in run.output


async def test_mcp_mode_supplies_no_drift_hook(tmp_path, monkeypatch):
    """MCP must never prompt, so the registry gets nothing to consult."""
    run = await _index(tmp_path, monkeypatch, env={"CHUNKHOUND_MCP_MODE": "1"})

    assert run.hook is None


# ---------------------------------------------------------------------------
# Upgrade hint
# ---------------------------------------------------------------------------


async def test_upgrade_hint_names_the_successor(tmp_path, monkeypatch):
    run = await _index(tmp_path, monkeypatch, interactive=True)

    assert _HINT in run.output


async def test_upgrade_hint_is_silenced_by_env(tmp_path, monkeypatch):
    run = await _index(
        tmp_path,
        monkeypatch,
        interactive=True,
        env={"CHUNKHOUND_NO_MODEL_SUGGESTIONS": "1"},
    )

    assert "supersedes" not in run.output


async def test_upgrade_hint_is_not_shown_to_non_interactive_runs(tmp_path, monkeypatch):
    run = await _index(tmp_path, monkeypatch)

    assert "supersedes" not in run.output


async def test_upgrade_hint_only_matches_its_own_provider(tmp_path, monkeypatch):
    run = await _index(
        tmp_path,
        monkeypatch,
        interactive=True,
        embedding=("openai", "voyage-code-3"),
    )

    assert "supersedes" not in run.output


async def test_upgrade_hint_is_suppressed_when_drift_was_reported(
    tmp_path, monkeypatch
):
    run = await _index(
        tmp_path, monkeypatch, drift=_MODEL_SWITCH, interactive=True, answer="n"
    )

    assert "supersedes" not in run.output
