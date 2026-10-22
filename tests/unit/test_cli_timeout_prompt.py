import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from chunkhound.api.cli.commands import run as run_mod
from chunkhound.core.config.config import Config


@pytest.fixture
def timeout_coordinator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    coord = AsyncMock()
    coord.resolve_rust_pipeline_decision = Mock(return_value=False)
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
        # Real indexing returns absolute paths, independent of the caller cwd.
        "skipped_due_to_timeout": [str(tmp_path / "big.bin")],
    }
    coord.compact_database_with_metrics.return_value = {"status": "skipped"}
    monkeypatch.setattr(run_mod, "configure_registry", lambda *a, **kw: None)
    monkeypatch.setattr(run_mod, "create_indexing_coordinator", lambda: coord)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)


async def run_timeout_prompt(project: Path) -> None:
    args = Namespace(
        path=project,
        verbose=False,
        no_embeddings=True,
        include=None,
        exclude=None,
        db=None,
    )
    config = Config(
        target_dir=project,
        database={"provider": "duckdb", "path": project / ".chunkhound" / "db"},
    )
    await run_mod.run_command(args, config)


@pytest.mark.asyncio
async def test_timeout_prompt_adds_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout_coordinator, capsys
):
    config_path = tmp_path / ".chunkhound.json"
    config_path.write_text("{}\n")
    monkeypatch.delenv("CHUNKHOUND_MCP_MODE", raising=False)
    monkeypatch.delenv("CHUNKHOUND_NO_PROMPTS", raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "y")

    await run_timeout_prompt(tmp_path)

    assert "Skipped Due to Timeout" in capsys.readouterr().out
    assert "big.bin" in json.loads(config_path.read_text())["indexing"]["exclude"]


def reject_input(*_):
    raise AssertionError("MCP mode must not request interactive input")


@pytest.mark.asyncio
async def test_timeout_prompt_skipped_in_mcp_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout_coordinator, capsys
):
    """CHUNKHOUND_MCP_MODE=1 must prevent input() and config writes."""
    monkeypatch.setenv("CHUNKHOUND_MCP_MODE", "1")
    monkeypatch.setattr("builtins.input", reject_input)

    await run_timeout_prompt(tmp_path)

    out = capsys.readouterr().out
    assert "files timed out" in out or "prompts disabled" in out
    assert not (tmp_path / ".chunkhound.json").exists()
