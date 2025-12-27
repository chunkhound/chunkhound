from pathlib import Path
from typing import Any

import pytest

import chunkhound.api.cli.commands.code_mapper as code_mapper_mod
from chunkhound.core.config.config import Config


class DummyLLMManager:
    """Placeholder LLM manager for overview-only code_mapper tests."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self._configured = True

    def is_configured(self) -> bool:
        return self._configured


@pytest.mark.asyncio
async def test_code_mapper_overview_only_writes_overview_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_root = tmp_path / "repo"
    scope_path = project_root / "scope"
    scope_path.mkdir(parents=True, exist_ok=True)
    (scope_path / "a.py").write_text("print('a')\n", encoding="utf-8")

    out_dir = tmp_path / "out"

    config = Config(
        target_dir=project_root,
        database={"path": project_root / ".chunkhound" / "db", "provider": "duckdb"},
        embedding={
            "provider": "openai",
            "api_key": "test",
            "model": "text-embedding-3-small",
        },
        llm={"provider": "openai", "api_key": "test"},
    )

    async def fake_overview(**_: Any) -> tuple[str, list[str]]:
        return "1. **Core Flow**: Test overview.\n", ["Core Flow: Test overview."]

    monkeypatch.setattr(code_mapper_mod, "LLMManager", DummyLLMManager)
    monkeypatch.setattr(code_mapper_mod, "_run_code_mapper_overview_hyde", fake_overview)

    class Args:
        def __init__(self) -> None:
            self.path = scope_path
            self.verbose = False
            self.overview_only = True
            self.out = out_dir
            self.comprehensiveness = "minimal"
            self.combined = None

    await code_mapper_mod.code_mapper_command(Args(), config)

    _stdout = capsys.readouterr().out
    overview_files = list(out_dir.glob("*_overview.md"))
    assert overview_files, "Expected overview markdown artifact to be written"
    content = overview_files[0].read_text(encoding="utf-8")
    assert "# Code Mapper Overview for scope" in content
    assert "Test overview" in content
