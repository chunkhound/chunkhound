from pathlib import Path
from typing import Any

import pytest

import chunkhound.api.cli.commands.autodoc as autodoc_mod
from chunkhound.autodoc import pipeline as autodoc_pipeline
from chunkhound.core.config.config import Config


@pytest.mark.asyncio
async def test_autodoc_overview_only_uses_config_dir_as_root_and_sets_default_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    cfg_path = workspace_root / ".chunkhound.json"
    cfg_path.write_text("{}", encoding="utf-8")

    # Start with a config rooted elsewhere; autodoc should treat the config file
    # directory as the workspace root and (when the config file omits database.path)
    # fall back to <workspace>/.chunkhound/db.
    other_root = tmp_path / "other"
    other_root.mkdir(parents=True, exist_ok=True)
    config = Config(
        target_dir=other_root,
        database={"path": other_root / ".chunkhound" / "db", "provider": "duckdb"},
        embedding={
            "provider": "openai",
            "api_key": "test",
            "model": "text-embedding-3-small",
        },
        llm={"provider": "openai", "api_key": "test"},
    )

    async def fake_overview(*_: Any, **__: Any) -> tuple[str, list[str]]:
        return "1. Example\n", ["Example"]

    monkeypatch.setattr(autodoc_pipeline, "_run_autodoc_overview_hyde", fake_overview)
    monkeypatch.setattr(autodoc_mod, "verify_database_exists", lambda *_: (_ for _ in ()).throw(AssertionError("verify_database_exists should not run in overview-only")))

    class Args:
        def __init__(self) -> None:
            self.path = Path("scope")
            self.verbose = False
            self.overview_only = True
            self.out_dir = tmp_path / "out"
            self.comprehensiveness = "low"
            self.config = cfg_path
            self.db = None
            self.database_path = None

    await autodoc_mod.autodoc_command(Args(), config)

    assert config.target_dir == workspace_root
    assert config.database.path == workspace_root / ".chunkhound" / "db"


@pytest.mark.asyncio
async def test_autodoc_does_not_override_explicit_db_path_from_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    cfg_path = workspace_root / ".chunkhound.json"
    cfg_path.write_text('{"database": {"path": "/explicit/db"}}', encoding="utf-8")

    config = Config(
        target_dir=tmp_path / "other",
        database={"path": tmp_path / "explicit_db", "provider": "duckdb"},
        embedding={
            "provider": "openai",
            "api_key": "test",
            "model": "text-embedding-3-small",
        },
        llm={"provider": "openai", "api_key": "test"},
    )
    original_db_path = config.database.path

    async def fake_overview(*_: Any, **__: Any) -> tuple[str, list[str]]:
        return "1. Example\n", ["Example"]

    monkeypatch.setattr(autodoc_pipeline, "_run_autodoc_overview_hyde", fake_overview)
    monkeypatch.setattr(autodoc_mod, "verify_database_exists", lambda *_: (_ for _ in ()).throw(AssertionError("verify_database_exists should not run in overview-only")))

    class Args:
        def __init__(self) -> None:
            self.path = Path("scope")
            self.verbose = False
            self.overview_only = True
            self.out_dir = tmp_path / "out"
            self.comprehensiveness = "low"
            self.config = cfg_path
            self.db = None
            self.database_path = None

    await autodoc_mod.autodoc_command(Args(), config)

    assert config.target_dir == workspace_root
    assert config.database.path == original_db_path
