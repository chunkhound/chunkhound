from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from chunkhound.code_mapper.orchestrator import CodeMapperOrchestrator
from chunkhound.core.config.config import Config


def test_orchestrator_resolves_dot_to_cwd_within_target_dir(tmp_path: Path, monkeypatch) -> None:
    workspace_root = tmp_path / "workspace"
    project_dir = workspace_root / "chunkhound"
    project_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(project_dir)

    config = Config(target_dir=workspace_root)
    args = SimpleNamespace(path=Path("."))

    orchestrator = CodeMapperOrchestrator(config=config, args=args, llm_manager=None)
    scope = orchestrator.resolve_scope()

    assert scope.target_dir == workspace_root
    assert scope.scope_path == project_dir.resolve()
    assert scope.scope_label == "chunkhound"
