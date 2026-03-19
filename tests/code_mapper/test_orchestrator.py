import subprocess
from pathlib import Path

import chunkhound.code_mapper.orchestrator as orchestrator_mod
from chunkhound.code_mapper.orchestrator import CodeMapperOrchestrator
from chunkhound.core.config.config import Config


def test_orchestrator_run_context_max_points(tmp_path: Path, clean_environment) -> None:
    class Args:
        def __init__(self) -> None:
            self.comprehensiveness = "low"
            self.path = "scope"

    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / ".chunkhound" / "db", "provider": "duckdb"},
        embedding={"provider": "openai", "api_key": "test", "model": "text-embedding-3-small"},
        llm={"provider": "openai", "api_key": "test"},
    )

    orchestrator = CodeMapperOrchestrator(config=config, args=Args(), llm_manager=None)
    run_context = orchestrator.run_context()

    assert run_context.comprehensiveness == "low"
    assert run_context.max_points == 5


def test_orchestrator_resolve_scope_label(tmp_path: Path, clean_environment) -> None:
    class Args:
        def __init__(self) -> None:
            self.comprehensiveness = "low"
            self.path = "scope"

    target_dir = tmp_path / "repo"
    scope_dir = target_dir / "scope"
    scope_dir.mkdir(parents=True)

    config = Config(
        target_dir=target_dir,
        database={"path": target_dir / ".chunkhound" / "db", "provider": "duckdb"},
        embedding={"provider": "openai", "api_key": "test", "model": "text-embedding-3-small"},
        llm={"provider": "openai", "api_key": "test"},
    )

    orchestrator = CodeMapperOrchestrator(config=config, args=Args(), llm_manager=None)
    scope = orchestrator.resolve_scope()

    assert scope.scope_label == "scope"
    assert scope.scope_path == scope_dir.resolve()


def test_orchestrator_metadata_bundle_overview_only(
    tmp_path: Path, clean_environment
) -> None:
    class Args:
        def __init__(self) -> None:
            self.comprehensiveness = "low"
            self.path = "scope"

    target_dir = tmp_path / "repo"
    scope_dir = target_dir / "scope"
    scope_dir.mkdir(parents=True)

    config = Config(
        target_dir=target_dir,
        database={"path": target_dir / ".chunkhound" / "db", "provider": "duckdb"},
        embedding={"provider": "openai", "api_key": "test", "model": "text-embedding-3-small"},
        llm={"provider": "openai", "api_key": "test"},
    )

    orchestrator = CodeMapperOrchestrator(config=config, args=Args(), llm_manager=None)
    bundle = orchestrator.metadata_bundle(
        scope_path=scope_dir.resolve(),
        target_dir=target_dir.resolve(),
        overview_only=True,
    )

    assert bundle.meta.generation_stats.get("overview_only") == "true"


def test_get_head_sha_uses_git_safe_and_returns_sha(
    tmp_path: Path, monkeypatch
) -> None:
    def explode_subprocess_run(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called by _get_head_sha")

    monkeypatch.setattr(subprocess, "run", explode_subprocess_run, raising=True)

    def fake_run_git(*_args, **_kwargs) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            ["git", "rev-parse", "HEAD"], 0, stdout="abc123\n", stderr=""
        )

    monkeypatch.setattr(orchestrator_mod, "run_git", fake_run_git, raising=False)

    assert orchestrator_mod._get_head_sha(tmp_path) == "abc123"


def test_get_head_sha_returns_placeholder_when_git_fails(
    tmp_path: Path, monkeypatch
) -> None:
    def explode_subprocess_run(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called by _get_head_sha")

    monkeypatch.setattr(subprocess, "run", explode_subprocess_run, raising=True)

    def fake_run_git(*_args, **_kwargs) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            ["git", "rev-parse", "HEAD"], 1, stdout="", stderr="fatal"
        )

    monkeypatch.setattr(orchestrator_mod, "run_git", fake_run_git, raising=False)

    assert orchestrator_mod._get_head_sha(tmp_path) == "NO_GIT_HEAD"
