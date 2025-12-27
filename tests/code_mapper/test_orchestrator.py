from pathlib import Path

from chunkhound.code_mapper.orchestrator import CodeMapperOrchestrator
from chunkhound.core.config.config import Config


def test_orchestrator_run_context_max_points() -> None:
    class Args:
        def __init__(self) -> None:
            self.comprehensiveness = "low"
            self.path = "scope"

    config = Config(
        target_dir=Path(".").resolve(),
        database={"path": Path(".") / ".chunkhound" / "db", "provider": "duckdb"},
        embedding={"provider": "openai", "api_key": "test", "model": "test"},
        llm={"provider": "openai", "api_key": "test"},
    )

    orchestrator = CodeMapperOrchestrator(config=config, args=Args(), llm_manager=None)
    run_context = orchestrator.run_context()

    assert run_context.comprehensiveness == "low"
    assert run_context.max_points == 5


def test_orchestrator_resolve_scope_label(tmp_path: Path) -> None:
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
        embedding={"provider": "openai", "api_key": "test", "model": "test"},
        llm={"provider": "openai", "api_key": "test"},
    )

    orchestrator = CodeMapperOrchestrator(config=config, args=Args(), llm_manager=None)
    scope = orchestrator.resolve_scope()

    assert scope.scope_label == "scope"
    assert scope.scope_path == scope_dir.resolve()


def test_orchestrator_metadata_bundle_overview_only(tmp_path: Path) -> None:
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
        embedding={"provider": "openai", "api_key": "test", "model": "test"},
        llm={"provider": "openai", "api_key": "test"},
    )

    orchestrator = CodeMapperOrchestrator(config=config, args=Args(), llm_manager=None)
    bundle = orchestrator.metadata_bundle(
        scope_path=scope_dir.resolve(),
        target_dir=target_dir.resolve(),
        overview_only=True,
    )

    assert bundle.meta.generation_stats.get("overview_only") == "true"
