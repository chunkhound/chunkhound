from pathlib import Path
from typing import Any

import pytest

import chunkhound.api.cli.commands.code_mapper as code_mapper_mod
from chunkhound.code_mapper import service as code_mapper_service
from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.core.types.common import Language
from chunkhound.llm_manager import LLMManager
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


def write_scope_repo_layout(repo_root: Path) -> dict[str, Path]:
    scope_dir = repo_root / "scope"
    other_dir = repo_root / "other"
    scope_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "scope/a.py": scope_dir / "a.py",
        "scope/b.py": scope_dir / "b.py",
        "scope/c.py": scope_dir / "c.py",
        "other/d.py": other_dir / "d.py",
    }

    files["scope/a.py"].write_text("def alpha():\n    return 'a'\n", encoding="utf-8")
    files["scope/b.py"].write_text("def beta():\n    return 'b'\n", encoding="utf-8")
    files["scope/c.py"].write_text("def gamma():\n    return 'c'\n", encoding="utf-8")
    files["other/d.py"].write_text("def delta():\n    return 'd'\n", encoding="utf-8")

    return files


async def index_repo(
    provider: Any, repo_root: Path, embedding_provider: FakeEmbeddingProvider
) -> None:
    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(
        provider,
        repo_root,
        embedding_provider,
        {Language.PYTHON: parser},
    )
    await coordinator.process_directory(
        repo_root, patterns=["**/*.py"], exclude_patterns=[]
    )


def build_config(repo_root: Path, provider: str) -> Config:
    return Config(
        target_dir=repo_root,
        database={
            "path": repo_root / ".chunkhound" / "db",
            "provider": provider,
            # Explicitly disable multi-repo mode to ensure per-repo behavior
            # (relative paths) regardless of environment variables.
            "multi_repo": {"enabled": False},
        },
        embedding={
            "provider": "openai",
            "api_key": "test",
            "model": "text-embedding-3-small",
            # Explicitly disable reranking to avoid env var leakage
            # (CHUNKHOUND_EMBEDDING__RERANK_MODEL from env would otherwise apply)
            "rerank_model": None,
        },
        llm={"provider": "openai", "api_key": "test"},
    )


def patch_code_mapper_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    referenced_files: list[str],
    referenced_chunks: list[dict[str, Any]],
) -> None:
    async def fake_overview(
        llm_manager: Any,
        target_dir: Path,
        scope_path: Path,
        scope_label: str,
        max_points: int = 10,
        comprehensiveness: str = "medium",
        out_dir: Path | None = None,
        assembly_provider: Any | None = None,
        indexing_cfg: Any | None = None,
    ) -> tuple[str, list[str]]:
        overview = (
            "1. **Core Flow**: High-level data flow.\n"
            "2. **Error Handling**: How failures are surfaced.\n"
        )
        points = [
            "Core Flow: High-level data flow.",
            "Error Handling: How failures are surfaced.",
        ]
        return overview, points[:max_points]

    async def fake_deep_research_impl(
        services: Any,
        embedding_manager: Any,
        llm_manager: Any,
        query: str,
        progress: Any,
        path: str | None = None,
    ) -> dict[str, Any]:
        return {
            "answer": f"Section for query: {query[:40]}",
            "metadata": {
                "sources": {
                    "files": referenced_files,
                    "chunks": referenced_chunks,
                },
                "aggregation_stats": {},
            },
        }

    monkeypatch.setattr(
        EmbeddingProviderFactory,
        "create_provider",
        lambda _cfg: FakeEmbeddingProvider(batch_size=100),
    )

    def _fake_create_provider(self, provider_config: dict[str, Any]) -> FakeLLMProvider:
        return FakeLLMProvider()

    monkeypatch.setattr(LLMManager, "_create_provider", _fake_create_provider)
    monkeypatch.setattr(
        code_mapper_service,
        "_run_code_mapper_overview_hyde",
        fake_overview,
        raising=True,
    )
    monkeypatch.setattr(
        code_mapper_service, "deep_research_impl", fake_deep_research_impl, raising=True
    )
    monkeypatch.setenv("CH_CODE_MAPPER_WRITE_COMBINED", "1")


def run_code_mapper(
    *,
    scope_path: Path,
    out_dir: Path,
    config: Config,
) -> Any:
    class Args:
        def __init__(self) -> None:
            self.path = scope_path
            self.verbose = False
            self.overview_only = False
            self.out = out_dir
            self.comprehensiveness = "low"
            self.combined = None

    return code_mapper_mod.code_mapper_command(Args(), config)
