import asyncio
from pathlib import Path
from typing import Any

import pytest

from chunkhound.embeddings import EmbeddingManager
import operations.deep_doc.deep_doc as deep_doc_mod
from operations.deep_doc.deep_doc import (
    AgentDocMetadata,
    HydeConfig,
)


class DummyProvider:
    """Minimal provider stub exposing chunks for coverage stats."""

    def __init__(self) -> None:
        # Three files under the scoped folder `scope/`
        self._chunks = [
            {"file_path": "scope/a.py", "start_line": 1, "end_line": 10},
            {"file_path": "scope/b.py", "start_line": 1, "end_line": 20},
            {"file_path": "scope/c.py", "start_line": 5, "end_line": 15},
        ]

    def get_all_chunks_with_metadata(self) -> list[dict[str, Any]]:
        return list(self._chunks)


class DummyServices:
    """Services stub that only exposes a provider for coverage stats."""

    def __init__(self) -> None:
        self.provider = DummyProvider()


@pytest.mark.asyncio
async def test_hyde_map_does_not_reduce_source_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HyDE map passes must not reduce unified sources or coverage.

    This guards against regressions where enabling CH_AGENT_DOC_CODE_RESEARCH_HYDE_MAP
    accidentally drops files from unified_source_files or reduces coverage stats,
    even though additional deep-research passes are performed.
    """

    project_root = tmp_path / "repo"
    scope_path = project_root / "scope"
    scope_path.mkdir(parents=True, exist_ok=True)
    # Create a dummy file so the scope is non-empty on disk.
    (scope_path / "dummy.txt").write_text("x\n", encoding="utf-8")

    meta = AgentDocMetadata(
        created_from_sha="TEST_SHA",
        previous_target_sha="TEST_SHA",
        target_sha="TEST_SHA",
        generated_at="2025-01-01T00:00:00Z",
        llm_config={},
        generation_stats={},
    )
    hyde_cfg = HydeConfig.from_env()
    services = DummyServices()
    embedding_manager = EmbeddingManager()
    llm_manager = None  # Not used by the stubs below

    async def fake_run_research_with_metadata(
        services: Any,
        embedding_manager: EmbeddingManager,
        llm_manager: Any,
        prompt: str,
        scope_label: str | None = None,
        path_override: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        # Single overview-style call that references one file.
        return (
            "OVERVIEW",
            {
                "sources": {
                    "files": ["scope/a.py"],
                    "chunks": [
                        {"file_path": "scope/a.py", "start_line": 1, "end_line": 10}
                    ],
                }
            },
        )

    async def fake_hyde_map_deep_research(
        services: Any,
        embedding_manager: EmbeddingManager | None,
        llm_manager: Any,
        scope_label: str,
        hyde_plan: str | None,
    ) -> tuple[str, list[dict[str, Any]], int]:
        # HyDE-guided passes that add two more files worth of sources.
        section = "## 7. HyDE-Guided Deep Dives\n\nBody"
        meta_list = [
            {
                "sources": {
                    "files": ["scope/b.py"],
                    "chunks": [
                        {"file_path": "scope/b.py", "start_line": 1, "end_line": 20}
                    ],
                }
            },
            {
                "sources": {
                    "files": ["scope/c.py"],
                    "chunks": [
                        {"file_path": "scope/c.py", "start_line": 5, "end_line": 15}
                    ],
                }
            },
        ]
        return section, meta_list, 2

    monkeypatch.setattr(
        deep_doc_mod,
        "_run_research_query_with_metadata",
        fake_run_research_with_metadata,
        raising=True,
    )
    monkeypatch.setattr(
        deep_doc_mod,
        "_run_hyde_map_deep_research",
        fake_hyde_map_deep_research,
        raising=True,
    )

    scope_label = "scope"

    # First run: HyDE map disabled (baseline).
    monkeypatch.delenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_MAP", raising=False)
    (
        _body_no_map,
        files_no_map,
        chunks_no_map,
        calls_no_map,
        enable_hyde_map_no,
        map_only_no,
    ) = await deep_doc_mod._run_deep_doc_body_pipeline(
        project_root=project_root,
        scope_path=scope_path,
        scope_label=scope_label,
        hyde_only=False,
        generator_mode="code_research",
        structure_mode="fluid",
        services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        assembly_provider=None,
        meta=meta,
        hyde_cfg=hyde_cfg,
        diff_since_created=[],
        diff_since_previous=[],
        hyde_plan="TEST_PLAN",
        out_dir=None,
    )

    assert not enable_hyde_map_no
    assert not map_only_no
    assert files_no_map and "scope/a.py" in files_no_map
    assert len(files_no_map) == 1
    assert len(chunks_no_map) == 1

    stats_no_map = deep_doc_mod._build_generation_stats(
        generator_mode="code_research",
        hyde_map_enabled=enable_hyde_map_no,
        code_research_map_only=map_only_no,
        structure_mode="fluid",
        hyde_only=False,
        total_research_calls=calls_no_map,
        unified_source_files=files_no_map,
        unified_chunks_dedup=chunks_no_map,
        services=services,
        scope_label=scope_label,
    )

    # Second run: HyDE map enabled, which should only add more sources.
    monkeypatch.setenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_MAP", "1")
    (
        _body_map,
        files_map,
        chunks_map,
        calls_map,
        enable_hyde_map_yes,
        map_only_yes,
    ) = await deep_doc_mod._run_deep_doc_body_pipeline(
        project_root=project_root,
        scope_path=scope_path,
        scope_label=scope_label,
        hyde_only=False,
        generator_mode="code_research",
        structure_mode="fluid",
        services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        assembly_provider=None,
        meta=meta,
        hyde_cfg=hyde_cfg,
        diff_since_created=[],
        diff_since_previous=[],
        hyde_plan="TEST_PLAN",
        out_dir=None,
    )

    assert enable_hyde_map_yes
    assert not map_only_yes

    # HyDE map should preserve existing sources and add more.
    assert set(files_no_map.keys()).issubset(set(files_map.keys()))
    assert len(files_map) >= len(files_no_map)
    assert len(chunks_map) >= len(chunks_no_map)

    stats_map = deep_doc_mod._build_generation_stats(
        generator_mode="code_research",
        hyde_map_enabled=enable_hyde_map_yes,
        code_research_map_only=map_only_yes,
        structure_mode="fluid",
        hyde_only=False,
        total_research_calls=calls_map,
        unified_source_files=files_map,
        unified_chunks_dedup=chunks_map,
        services=services,
        scope_label=scope_label,
    )

    # Coverage stats should not regress when HyDE map is enabled.
    cov_no = float(stats_no_map.get("scope_coverage_percent_indexed", "0") or "0")
    cov_map = float(stats_map.get("scope_coverage_percent_indexed", "0") or "0")
    assert cov_map >= cov_no
