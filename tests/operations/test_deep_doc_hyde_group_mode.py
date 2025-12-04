import asyncio
from pathlib import Path
from typing import Any

import pytest

from chunkhound.embeddings import EmbeddingManager
import operations.deep_doc.deep_doc as deep_doc_mod


class DummyServicesForMap:
    """Minimal services stub for HyDE map tests."""

    def __init__(self) -> None:
        # Provider is not used directly by _run_hyde_map_passes, but some
        # callers expect services.provider to exist, so keep it minimal.
        class _P:
            def get_all_chunks_with_metadata(self) -> list[dict[str, Any]]:
                return []

        self.provider = _P()


@pytest.mark.asyncio
async def test_hyde_map_passes_preserve_sources_in_bullet_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HyDE bullet grouping should preserve per-call sources metadata.

    This ensures that when CH_AGENT_DOC_CODE_RESEARCH_HYDE_GROUP_MODE=bullet
    we still aggregate sources from each deep_research call correctly before
    coverage stats are computed.
    """
    services = DummyServicesForMap()
    embedding_manager = EmbeddingManager()
    llm_manager = None

    # Simple HyDE plan with two bullet candidates under Global Hooks.
    hyde_plan = """
## Global Hooks
- First hook describing an important subsystem interaction in enough detail.
- Second hook describing another important flow in enough detail.
""".strip()

    calls: list[str] = []

    async def fake_run_research_with_metadata(
        services: Any,
        embedding_manager: EmbeddingManager,
        llm_manager: Any,
        prompt: str,
        scope_label: str | None = None,
        path_override: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        calls.append(prompt)
        idx = len(calls)
        if idx == 1:
            files = ["scope/a.py"]
        else:
            files = ["scope/b.py"]
        meta = {
            "sources": {
                "files": files,
                "chunks": [
                    {
                        "file_path": files[0],
                        "start_line": 1,
                        "end_line": 10,
                    }
                ],
            }
        }
        return f"ANSWER {idx}", meta

    monkeypatch.setenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_GROUP_MODE", "bullet")
    monkeypatch.delenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_MAP_LIMIT", raising=False)
    monkeypatch.setattr(
        deep_doc_mod,
        "_run_research_query_with_metadata",
        fake_run_research_with_metadata,
        raising=True,
    )

    findings, call_count = await deep_doc_mod._run_hyde_map_passes(
        services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        scope_label="scope",
        hyde_plan=hyde_plan,
    )

    # We should have run one deep_research call per bullet.
    assert call_count == 2
    assert len(findings) == 2

    files_seen: set[str] = set()
    for f in findings:
        src = f.get("sources") or {}
        for fp in src.get("files", []):
            files_seen.add(fp)

    assert files_seen == {"scope/a.py", "scope/b.py"}


@pytest.mark.asyncio
async def test_hyde_map_passes_preserve_sources_in_section_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HyDE section grouping should preserve per-call sources metadata.

    This ensures that when CH_AGENT_DOC_CODE_RESEARCH_HYDE_GROUP_MODE=section
    we still aggregate sources from each grouped section correctly.
    """
    services = DummyServicesForMap()
    embedding_manager = EmbeddingManager()
    llm_manager = None

    # HyDE plan with two sections under Global Hooks.
    hyde_plan = """
# HyDE Research Plan for ./scope

## Global Hooks

### Section One
- First hook describing an important subsystem.

### Section Two
- Second hook describing another key flow.
""".strip()

    calls: list[str] = []

    async def fake_run_research_with_metadata(
        services: Any,
        embedding_manager: EmbeddingManager,
        llm_manager: Any,
        prompt: str,
        scope_label: str | None = None,
        path_override: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        calls.append(prompt)
        idx = len(calls)
        if idx == 1:
            files = ["scope/a.py"]
        else:
            files = ["scope/b.py"]
        meta = {
            "sources": {
                "files": files,
                "chunks": [
                    {
                        "file_path": files[0],
                        "start_line": 1,
                        "end_line": 10,
                    }
                ],
            }
        }
        return f"ANSWER {idx}", meta

    monkeypatch.setenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_GROUP_MODE", "section")
    monkeypatch.delenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_MAP_LIMIT", raising=False)
    monkeypatch.setattr(
        deep_doc_mod,
        "_run_research_query_with_metadata",
        fake_run_research_with_metadata,
        raising=True,
    )

    findings, call_count = await deep_doc_mod._run_hyde_map_passes(
        services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        scope_label="scope",
        hyde_plan=hyde_plan,
    )

    # We should have run one deep_research call per section (two sections).
    assert call_count == 2
    assert len(findings) == 2

    files_seen: set[str] = set()
    for f in findings:
        src = f.get("sources") or {}
        for fp in src.get("files", []):
            files_seen.add(fp)

    assert files_seen == {"scope/a.py", "scope/b.py"}
