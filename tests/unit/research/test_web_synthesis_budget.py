"""The outgoing synthesis request must preserve web evidence selection budgets."""

from types import SimpleNamespace

import pytest

from chunkhound.services.clustering_service import ClusterGroup
from chunkhound.services.research import SynthesisEngine
from chunkhound.services.research.shared.citation_manager import CitationManager
from chunkhound.services.research.shared.file_reader import FileReader
from chunkhound.services.research.shared.models import ResearchContext


async def _synthesize(stage, manager, chunks, files):
    engine = SynthesisEngine(
        manager,
        database_services=object(),
        parent_service=SimpleNamespace(_citation_manager=CitationManager()),
    )
    context = ResearchContext(root_query="Explain the selected evidence")
    budgets = {"output_tokens": 10_000}
    if stage == "single":
        await engine._single_pass_synthesis(chunks, files, context, budgets)
    else:
        # Clustering partitions values into fresh ordinary dictionaries. A
        # marker attached only to the original mapping would be lost here.
        cluster = ClusterGroup(
            cluster_id=0,
            file_paths=list(files),
            files_content={path: files[path] for path in files},
            total_tokens=100,
        )
        await engine._map_synthesis_on_cluster(
            cluster, chunks, context, budgets, total_input_tokens=100
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["single", "map"])
@pytest.mark.parametrize(
    ("byte_budget", "token_budget"),
    [(256, None), (4000, 16), (100_000, 16)],
    ids=["chunk-byte-cap", "chunk-token-cap", "full-source-token-cap"],
)
async def test_web_prompt_uses_only_budgeted_evidence(
    stage, byte_budget, token_budget, capturing_llm_manager
):
    manager, llm = capturing_llm_manager
    path = "https://example.invalid/evidence"
    selected = "SELECTED_WEB_EVIDENCE " + "x" * 1000
    excluded = "EXCLUDED_CHUNK_MUST_NOT_REACH_LLM"
    source = selected + "\n" + excluded + "\n" + "z" * 5000

    async def load_source(file_path):
        assert file_path == path
        return source

    services = SimpleNamespace(
        provider=SimpleNamespace(
            transient_source_budget_bytes=byte_budget,
            get_transient_file_content_async=load_source,
        )
    )
    chunks = [
        {"file_path": path, "content": selected, "start_line": 1, "end_line": 1},
        {"file_path": path, "content": excluded, "start_line": 2, "end_line": 2},
    ]
    original_chunks = [dict(chunk) for chunk in chunks]
    files = await FileReader(services).read_files_with_budget(
        chunks, manager, max_tokens=token_budget
    )
    assert sum(len(text.encode("utf-8")) for text in files.values()) <= byte_budget
    if token_budget is not None:
        assert sum(llm.estimate_tokens(text) for text in files.values()) <= token_budget

    await _synthesize(stage, manager, chunks, dict(files))

    assert len(llm.calls) == 1
    prompt = llm.calls[0]["prompt"]
    assert "SELECTED_WEB_EVIDENCE" in prompt
    assert files[path] in prompt
    assert excluded not in prompt
    assert "z" * 100 not in prompt
    assert chunks == original_chunks  # Selection must not mutate search results.


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["single", "map"])
async def test_code_prompt_retains_chunk_line_markers(stage, capturing_llm_manager):
    manager, llm = capturing_llm_manager
    chunks = [
        {
            "file_path": "src/code.py",
            "content": "SELECTED_CODE_CHUNK",
            "start_line": 12,
            "end_line": 15,
        }
    ]
    await _synthesize(
        stage, manager, chunks, {"src/code.py": "UNSELECTED_FILE_CONTEXT"}
    )
    prompt = llm.calls[0]["prompt"]
    assert "# Lines 12-15\nSELECTED_CODE_CHUNK" in prompt
    assert "UNSELECTED_FILE_CONTEXT" not in prompt
