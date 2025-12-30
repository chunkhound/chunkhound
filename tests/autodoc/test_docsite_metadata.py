from chunkhound.autodoc.models import CodeMapperIndex
from chunkhound.autodoc.site_writer import _render_index_metadata


def test_render_index_metadata_includes_nested_details() -> None:
    metadata_block = """agent_doc_metadata:
  created_from_sha: abc123
  generated_at: 2025-01-01T00:00:00Z
  llm_config:
    provider: codex-cli
    synthesis_provider: codex-cli
    synthesis_model: gpt-5
    utility_model: gpt-5-mini
    codex_reasoning_effort_synthesis: high
    codex_reasoning_effort_utility: low
    map_hyde_provider: codex-cli
    map_hyde_model: gpt-5
    map_hyde_reasoning_effort: medium
  generation_stats:
    generator_mode: code_research
    total_research_calls: 10
    autodoc_comprehensiveness: medium
    files:
      referenced: 5
      total_indexed: 20
      basis: scope
      coverage: 25.0%
      referenced_in_scope: 5
      unreferenced_in_scope: 15
      unreferenced_list_file: unref.txt
    chunks:
      referenced: 30
      total_indexed: 200
      basis: scope
      coverage: 15.0%
"""

    index = CodeMapperIndex(
        title="AutoDoc Topics",
        scope_label="/",
        metadata_block=metadata_block,
        topics=[],
    )

    lines = _render_index_metadata(index)

    assert "- Generated at: 2025-01-01T00:00:00Z" in lines
    assert "- Source SHA: abc123" in lines
    assert "- LLM provider: codex-cli" in lines
    assert "- Synthesis model: gpt-5" in lines
    assert "- Utility model: gpt-5-mini" in lines
    assert "- Synthesis reasoning effort: high" in lines
    assert "- Utility reasoning effort: low" in lines
    assert "- HyDE planning provider: codex-cli" in lines
    assert "- HyDE planning model: gpt-5" in lines
    assert "- HyDE planning reasoning effort: medium" in lines
    assert "- Generator mode: code_research" in lines
    assert "- Comprehensiveness: medium" in lines
    assert "- Research calls: 10" in lines
    assert "- Files referenced: 5 / 20 (25.0%), basis: scope" in lines
    assert "- Files referenced in scope: 5" in lines
    assert "- Files unreferenced in scope: 15" in lines
    assert "- Chunks referenced: 30 / 200 (15.0%), basis: scope" in lines
