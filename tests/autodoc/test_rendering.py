from chunkhound.autodoc import pipeline as autodoc_pipeline
from chunkhound.autodoc import render as autodoc_render
from chunkhound.autodoc.models import AgentDocMetadata


def _meta() -> AgentDocMetadata:
    return AgentDocMetadata(
        created_from_sha="abc123",
        previous_target_sha="abc123",
        target_sha="abc123",
        generated_at="2025-12-19T00:00:00Z",
        llm_config={"provider": "test"},
        generation_stats={"autodoc_comprehensiveness": "low"},
    )


def test_render_overview_document_includes_metadata_and_header() -> None:
    doc = autodoc_render.render_overview_document(
        meta=_meta(),
        scope_label="scope",
        overview_answer="Overview content",
    )

    assert "agent_doc_metadata:" in doc
    assert "# AutoDoc Overview for scope" in doc
    assert "Overview content" in doc


def test_render_combined_document_includes_sections_and_coverage() -> None:
    coverage_lines = autodoc_pipeline._coverage_summary_lines(
        referenced_files=1,
        referenced_chunks=2,
        files_denominator=2,
        chunks_denominator=4,
        scope_total_files=2,
        scope_total_chunks=4,
    )
    poi_sections = [("Core Flow: details", {"answer": "Section body"})]

    doc = autodoc_render.render_combined_document(
        meta=_meta(),
        scope_label="scope",
        overview_answer="Overview content",
        poi_sections=poi_sections,
        coverage_lines=coverage_lines,
    )

    assert doc.count("## Coverage Summary") == 2
    assert "# AutoDoc for scope" in doc
    assert "## Points of Interest Overview" in doc
    assert "## 1. Core Flow" in doc
    assert "Section body" in doc


def test_build_topic_artifacts_and_index() -> None:
    poi_sections = [("**Core Flow**: details", {"answer": "Section body"})]

    topic_files, index_entries = autodoc_render.build_topic_artifacts(
        scope_label="scope",
        poi_sections=poi_sections,
    )

    assert topic_files, "Expected at least one topic file"
    filename, content = topic_files[0]
    assert filename == "scope_topic_01_core-flow.md"
    assert content.startswith("# Core Flow")
    assert "Section body" in content

    index_doc = autodoc_render.render_index_document(
        meta=_meta(),
        scope_label="scope",
        index_entries=index_entries,
        unref_filename="scope_scope_unreferenced_files.txt",
    )

    assert "AutoDoc Topics for scope" in index_doc
    assert "[Core Flow](scope_topic_01_core-flow.md)" in index_doc
    assert "scope_scope_unreferenced_files.txt" in index_doc
