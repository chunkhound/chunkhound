from pathlib import Path

from chunkhound.code_mapper.models import AgentDocMetadata
from chunkhound.code_mapper.writer import write_code_mapper_outputs


def _meta() -> AgentDocMetadata:
    return AgentDocMetadata(
        created_from_sha="abc123",
        previous_target_sha="abc123",
        target_sha="abc123",
        generated_at="2025-12-19T00:00:00Z",
        llm_config={},
        generation_stats={"files": {}},
    )


def test_write_code_mapper_outputs_without_topics(tmp_path: Path) -> None:
    result = write_code_mapper_outputs(
        out_dir=tmp_path,
        scope_label="scope",
        meta=_meta(),
        overview_answer="Overview",
        poi_sections=[],
        coverage_lines=["## Coverage Summary", ""],
        include_topics=False,
        include_combined=True,
        unreferenced_files=None,
    )

    assert result.doc_path.exists()
    assert result.index_path is None
    assert result.topic_paths == []
    assert result.unref_path is None


def test_write_code_mapper_outputs_with_topics_and_unreferenced(tmp_path: Path) -> None:
    meta = _meta()
    result = write_code_mapper_outputs(
        out_dir=tmp_path,
        scope_label="scope",
        meta=meta,
        overview_answer="Overview",
        poi_sections=[("Core Flow", {"answer": "Section body"})],
        coverage_lines=["## Coverage Summary", ""],
        include_topics=True,
        include_combined=True,
        unreferenced_files=["scope/a.py", "scope/b.py"],
    )

    assert result.doc_path.exists()
    assert result.index_path is not None and result.index_path.exists()
    assert result.topic_paths and result.topic_paths[0].exists()
    assert result.unref_path is not None and result.unref_path.exists()

    index_content = result.index_path.read_text(encoding="utf-8")
    assert "scope_scope_unreferenced_files.txt" in index_content
    assert (
        meta.generation_stats["files"]["unreferenced_list_file"]
        == "scope_scope_unreferenced_files.txt"
    )
