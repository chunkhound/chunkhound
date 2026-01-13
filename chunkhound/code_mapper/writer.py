from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.code_mapper.models import AgentDocMetadata, CodeMapperPOI
from chunkhound.code_mapper.render import (
    build_topic_artifacts,
    render_combined_document,
    render_index_document,
)
from chunkhound.utils.text import safe_scope_label


@dataclass
class CodeMapperWriteResult:
    doc_path: Path | None
    index_path: Path | None
    topic_paths: list[Path]
    unref_path: Path | None


def write_code_mapper_outputs(
    *,
    out_dir: Path,
    scope_label: str,
    meta: AgentDocMetadata,
    overview_answer: str,
    poi_sections: list[tuple[CodeMapperPOI, dict[str, Any]]],
    poi_sections_indexed: list[tuple[int, CodeMapperPOI, dict[str, Any]]],
    failed_poi_sections: list[tuple[int, CodeMapperPOI, str]] | None,
    coverage_lines: list[str],
    include_topics: bool,
    include_combined: bool,
    unreferenced_files: list[str] | None = None,
) -> CodeMapperWriteResult:
    """Write Code Mapper artifacts to disk and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_scope = safe_scope_label(scope_label)
    doc_path: Path | None = None
    index_path: Path | None = None
    topic_paths: list[Path] = []
    unref_path: Path | None = None

    if include_topics:
        unref_filename: str | None = None
        if unreferenced_files is not None:
            unref_filename = f"{safe_scope}_scope_unreferenced_files.txt"
            unref_path = out_dir / unref_filename
            unref_path.write_text(
                "\n".join(unreferenced_files) + ("\n" if unreferenced_files else ""),
                encoding="utf-8",
            )
            try:
                files_stats = meta.generation_stats.get("files")
                if isinstance(files_stats, dict):
                    files_stats["unreferenced_list_file"] = unref_filename
            except (AttributeError, TypeError) as exc:
                logger.debug(
                    f"Code Mapper: failed to attach unreferenced file artifact: {exc}"
                )

        topic_files, index_entries_by_mode = build_topic_artifacts(
            scope_label=scope_label,
            poi_sections_indexed=poi_sections_indexed,
            failed_poi_sections=failed_poi_sections,
        )
        for filename, content in topic_files:
            topic_path = out_dir / filename
            topic_path.write_text(content, encoding="utf-8")
            topic_paths.append(topic_path)

        index_doc = render_index_document(
            meta=meta,
            scope_label=scope_label,
            index_entries_by_mode=index_entries_by_mode,
            unref_filename=unref_filename,
        )
        index_path = out_dir / f"{safe_scope}_code_mapper_index.md"
        index_path.write_text(index_doc, encoding="utf-8")

    if include_combined:
        doc_path = out_dir / f"{safe_scope}_code_mapper.md"
        doc_content = render_combined_document(
            meta=meta,
            scope_label=scope_label,
            overview_answer=overview_answer,
            poi_sections=poi_sections,
            coverage_lines=coverage_lines,
        )
        doc_path.write_text(doc_content, encoding="utf-8")

    return CodeMapperWriteResult(
        doc_path=doc_path,
        index_path=index_path,
        topic_paths=topic_paths,
        unref_path=unref_path,
    )
