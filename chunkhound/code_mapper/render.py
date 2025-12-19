from __future__ import annotations

from typing import Any

from chunkhound.code_mapper.metadata import format_metadata_block
from chunkhound.code_mapper.models import AgentDocMetadata
from chunkhound.code_mapper.pipeline import _derive_heading_from_point, _slugify_heading


def render_overview_document(
    *,
    meta: AgentDocMetadata,
    scope_label: str,
    overview_answer: str,
) -> str:
    """Render overview-only Code Mapper output."""
    lines: list[str] = [
        format_metadata_block(meta).rstrip("\n"),
        f"# Code Mapper Overview for {scope_label}",
        "",
        overview_answer.strip(),
        "",
    ]
    return "\n".join(lines)


def render_combined_document(
    *,
    meta: AgentDocMetadata,
    scope_label: str,
    overview_answer: str,
    poi_sections: list[tuple[str, dict[str, Any]]],
    coverage_lines: list[str],
) -> str:
    """Render the combined Code Mapper document for a scope."""
    lines: list[str] = [
        format_metadata_block(meta).rstrip("\n"),
        f"# Code Mapper for {scope_label}",
        "",
    ]
    lines.extend(coverage_lines)
    lines.append("")
    lines.append("## Points of Interest Overview")
    lines.append("")
    lines.append(overview_answer.strip())
    lines.append("")

    for idx, (poi, result) in enumerate(poi_sections, start=1):
        heading = _derive_heading_from_point(poi)
        lines.append(f"## {idx}. {heading}")
        lines.append("")
        lines.append(str(result.get("answer", "")).strip())
        lines.append("")

    lines.extend(coverage_lines)
    lines.append("")
    return "\n".join(lines)


def build_topic_artifacts(
    *,
    scope_label: str,
    poi_sections: list[tuple[str, dict[str, Any]]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return topic file contents plus index entries for each topic."""
    safe_scope = scope_label.replace("/", "_") or "root"
    topic_files: list[tuple[str, str]] = []
    index_entries: list[tuple[str, str]] = []

    for idx, (poi, result) in enumerate(poi_sections, start=1):
        heading = _derive_heading_from_point(poi)
        slug = _slugify_heading(heading)
        filename = f"{safe_scope}_topic_{idx:02d}_{slug}.md"
        content = "\n".join(
            [
                f"# {heading}",
                "",
                str(result.get("answer", "")).strip(),
                "",
            ]
        )
        topic_files.append((filename, content))
        index_entries.append((heading, filename))

    return topic_files, index_entries


def render_index_document(
    *,
    meta: AgentDocMetadata,
    scope_label: str,
    index_entries: list[tuple[str, str]],
    unref_filename: str | None = None,
) -> str:
    """Render the per-scope index of Code Mapper topics."""
    lines: list[str] = [
        format_metadata_block(meta).rstrip("\n"),
        f"# Code Mapper Topics for {scope_label}",
        "",
        "This index lists the per-topic Code Mapper sections generated for this scope.",
    ]
    if unref_filename is not None:
        lines.append(
            f"- Unreferenced files in scope: [{unref_filename}]({unref_filename})"
        )
    lines.append("")
    lines.append("## Topics")
    lines.append("")

    for idx, (heading, filename) in enumerate(index_entries, start=1):
        lines.append(f"{idx}. [{heading}]({filename})")

    return "\n".join(lines) + "\n"
