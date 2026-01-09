from __future__ import annotations

from typing import Any

from chunkhound.code_mapper.metadata import format_metadata_block
from chunkhound.code_mapper.models import AgentDocMetadata, CodeMapperPOI
from chunkhound.code_mapper.public_utils import (
    derive_heading_from_point,
    slugify_heading,
)
from chunkhound.utils.text import safe_scope_label


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
    poi_sections: list[tuple[CodeMapperPOI, dict[str, Any]]],
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
    lines.append("## HyDE Overview")
    lines.append("")
    lines.append(overview_answer.strip())
    lines.append("")

    arch_sections = [
        (poi, result)
        for poi, result in poi_sections
        if poi.mode == "architectural"
    ]
    ops_sections = [
        (poi, result)
        for poi, result in poi_sections
        if poi.mode == "operational"
    ]

    if arch_sections:
        lines.append("## Architectural Map")
        lines.append("")
        for idx, (poi, result) in enumerate(arch_sections, start=1):
            heading = derive_heading_from_point(poi.text)
            lines.append(f"### {idx}. {heading}")
            lines.append("")
            lines.append(str(result.get("answer", "")).strip())
            lines.append("")

    if ops_sections:
        lines.append("## Operational Map")
        lines.append("")
        for idx, (poi, result) in enumerate(ops_sections, start=1):
            heading = derive_heading_from_point(poi.text)
            lines.append(f"### {idx}. {heading}")
            lines.append("")
            lines.append(str(result.get("answer", "")).strip())
            lines.append("")

    return "\n".join(lines)


def build_topic_artifacts(
    *,
    scope_label: str,
    poi_sections_indexed: list[tuple[int, CodeMapperPOI, dict[str, Any]]],
    failed_poi_sections: list[tuple[int, CodeMapperPOI, str]] | None = None,
) -> tuple[list[tuple[str, str]], dict[str, list[tuple[str, str]]]]:
    """Return topic file contents plus index entries for each topic."""
    safe_scope = safe_scope_label(scope_label)
    topic_files: list[tuple[str, str]] = []
    index_entries_by_mode: dict[str, list[tuple[str, str]]] = {
        "architectural": [],
        "operational": [],
    }
    entries: list[tuple[int, CodeMapperPOI, str, str, bool]] = []
    for idx, poi, result in poi_sections_indexed:
        heading = derive_heading_from_point(poi.text)
        content = "\n".join(
            [
                f"# {heading}",
                "",
                str(result.get("answer", "")).strip(),
                "",
            ]
        )
        entries.append((idx, poi, heading, content, False))

    if failed_poi_sections:
        for idx, poi, content in failed_poi_sections:
            heading = derive_heading_from_point(poi.text)
            entries.append((idx, poi, heading, content.rstrip() + "\n", True))

    entries.sort(key=lambda entry: entry[0])

    arch_idx = 0
    ops_idx = 0
    for _idx, poi, heading, content, is_failed in entries:
        slug = slugify_heading(heading)
        mode = poi.mode
        if mode == "operational":
            ops_idx += 1
            filename = f"{safe_scope}_ops_topic_{ops_idx:02d}_{slug}.md"
        else:
            arch_idx += 1
            filename = f"{safe_scope}_arch_topic_{arch_idx:02d}_{slug}.md"
        topic_files.append((filename, content))

        display_heading = f"{heading} (failed)" if is_failed else heading
        index_entries_by_mode.setdefault(mode, []).append((display_heading, filename))

    return topic_files, index_entries_by_mode


def render_index_document(
    *,
    meta: AgentDocMetadata,
    scope_label: str,
    index_entries_by_mode: dict[str, list[tuple[str, str]]],
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
    lines.append("## Architectural Map")
    lines.append("")

    for idx, (heading, filename) in enumerate(
        index_entries_by_mode.get("architectural") or [], start=1
    ):
        lines.append(f"{idx}. [{heading}]({filename})")

    lines.append("")
    lines.append("## Operational Map")
    lines.append("")

    for idx, (heading, filename) in enumerate(
        index_entries_by_mode.get("operational") or [], start=1
    ):
        lines.append(f"{idx}. [{heading}]({filename})")

    return "\n".join(lines) + "\n"
