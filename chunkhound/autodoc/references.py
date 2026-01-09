from __future__ import annotations

import re

_SOURCES_HEADING_RE = re.compile(r"^##\s+Sources\s*$", re.IGNORECASE)
_REFERENCES_HEADING_RE = re.compile(r"^##\s+References\s*$", re.IGNORECASE)
_TREE_LINE_RE = re.compile(r"^(?P<prefix>.*?)[├└]──\s+(?P<content>.+)$")
_FILE_LINE_RE = re.compile(
    r"^\[(?P<ref>\d+)\]\s+(?P<name>.+?)(?:\s+\((?P<details>.+)\))?$"
)
_CITATION_RE = re.compile(r"\[(?P<ref>\d+)\]")
_PLAIN_CITATION_RE = re.compile(r"(?<!\[)\[(?P<ref>\d+)\](?!\])")
_FLAT_REFERENCE_RE = re.compile(r"^\s*-\s*\[(?P<ref>\d+)\]\s+")
_FLAT_REFERENCE_LINE_RE = re.compile(
    r"^(?P<prefix>\s*-\s+)\[(?P<ref>\d+)\](?P<rest>.*)$"
)


def _linkify_citations(markdown: str) -> str:
    output_lines: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            output_lines.append(line)
            continue
        if in_fence:
            output_lines.append(line)
            continue

        segments = line.split("`")
        for idx in range(0, len(segments), 2):
            segments[idx] = _PLAIN_CITATION_RE.sub(
                lambda match: f"[[{match.group('ref')}]](#ref-{match.group('ref')})",
                segments[idx],
            )
        output_lines.append("`".join(segments))
    return "\n".join(output_lines)


def _extract_cited_refs(markdown: str) -> set[str]:
    cited: set[str] = set()
    in_fence = False
    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        segments = line.split("`")
        for idx in range(0, len(segments), 2):
            cited.update(_PLAIN_CITATION_RE.findall(segments[idx]))
    return cited


def _anchor_reference_item(item: str) -> str:
    match = _FLAT_REFERENCE_LINE_RE.match(item)
    if not match:
        return item
    ref = match.group("ref")
    return f'{match.group("prefix")}<a id="ref-{ref}"></a>[{ref}]{match.group("rest")}'


def extract_sources_block(markdown: str) -> str | None:
    lines = markdown.splitlines()
    start_index = None
    for index, line in enumerate(lines):
        if _SOURCES_HEADING_RE.match(line.strip()):
            start_index = index
            break
    if start_index is None:
        return None
    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        if lines[index].startswith("## "):
            end_index = index
            break
    block = "\n".join(lines[start_index:end_index]).strip()
    return block or None


def strip_references_section(markdown: str) -> str:
    lines = markdown.splitlines()
    output_lines: list[str] = []
    skipping = False
    for line in lines:
        heading = line.strip()
        if heading.startswith("## "):
            if _SOURCES_HEADING_RE.match(heading) or _REFERENCES_HEADING_RE.match(
                heading
            ):
                while output_lines and not output_lines[-1].strip():
                    output_lines.pop()
                if output_lines and output_lines[-1].strip() == "---":
                    output_lines.pop()
                skipping = True
                continue
            if skipping:
                skipping = False
        if skipping:
            continue
        output_lines.append(line)
    return "\n".join(output_lines).strip()


def flatten_sources_block(sources_block: str) -> list[str]:
    lines = sources_block.splitlines()
    stack: list[str] = []
    flattened: list[str] = []
    for line in lines:
        match = _TREE_LINE_RE.match(line)
        if not match:
            continue
        prefix = match.group("prefix")
        content = match.group("content").strip()
        depth = prefix.count("\t")
        if content.endswith("/"):
            dirname = content.rstrip("/")
            if len(stack) <= depth:
                stack.extend([""] * (depth + 1 - len(stack)))
            stack[depth] = dirname
            del stack[depth + 1 :]
            continue

        file_match = _FILE_LINE_RE.match(content)
        if not file_match:
            continue
        ref = file_match.group("ref")
        name = file_match.group("name").strip()
        details = file_match.group("details")
        path_parts = stack[:depth]
        full_path = "/".join([*path_parts, name]).lstrip("/")
        path_display = f"`{full_path}`"
        if details:
            flattened.append(f"- [{ref}] {path_display} ({details})")
        else:
            flattened.append(f"- [{ref}] {path_display}")
    return flattened


def build_references_section(flat_items: list[str]) -> str:
    if not flat_items:
        return ""
    anchored_items = [_anchor_reference_item(item) for item in flat_items]
    lines = ["## References", "", *anchored_items]
    return "\n".join(lines).strip()


def _select_flat_references_for_cleaned_body(
    cleaned_body: str, sources_block: str
) -> list[str]:
    flat_items = flatten_sources_block(sources_block)
    if not flat_items:
        raise ValueError(
            "AutoDoc could not parse the `## Sources` tree into references. "
            "Expected Code Mapper tree formatting (├──/└── lines with tab-indented "
            "paths and `[N] filename (chunks: ...)` leaves). Refusing to drop "
            "references."
        )

    cited = _extract_cited_refs(cleaned_body)
    if not cited:
        return flat_items

    filtered: list[str] = []
    for item in flat_items:
        match = _FLAT_REFERENCE_RE.match(item)
        if not match:
            filtered.append(item)
            continue
        if match.group("ref") in cited:
            filtered.append(item)

    return filtered if filtered else flat_items


def _apply_reference_normalization(body: str, sources_block: str | None) -> str:
    cleaned = strip_references_section(body)
    if not sources_block:
        return cleaned.strip()
    flat_items = _select_flat_references_for_cleaned_body(cleaned, sources_block)
    references_block = build_references_section(flat_items)
    if not references_block:
        return cleaned.strip()
    linked_cleaned = _linkify_citations(cleaned)
    if linked_cleaned.strip():
        return linked_cleaned.strip() + "\n\n" + references_block
    return references_block
