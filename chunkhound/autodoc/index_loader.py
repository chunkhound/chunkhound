from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

from chunkhound.autodoc.markdown_utils import (
    _first_heading,
    _heading_text,
    _remove_duplicate_title_line,
    _scope_from_heading,
    _strip_first_heading,
    _strip_metadata_block,
)
from chunkhound.autodoc.models import CodeMapperIndex, CodeMapperTopic, IndexTopicEntry

_INDEX_PATTERNS = (
    "*_code_mapper_index.md",
    "*_autodoc_index.md",
)

_TOPIC_LINK_RE = re.compile(r"^\s*\d+\.\s+\[(?P<title>.+?)\]\((?P<filename>.+?)\)\s*$")


def find_index_file(
    input_dir: Path,
    patterns: Iterable[str] | None = None,
    log_warning: Callable[[str], None] | None = None,
) -> Path:
    pattern_list = list(patterns) if patterns else list(_INDEX_PATTERNS)
    candidates: list[Path] = []
    for pattern in pattern_list:
        candidates.extend(sorted(input_dir.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(
            "No AutoDoc index file found (expected " + ", ".join(pattern_list) + ")."
        )
    if len(candidates) > 1 and log_warning:
        log_warning(
            "Multiple AutoDoc index files found; using first match: "
            f"{candidates[0]}. Consider --index-pattern to disambiguate."
        )
    return candidates[0]


def parse_index_file(index_path: Path) -> CodeMapperIndex:
    content = index_path.read_text(encoding="utf-8")
    metadata_block, body = _strip_metadata_block(content)

    title_line = _first_heading(body) or "AutoDoc Topics"
    scope_label = _scope_from_heading(title_line)

    topics: list[IndexTopicEntry] = []
    for line in body.splitlines():
        match = _TOPIC_LINK_RE.match(line.strip())
        if not match:
            continue
        order = len(topics) + 1
        topics.append(
            IndexTopicEntry(
                order=order,
                title=match.group("title").strip(),
                filename=match.group("filename").strip(),
            )
        )

    title = _heading_text(title_line) or "AutoDoc Topics"
    return CodeMapperIndex(
        title=title,
        scope_label=scope_label,
        metadata_block=metadata_block,
        topics=topics,
    )


def load_topics(
    input_dir: Path,
    index: CodeMapperIndex,
    log_warning: Callable[[str], None] | None = None,
) -> tuple[list[CodeMapperTopic], list[str]]:
    topics: list[CodeMapperTopic] = []
    missing: list[str] = []

    for entry in index.topics:
        topic_path = input_dir / entry.filename
        if not topic_path.exists():
            missing.append(entry.filename)
            if log_warning:
                log_warning(f"Missing topic file referenced in index: {entry.filename}")
            continue
        raw = topic_path.read_text(encoding="utf-8")
        raw = _strip_metadata_block(raw)[1]
        heading_line = _first_heading(raw) or entry.title
        heading = _heading_text(heading_line)
        body = _strip_first_heading(raw)
        body = _remove_duplicate_title_line(body, heading)
        topics.append(
            CodeMapperTopic(
                order=entry.order,
                title=heading,
                source_path=topic_path,
                raw_markdown=raw,
                body_markdown=body.strip(),
            )
        )

    return topics, missing

