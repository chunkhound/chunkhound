from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class IndexTopicEntry:
    order: int
    title: str
    filename: str


@dataclass
class CodeMapperIndex:
    title: str
    scope_label: str
    metadata_block: str | None
    topics: list[IndexTopicEntry]


@dataclass
class CodeMapperTopic:
    order: int
    title: str
    source_path: Path
    raw_markdown: str
    body_markdown: str


@dataclass
class DocsitePage:
    order: int
    title: str
    slug: str
    description: str
    body_markdown: str
    source_path: str | None = None
    scope_label: str | None = None
    references_count: int | None = None


@dataclass
class DocsiteSite:
    title: str
    tagline: str
    scope_label: str
    generated_at: str
    source_dir: str
    topic_count: int


@dataclass
class DocsiteResult:
    output_dir: Path
    pages: list[DocsitePage]
    index: CodeMapperIndex
    missing_topics: list[str]


@dataclass
class CleanupConfig:
    mode: str
    batch_size: int
    max_completion_tokens: int
    audience: str = "balanced"


@dataclass
class NavGroup:
    title: str
    slugs: list[str]


@dataclass
class GlossaryTerm:
    term: str
    definition: str
    pages: list[str]
