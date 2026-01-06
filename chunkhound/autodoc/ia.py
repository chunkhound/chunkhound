from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from chunkhound.autodoc.markdown_utils import (
    _strip_first_heading,
    _strip_markdown_for_search,
)
from chunkhound.autodoc.models import DocsitePage, GlossaryTerm, NavGroup
from chunkhound.core.audience import normalize_audience
from chunkhound.interfaces.llm_provider import LLMProvider

_SITE_IA_INPUT_LINE = (
    "Input: a list of pages (title, slug, description, headings, overview snippet)."
)
_SITE_IA_TECH_PREFER_LINE = (
    "Prefer architecture- and implementation-oriented groups/terms when supported "
    "by the provided headings/snippets."
)
_SITE_IA_TECH_STYLE_LINE = (
    "Use precise technical language; keep it concrete and grounded in the input."
)
_SITE_IA_END_USER_PREFER_LINE = (
    "Prefer user goals: setup, configuration, usage, and integration workflows "
    "when supported by the provided headings/snippets."
)
_SITE_IA_END_USER_CODE_IDS_LINE = (
    "Keep code identifiers, but explain them in plain language; keep definitions "
    "short and avoid jargon unless the input uses it prominently."
)
_SITE_IA_END_USER_DEEMPHASIZE_LINE = (
    "De-emphasize deep internal implementation details unless central in the "
    "provided snippets/headings."
)
_SITE_IA_NAV_INTRO_LINE = (
    "1) A navigation structure that groups pages where it helps discoverability."
)
_SITE_IA_NAV_GROUP_COUNT_LINE = (
    "   - Use as many or as few groups as fits the content (for a small site, "
    "a single group is fine)."
)
_SITE_IA_GLOSSARY_DEF_LINE = (
    "   - definition (1–2 sentences, only supported by provided snippets/headings)"
)

_HOMEPAGE_INPUT_LINE = (
    "Input: a list of pages (title, slug, description, headings, overview snippet)."
)
_HOMEPAGE_GOAL_TECH_LINE = (
    "Goal: briefly explain what the project does and the major system boundaries "
    "and flows."
)
_HOMEPAGE_GOAL_TECH_PREF_LINE = (
    "Prefer concrete technical terms present in the input (commands, modules, "
    "components)."
)
_HOMEPAGE_GOAL_END_USER_LINE = (
    "Goal: briefly explain what the project is for and the main ways a user "
    "will interact with it."
)
_HOMEPAGE_NO_INVENT_LINE = (
    "Do NOT invent installation steps, commands, or configuration keys that "
    "are not supported by the input."
)


def _site_ia_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "nav": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "groups": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "title": {"type": "string"},
                                "slugs": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["title", "slugs"],
                        },
                    }
                },
                "required": ["groups"],
            },
            "glossary": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "term": {"type": "string"},
                        "definition": {"type": "string"},
                        "pages": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["term", "definition", "pages"],
                },
            },
        },
        "required": ["nav", "glossary"],
    }


def _build_site_ia_prompt(*, context: list[dict[str, Any]], audience: str) -> str:
    normalized = normalize_audience(audience)
    audience_lines: list[str] = []
    if normalized == "technical":
        audience_lines = [
            "Audience: technical (software engineers).",
            (
                _SITE_IA_TECH_PREFER_LINE
            ),
            (
                _SITE_IA_TECH_STYLE_LINE
            ),
        ]
    elif normalized == "end-user":
        audience_lines = [
            "Audience: end-user (less technical).",
            (
                _SITE_IA_END_USER_PREFER_LINE
            ),
            (
                _SITE_IA_END_USER_CODE_IDS_LINE
            ),
            (
                _SITE_IA_END_USER_DEEMPHASIZE_LINE
            ),
        ]

    parts = [
        "You are the information architect for an engineering docs site.",
        _SITE_IA_INPUT_LINE,
    ]
    if audience_lines:
        parts.extend(["", *audience_lines])
    parts.extend(
        [
            "",
            "Produce:",
            (
                _SITE_IA_NAV_INTRO_LINE
            ),
            (
                _SITE_IA_NAV_GROUP_COUNT_LINE
            ),
            "   - Each group has a title and an ordered list of page slugs.",
            "   - Each page slug should appear at most once across all groups.",
            "2) A glossary of 20–60 terms. Each term has:",
            "   - term",
            (
                _SITE_IA_GLOSSARY_DEF_LINE
            ),
            "   - pages: list of slugs where it appears",
            "",
            "Output STRICT JSON with keys: nav, glossary.",
            "",
            "Pages JSON:",
            json.dumps(context, ensure_ascii=True, indent=2),
        ]
    )
    return "\n".join(parts)


async def _synthesize_site_ia(
    *,
    pages: list[DocsitePage],
    provider: LLMProvider,
    audience: str = "balanced",
    log_info: Callable[[str], None] | None,
    log_warning: Callable[[str], None] | None,
) -> tuple[list[NavGroup] | None, list[GlossaryTerm] | None]:
    if not pages:
        return None, None

    slugs = {page.slug for page in pages}
    context = _build_site_context(pages)

    prompt = _build_site_ia_prompt(context=context, audience=audience)

    schema = _site_ia_schema()

    if log_info:
        log_info("Synthesizing global navigation and glossary.")

    response = await provider.complete_structured(
        prompt,
        json_schema=schema,
        system=None,
        max_completion_tokens=4096,
    )

    nav_groups = _validate_nav_groups(
        response.get("nav"),
        slugs,
        [page.slug for page in pages],
        log_warning,
    )
    glossary_terms = _validate_glossary_terms(
        response.get("glossary"),
        slugs,
        log_warning,
    )

    if not nav_groups:
        return None, glossary_terms or None
    return nav_groups, glossary_terms or None


def _build_site_context(pages: list[DocsitePage]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in pages:
        headings = _extract_headings(page.body_markdown)
        overview_snippet = _extract_overview_snippet(page.body_markdown, limit=400)
        records.append(
            {
                "title": page.title,
                "slug": page.slug,
                "description": page.description,
                "headings": headings,
                "overviewSnippet": overview_snippet,
            }
        )
    return records


def _normalize_homepage_overview(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = _strip_first_heading(cleaned)
    cleaned = re.sub(
        r"^\s*##\s+Overview\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE
    ).strip()
    cleaned = re.sub(
        r"^\s*##\s+Topics\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE
    ).strip()
    return cleaned.strip()


async def _synthesize_homepage_overview(
    *,
    pages: list[DocsitePage],
    provider: LLMProvider,
    audience: str = "balanced",
    log_info: Callable[[str], None] | None,
    log_warning: Callable[[str], None] | None,
) -> str | None:
    if not pages:
        return None

    context = _build_site_context(pages)
    normalized = normalize_audience(audience)

    audience_line = "Audience: balanced."
    goal_lines = [
        "Goal: briefly explain what the project is for and how to navigate the docs.",
        "Use only information supported by the provided snippets/headings.",
    ]
    if normalized == "technical":
        audience_line = "Audience: technical (software engineers)."
        goal_lines = [
            _HOMEPAGE_GOAL_TECH_LINE,
            _HOMEPAGE_GOAL_TECH_PREF_LINE,
        ]
    elif normalized == "end-user":
        audience_line = "Audience: end-user (less technical)."
        goal_lines = [
            _HOMEPAGE_GOAL_END_USER_LINE,
            "Keep code identifiers if prominent, but explain them in plain language.",
        ]

    prompt = "\n".join(
        [
            "You are writing a short overview for a documentation site homepage.",
            _HOMEPAGE_INPUT_LINE,
            "",
            audience_line,
            *goal_lines,
            (
                _HOMEPAGE_NO_INVENT_LINE
            ),
            "Do NOT include YAML frontmatter. Do NOT include a level-1 heading.",
            "Do NOT include a '## Topics' section or list topic filenames/titles.",
            "",
            "Output Markdown only. Prefer 1 short paragraph plus 3–6 bullet points.",
            "",
            "Pages JSON:",
            json.dumps(context, ensure_ascii=True, indent=2),
        ]
    )

    if log_info:
        log_info("Synthesizing homepage overview.")

    try:
        response = await provider.complete(
            prompt,
            system=None,
            max_completion_tokens=600,
        )
    except Exception as exc:  # noqa: BLE001
        if log_warning:
            log_warning(f"Homepage overview synthesis failed; skipping. Error: {exc}")
        return None

    overview = _normalize_homepage_overview(response.content)
    return overview or None


def _extract_headings(markdown: str) -> list[str]:
    headings: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            title = stripped[3:].strip()
            if title.lower() in {"references", "sources"}:
                continue
            headings.append(title)
        elif stripped.startswith("### "):
            headings.append(stripped[4:].strip())
    return headings


def _extract_overview_snippet(markdown: str, *, limit: int) -> str:
    lines = markdown.splitlines()
    start: int | None = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "## overview":
            start = idx + 1
            break
    if start is None:
        return ""
    collected: list[str] = []
    for line in lines[start:]:
        if line.strip().startswith("## "):
            break
        collected.append(line)
    snippet = _strip_markdown_for_search("\n".join(collected)).strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 3].rstrip() + "..."


def _validate_nav_groups(
    nav: object,
    valid_slugs: set[str],
    ordered_slugs: list[str],
    log_warning: Callable[[str], None] | None,
) -> list[NavGroup]:
    if not isinstance(nav, dict):
        if log_warning:
            log_warning("Global IA output missing 'nav' object; skipping nav.json.")
        return []
    groups = nav.get("groups")
    if not isinstance(groups, list):
        return []
    output: list[NavGroup] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            continue
        title = group.get("title")
        slugs = group.get("slugs")
        if not isinstance(title, str) or not isinstance(slugs, list):
            continue
        cleaned_slugs: list[str] = []
        for slug in slugs:
            if not isinstance(slug, str) or slug not in valid_slugs:
                continue
            if slug in seen:
                continue
            cleaned_slugs.append(slug)
            seen.add(slug)
        if not cleaned_slugs:
            continue
        output.append(NavGroup(title=title.strip() or "Group", slugs=cleaned_slugs))

    missing = [
        slug for slug in ordered_slugs if slug in valid_slugs and slug not in seen
    ]
    if missing:
        if output:
            output.append(NavGroup(title="More", slugs=missing))
        else:
            output.append(NavGroup(title="Topics", slugs=missing))
    return output


def _validate_glossary_terms(
    glossary: object,
    valid_slugs: set[str],
    log_warning: Callable[[str], None] | None,
) -> list[GlossaryTerm]:
    if not isinstance(glossary, list):
        if log_warning:
            log_warning(
                "Global IA output missing 'glossary' list; skipping glossary.md."
            )
        return []
    output: list[GlossaryTerm] = []
    for entry in glossary:
        if not isinstance(entry, dict):
            continue
        term = entry.get("term")
        definition = entry.get("definition")
        pages = entry.get("pages")
        if (
            not isinstance(term, str)
            or not isinstance(definition, str)
            or not isinstance(pages, list)
        ):
            continue
        cleaned_pages = [
            slug for slug in pages if isinstance(slug, str) and slug in valid_slugs
        ]
        output.append(
            GlossaryTerm(
                term=term.strip(),
                definition=definition.strip(),
                pages=cleaned_pages,
            )
        )
    return [item for item in output if item.term and item.definition]
