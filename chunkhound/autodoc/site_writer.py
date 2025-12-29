from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from shutil import rmtree
from typing import Any

from chunkhound.code_mapper.utils import safe_scope_label
from chunkhound.autodoc.models import (
    CodeMapperIndex,
    DocsitePage,
    DocsiteSite,
    GlossaryTerm,
    NavGroup,
)
from chunkhound.autodoc.template_loader import load_bytes, load_text
from chunkhound.autodoc.markdown_utils import (
    _ensure_overview_heading,
    _escape_json,
    _escape_yaml,
    _strip_markdown_for_search,
)
from chunkhound.autodoc.references import strip_references_section


def write_astro_assets_only(*, output_dir: Path) -> None:
    """
    Update only the Astro runtime assets in an already-generated docsite.

    Intended for iterating on UI/layout without rewriting topic markdown pages.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    src_dir = output_dir / "src"
    layouts_dir = src_dir / "layouts"
    styles_dir = src_dir / "styles"
    public_dir = output_dir / "public"

    for path in (layouts_dir, styles_dir, public_dir):
        path.mkdir(parents=True, exist_ok=True)

    _write_text(output_dir / "astro.config.mjs", _render_astro_config())
    _write_text(output_dir / "tsconfig.json", _render_tsconfig())

    site = _load_site_from_existing(output_dir)
    if site is not None:
        _write_text(output_dir / "package.json", _render_package_json(site))
        _write_text(output_dir / "README.md", _render_readme(site))

    _write_text(layouts_dir / "DocLayout.astro", _render_doc_layout())
    _write_text(styles_dir / "global.css", _render_global_css())
    _write_bytes(public_dir / "favicon.ico", _render_favicon_bytes())


def write_astro_site(
    *,
    output_dir: Path,
    site: DocsiteSite,
    pages: list[DocsitePage],
    index: CodeMapperIndex,
    nav_groups: list[NavGroup] | None = None,
    glossary_terms: list[GlossaryTerm] | None = None,
    homepage_overview: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_text(output_dir / "package.json", _render_package_json(site))
    _write_text(output_dir / "astro.config.mjs", _render_astro_config())
    _write_text(output_dir / "tsconfig.json", _render_tsconfig())
    _write_text(output_dir / "README.md", _render_readme(site))

    src_dir = output_dir / "src"
    pages_dir = src_dir / "pages"
    topics_dir = pages_dir / "topics"
    layouts_dir = src_dir / "layouts"
    styles_dir = src_dir / "styles"
    data_dir = src_dir / "data"
    public_dir = output_dir / "public"

    if topics_dir.exists():
        rmtree(topics_dir)

    for path in (pages_dir, topics_dir, layouts_dir, styles_dir, data_dir, public_dir):
        path.mkdir(parents=True, exist_ok=True)

    _write_text(data_dir / "site.json", _render_site_json(site))
    _write_text(data_dir / "search.json", _render_search_index(pages))
    _write_text(layouts_dir / "DocLayout.astro", _render_doc_layout())
    _write_text(styles_dir / "global.css", _render_global_css())
    _write_bytes(public_dir / "favicon.ico", _render_favicon_bytes())

    nav_payload: list[NavGroup]
    if nav_groups:
        nav_payload = nav_groups
    else:
        nav_payload = [{"title": "Topics", "slugs": [page.slug for page in pages]}]
    _write_text(data_dir / "nav.json", _render_nav_json(nav_payload))

    _write_text(
        pages_dir / "index.md",
        _render_index_page(
            site=site, pages=pages, index=index, overview_markdown=homepage_overview
        ),
    )

    glossary_path = pages_dir / "glossary.md"
    if glossary_terms:
        _write_text(glossary_path, _render_glossary_page(glossary_terms))
    elif glossary_path.exists():
        glossary_path.unlink()

    for page in pages:
        _write_text(
            topics_dir / f"{page.slug}.md",
            _render_topic_page(page),
        )


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.write_bytes(content)


def _render_favicon_bytes() -> bytes:
    return load_bytes("public/favicon.ico")


def _render_package_json(site: DocsiteSite) -> str:
    safe_name = safe_scope_label(site.scope_label).replace("_", "-")
    package_name = f"chunkhound-docs-{safe_name}" if safe_name else "chunkhound-docs"
    return load_text("package.json").replace("{{PACKAGE_NAME}}", package_name)


def _render_astro_config() -> str:
    return load_text("astro.config.mjs")


def _render_tsconfig() -> str:
    return load_text("tsconfig.json")


def _render_readme(site: DocsiteSite) -> str:
    return load_text("README.md").replace("{{TITLE}}", site.title)


def _render_site_json(site: DocsiteSite) -> str:
    return "\n".join(
        [
            "{",
            f'  "title": "{_escape_json(site.title)}",',
            f'  "tagline": "{_escape_json(site.tagline)}",',
            f'  "scopeLabel": "{_escape_json(site.scope_label)}",',
            f'  "generatedAt": "{_escape_json(site.generated_at)}",',
            f'  "sourceDir": "{_escape_json(site.source_dir)}",',
            f'  "topicCount": {site.topic_count},',
            '  "watermark": "Generated by ChunkHound"',
            "}",
            "",
        ]
    )


def _load_site_from_existing(output_dir: Path) -> DocsiteSite | None:
    site_path = output_dir / "src" / "data" / "site.json"
    try:
        if not site_path.exists():
            return None
        payload = json.loads(site_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        title = payload.get("title")
        tagline = payload.get("tagline")
        scope_label = payload.get("scopeLabel")
        generated_at = (
            payload.get("generatedAt") or datetime.now(timezone.utc).isoformat()
        )
        source_dir = payload.get("sourceDir") or str(output_dir)
        topic_count = payload.get("topicCount") or 0
        if (
            not isinstance(title, str)
            or not isinstance(tagline, str)
            or not isinstance(scope_label, str)
        ):
            return None
        if not isinstance(generated_at, str) or not isinstance(source_dir, str):
            return None
        if not isinstance(topic_count, int):
            topic_count = 0
        return DocsiteSite(
            title=title,
            tagline=tagline,
            scope_label=scope_label,
            generated_at=generated_at,
            source_dir=source_dir,
            topic_count=topic_count,
        )
    except Exception:  # noqa: BLE001
        return None


def _render_search_index(pages: list[DocsitePage]) -> str:
    records: list[dict[str, str]] = []
    for page in pages:
        body = strip_references_section(page.body_markdown)
        body_text = _strip_markdown_for_search(body)
        records.append(
            {
                "title": page.title,
                "slug": page.slug,
                "description": page.description,
                "body": body_text,
                "url": f"/topics/{page.slug}/",
            }
        )
    return json.dumps(records, ensure_ascii=True, indent=2)


def _render_nav_json(groups: list[NavGroup]) -> str:
    payload: dict[str, Any] = {"groups": groups}
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _render_glossary_page(terms: list[GlossaryTerm]) -> str:
    body_lines: list[str] = [
        "This page defines canonical terms used throughout the documentation.",
        "",
    ]
    for entry in terms:
        term = entry.get("term", "").strip()
        definition = entry.get("definition", "").strip()
        pages = entry.get("pages", [])
        if not term or not definition:
            continue
        body_lines.append(f"## {term}")
        body_lines.append("")
        body_lines.append(definition)
        body_lines.append("")
        if isinstance(pages, list) and pages:
            links = ", ".join([f"[{slug}](/topics/{slug}/)" for slug in pages])
            body_lines.append(f"- Pages: {links}")
            body_lines.append("")
    body = "\n".join(body_lines).strip() + "\n"
    return _render_page_frontmatter(
        layout="../layouts/DocLayout.astro",
        title="Glossary",
        description="Definitions of common terms used across the docs.",
        order=None,
        body=body,
        source_path=None,
        scope_label=None,
        references_count=None,
    )


def _render_doc_layout() -> str:
    return load_text("src/layouts/DocLayout.astro")


def _render_global_css() -> str:
    return load_text("src/styles/global.css")


def _render_index_page(
    *,
    site: DocsiteSite,
    pages: list[DocsitePage],
    index: CodeMapperIndex,
    overview_markdown: str | None = None,
) -> str:
    body_parts: list[str] = ["Welcome to the generated AutoDoc documentation site.", ""]

    if overview_markdown and overview_markdown.strip():
        body_parts.append(_ensure_overview_heading(overview_markdown.strip()))
        body_parts.append("")

    body_parts.append("## Topics")
    body_parts.append("")

    ordered_pages = sorted(
        pages,
        key=lambda page: (page.order if page.order is not None else 9999, page.title),
    )
    for idx, page in enumerate(ordered_pages, start=1):
        body_parts.append(f"{idx}. [{page.title}](topics/{page.slug}/)")

    metadata_lines = _render_index_metadata(index)
    if metadata_lines:
        body_parts.extend(
            [
                "",
                '<details class="generation-details">',
                "<summary>Generation Details</summary>",
                "",
                *metadata_lines,
                "",
                "</details>",
            ]
        )

    body = "\n".join(body_parts).strip() + "\n"
    return _render_page_frontmatter(
        layout="../layouts/DocLayout.astro",
        title=site.title,
        description=site.tagline,
        order=None,
        body=body,
    )


def _render_topic_page(page: DocsitePage) -> str:
    return _render_page_frontmatter(
        layout="../../layouts/DocLayout.astro",
        title=page.title,
        description=page.description,
        order=page.order,
        body=page.body_markdown.strip() + "\n",
        source_path=page.source_path,
        scope_label=page.scope_label,
        references_count=page.references_count,
    )


def _render_page_frontmatter(
    *,
    layout: str,
    title: str,
    description: str,
    order: int | None,
    body: str,
    source_path: str | None = None,
    scope_label: str | None = None,
    references_count: int | None = None,
) -> str:
    lines = [
        "---",
        f"layout: {layout}",
        f'title: "{_escape_yaml(title)}"',
        f'description: "{_escape_yaml(description)}"',
    ]
    if order is not None:
        lines.append(f"order: {order}")
    if source_path:
        lines.append(f'sourcePath: "{_escape_yaml(source_path)}"')
    if scope_label:
        lines.append(f'scope: "{_escape_yaml(scope_label)}"')
    if references_count is not None:
        lines.append(f"referencesCount: {references_count}")
    lines.append("---")
    lines.append("")
    lines.append(body.rstrip())
    lines.append("")
    return "\n".join(lines)


def _parse_metadata_block(metadata: str) -> dict[str, object]:
    data: dict[str, object] = {}
    current_top: str | None = None
    current_sub: str | None = None

    for raw in metadata.splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()

        if indent == 2:
            current_top = key
            current_sub = None
            if value:
                data[key] = value
            else:
                data.setdefault(key, {})
            continue

        if indent == 4:
            if current_top == "llm_config":
                llm_config = data.setdefault("llm_config", {})
                if isinstance(llm_config, dict):
                    llm_config[key] = value
            elif current_top == "generation_stats":
                generation_stats = data.setdefault("generation_stats", {})
                if isinstance(generation_stats, dict):
                    if value:
                        generation_stats[key] = value
                        current_sub = None
                    else:
                        generation_stats.setdefault(key, {})
                        current_sub = key
            elif current_top:
                section = data.setdefault(current_top, {})
                if isinstance(section, dict):
                    section[key] = value
            continue

        if indent >= 6 and current_top == "generation_stats" and current_sub:
            generation_stats = data.setdefault("generation_stats", {})
            if isinstance(generation_stats, dict):
                subsection = generation_stats.setdefault(current_sub, {})
                if isinstance(subsection, dict):
                    subsection[key] = value

    return data


def _render_index_metadata(index: CodeMapperIndex) -> list[str]:
    if not index.metadata_block:
        return []

    metadata = _parse_metadata_block(index.metadata_block)
    lines: list[str] = []

    generated_at = metadata.get("generated_at")
    if isinstance(generated_at, str):
        lines.append(f"- Generated at: {generated_at}")

    created_from_sha = metadata.get("created_from_sha")
    if isinstance(created_from_sha, str):
        lines.append(f"- Source SHA: {created_from_sha}")

    llm_config = metadata.get("llm_config")
    if isinstance(llm_config, dict):
        provider = llm_config.get("provider") or llm_config.get("synthesis_provider")
        if provider:
            lines.append(f"- LLM provider: {provider}")
        synthesis_provider = llm_config.get("synthesis_provider")
        if synthesis_provider and synthesis_provider != provider:
            lines.append(f"- Synthesis provider: {synthesis_provider}")
        synthesis_model = llm_config.get("synthesis_model")
        if synthesis_model:
            lines.append(f"- Synthesis model: {synthesis_model}")
        utility_model = llm_config.get("utility_model")
        if utility_model:
            lines.append(f"- Utility model: {utility_model}")
        synth_effort = llm_config.get("codex_reasoning_effort_synthesis")
        if synth_effort:
            lines.append(f"- Synthesis reasoning effort: {synth_effort}")
        util_effort = llm_config.get("codex_reasoning_effort_utility")
        if util_effort:
            lines.append(f"- Utility reasoning effort: {util_effort}")
        hyde_provider = llm_config.get("map_hyde_provider")
        if hyde_provider:
            lines.append(f"- HyDE planning provider: {hyde_provider}")
        hyde_model = llm_config.get("map_hyde_model")
        if hyde_model:
            lines.append(f"- HyDE planning model: {hyde_model}")
        hyde_effort = llm_config.get("map_hyde_reasoning_effort")
        if hyde_effort:
            lines.append(f"- HyDE planning reasoning effort: {hyde_effort}")

    generation_stats = metadata.get("generation_stats")
    if isinstance(generation_stats, dict):
        generator_mode = generation_stats.get("generator_mode")
        if generator_mode:
            lines.append(f"- Generator mode: {generator_mode}")
        comprehensiveness = generation_stats.get(
            "autodoc_comprehensiveness"
        ) or generation_stats.get("code_mapper_comprehensiveness")
        if comprehensiveness:
            lines.append(f"- Comprehensiveness: {comprehensiveness}")
        total_calls = generation_stats.get("total_research_calls")
        if total_calls:
            lines.append(f"- Research calls: {total_calls}")

        files = generation_stats.get("files")
        if isinstance(files, dict):
            referenced = files.get("referenced")
            total = files.get("total_indexed")
            coverage = files.get("coverage")
            basis = files.get("basis")
            if referenced is not None and total is not None:
                detail = f"{referenced} / {total}"
                if coverage:
                    detail = f"{detail} ({coverage})"
                if basis:
                    detail = f"{detail}, basis: {basis}"
                lines.append(f"- Files referenced: {detail}")
            referenced_in_scope = files.get("referenced_in_scope")
            if referenced_in_scope is not None:
                lines.append(f"- Files referenced in scope: {referenced_in_scope}")
            unreferenced = files.get("unreferenced_in_scope")
            if unreferenced is not None:
                lines.append(f"- Files unreferenced in scope: {unreferenced}")
        chunks = generation_stats.get("chunks")
        if isinstance(chunks, dict):
            referenced = chunks.get("referenced")
            total = chunks.get("total_indexed")
            coverage = chunks.get("coverage")
            basis = chunks.get("basis")
            if referenced is not None and total is not None:
                detail = f"{referenced} / {total}"
                if coverage:
                    detail = f"{detail} ({coverage})"
                if basis:
                    detail = f"{detail}, basis: {basis}"
                lines.append(f"- Chunks referenced: {detail}")

    return lines
