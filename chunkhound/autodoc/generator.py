from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

from chunkhound.autodoc.cleanup import _cleanup_with_llm, _minimal_cleanup
from chunkhound.autodoc.ia import _synthesize_homepage_overview, _synthesize_site_ia
from chunkhound.autodoc.index_loader import (
    find_index_file,
    load_topics,
    parse_index_file,
)
from chunkhound.autodoc.markdown_utils import (
    _default_site_title,
    _extract_description,
    _slugify_title,
)
from chunkhound.autodoc.models import (
    CleanupConfig,
    CodeMapperTopic,
    DocsitePage,
    DocsiteResult,
    DocsiteSite,
    GlossaryTerm,
    NavGroup,
)
from chunkhound.autodoc.references import (
    _apply_reference_normalization,
    _select_flat_references_for_cleaned_body,
    extract_sources_block,
    strip_references_section,
)
from chunkhound.autodoc.site_writer import write_astro_site
from chunkhound.autodoc.taint import _normalize_taint
from chunkhound.llm_manager import LLMManager


async def cleanup_topics(
    topics: list[CodeMapperTopic],
    llm_manager: LLMManager | None,
    config: CleanupConfig,
    scope_label: str | None = None,
    log_info: Callable[[str], None] | None = None,
    log_warning: Callable[[str], None] | None = None,
) -> list[DocsitePage]:
    if not topics:
        return []

    if config.mode == "llm" and llm_manager is not None:
        provider = llm_manager.get_synthesis_provider()
        cleaned = await _cleanup_with_llm(
            topics=topics,
            provider=provider,
            config=config,
            log_info=log_info,
            log_warning=log_warning,
        )
    else:
        if config.mode == "llm" and log_warning:
            log_warning(
                "LLM cleanup requested but no LLM provider configured; "
                "falling back to minimal cleanup."
            )
        cleaned = [_minimal_cleanup(topic) for topic in topics]

    pages: list[DocsitePage] = []
    for topic, body in zip(topics, cleaned, strict=False):
        sources_block = extract_sources_block(topic.body_markdown)
        cleaned_body = strip_references_section(body)
        flat_references = (
            _select_flat_references_for_cleaned_body(cleaned_body, sources_block)
            if sources_block
            else []
        )
        normalized_body = _apply_reference_normalization(body, sources_block)
        description = _extract_description(normalized_body)
        slug = _slugify_title(topic.title, topic.order)
        pages.append(
            DocsitePage(
                order=topic.order,
                title=topic.title,
                slug=slug,
                description=description,
                body_markdown=normalized_body,
                source_path=str(topic.source_path),
                scope_label=scope_label,
                references_count=(len(flat_references) if flat_references else None),
            )
        )

    return pages


def _default_site_tagline(*, taint: str, llm_cleanup_active: bool) -> str:
    base = "Approachable documentation generated from AutoDoc output."
    if not llm_cleanup_active:
        return base
    normalized = _normalize_taint(taint)
    if normalized == "technical":
        return "Engineering-focused documentation generated from AutoDoc output."
    if normalized == "end-user":
        return "End-user-friendly documentation generated from AutoDoc output."
    return base


async def generate_docsite(
    *,
    input_dir: Path,
    output_dir: Path,
    llm_manager: LLMManager | None,
    cleanup_config: CleanupConfig,
    site_title: str | None,
    site_tagline: str | None,
    index_patterns: Iterable[str] | None = None,
    log_info: Callable[[str], None] | None = None,
    log_warning: Callable[[str], None] | None = None,
) -> DocsiteResult:
    index_path = find_index_file(
        input_dir,
        patterns=index_patterns,
        log_warning=log_warning,
    )
    index = parse_index_file(index_path)

    if log_info:
        log_info(f"Using AutoDoc index: {index_path}")

    topics, missing = load_topics(
        input_dir=input_dir,
        index=index,
        log_warning=log_warning,
    )

    pages = await cleanup_topics(
        topics=topics,
        llm_manager=llm_manager,
        config=cleanup_config,
        scope_label=index.scope_label,
        log_info=log_info,
        log_warning=log_warning,
    )

    llm_cleanup_active = cleanup_config.mode == "llm" and llm_manager is not None
    tagline = site_tagline or _default_site_tagline(
        taint=cleanup_config.taint,
        llm_cleanup_active=llm_cleanup_active,
    )

    site = DocsiteSite(
        title=site_title or _default_site_title(index.scope_label),
        tagline=tagline,
        scope_label=index.scope_label,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_dir=str(input_dir),
        topic_count=len(pages),
    )

    nav_groups: list[NavGroup] | None = None
    glossary_terms: list[GlossaryTerm] | None = None
    homepage_overview: str | None = None
    if llm_cleanup_active:
        try:
            assert llm_manager is not None
            provider = llm_manager.get_synthesis_provider()
            homepage_overview = await _synthesize_homepage_overview(
                pages=pages,
                provider=provider,
                taint=cleanup_config.taint,
                log_info=log_info,
                log_warning=log_warning,
            )
            nav_groups, glossary_terms = await _synthesize_site_ia(
                pages=pages,
                provider=provider,
                taint=cleanup_config.taint,
                log_info=log_info,
                log_warning=log_warning,
            )
        except Exception as exc:  # noqa: BLE001
            if log_warning:
                log_warning(f"Global IA synthesis failed; skipping. Error: {exc}")

    write_astro_site(
        output_dir=output_dir,
        site=site,
        pages=pages,
        index=index,
        nav_groups=nav_groups,
        glossary_terms=glossary_terms,
        homepage_overview=homepage_overview,
    )

    return DocsiteResult(
        output_dir=output_dir,
        pages=pages,
        index=index,
        missing_topics=missing,
    )
