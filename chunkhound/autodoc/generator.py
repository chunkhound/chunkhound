from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

from chunkhound.autodoc.cleanup import _cleanup_with_llm
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
    CodeMapperIndex,
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
from chunkhound.core.audience import normalize_audience
from chunkhound.llm_manager import LLMManager


def _now_isoformat() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    cleaned = await _cleanup_topic_bodies(
        topics=topics,
        llm_manager=llm_manager,
        config=config,
        log_info=log_info,
        log_warning=log_warning,
    )

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


async def _cleanup_topic_bodies(
    *,
    topics: list[CodeMapperTopic],
    llm_manager: LLMManager | None,
    config: CleanupConfig,
    log_info: Callable[[str], None] | None,
    log_warning: Callable[[str], None] | None,
) -> list[str]:
    if config.mode != "llm":
        raise ValueError(
            f"Unsupported AutoDoc cleanup mode: {config.mode!r}. Expected: 'llm'."
        )

    if llm_manager is None:
        raise RuntimeError(
            "AutoDoc cleanup requires an LLM provider, but none is configured."
        )

    provider = llm_manager.get_synthesis_provider()
    return await _cleanup_with_llm(
        topics=topics,
        provider=provider,
        config=config,
        log_info=log_info,
        log_warning=log_warning,
    )


def _default_site_tagline(*, audience: str, llm_cleanup_active: bool) -> str:
    base = "Approachable documentation generated from AutoDoc output."
    if not llm_cleanup_active:
        return base
    normalized = normalize_audience(audience)
    if normalized == "technical":
        return "Engineering-focused documentation generated from AutoDoc output."
    if normalized == "end-user":
        return "End-user-friendly documentation generated from AutoDoc output."
    return base


def _build_site(
    *,
    index: CodeMapperIndex,
    input_dir: Path,
    pages: list[DocsitePage],
    cleanup_config: CleanupConfig,
    llm_cleanup_active: bool,
    site_title: str | None,
    site_tagline: str | None,
) -> DocsiteSite:
    tagline = site_tagline or _default_site_tagline(
        audience=cleanup_config.audience,
        llm_cleanup_active=llm_cleanup_active,
    )
    return DocsiteSite(
        title=site_title or _default_site_title(index.scope_label),
        tagline=tagline,
        scope_label=index.scope_label,
        generated_at=_now_isoformat(),
        source_dir=str(input_dir),
        topic_count=len(pages),
    )


async def _maybe_synthesize_global_ia(
    *,
    llm_manager: LLMManager | None,
    pages: list[DocsitePage],
    audience: str,
    log_info: Callable[[str], None] | None,
    log_warning: Callable[[str], None] | None,
) -> tuple[list[NavGroup] | None, list[GlossaryTerm] | None, str | None]:
    if llm_manager is None:
        return None, None, None

    try:
        provider = llm_manager.get_synthesis_provider()
    except Exception as exc:  # noqa: BLE001
        if log_warning:
            log_warning(
                "Global IA synthesis provider unavailable; skipping. "
                f"Error: {exc}"
            )
        return None, None, None

    homepage_overview: str | None = None
    try:
        homepage_overview = await _synthesize_homepage_overview(
            pages=pages,
            provider=provider,
            audience=audience,
            log_info=log_info,
            log_warning=log_warning,
        )
    except Exception as exc:  # noqa: BLE001
        if log_warning:
            log_warning(
                f"Homepage overview synthesis failed; skipping overview. Error: {exc}"
            )

    nav_groups: list[NavGroup] | None = None
    glossary_terms: list[GlossaryTerm] | None = None
    try:
        nav_groups, glossary_terms = await _synthesize_site_ia(
            pages=pages,
            provider=provider,
            audience=audience,
            log_info=log_info,
            log_warning=log_warning,
        )
    except Exception as exc:  # noqa: BLE001
        if log_warning:
            log_warning(
                "Global navigation/glossary synthesis failed; skipping. "
                f"Error: {exc}"
            )

    return nav_groups, glossary_terms, homepage_overview


async def generate_docsite(
    *,
    input_dir: Path,
    output_dir: Path,
    llm_manager: LLMManager | None,
    cleanup_config: CleanupConfig,
    site_title: str | None,
    site_tagline: str | None,
    allow_delete_topics_dir: bool = False,
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
    site = _build_site(
        index=index,
        input_dir=input_dir,
        pages=pages,
        cleanup_config=cleanup_config,
        llm_cleanup_active=llm_cleanup_active,
        site_title=site_title,
        site_tagline=site_tagline,
    )

    nav_groups: list[NavGroup] | None = None
    glossary_terms: list[GlossaryTerm] | None = None
    homepage_overview: str | None = None
    if llm_cleanup_active:
        (
            nav_groups,
            glossary_terms,
            homepage_overview,
        ) = await _maybe_synthesize_global_ia(
            llm_manager=llm_manager,
            pages=pages,
            audience=cleanup_config.audience,
            log_info=log_info,
            log_warning=log_warning,
        )

    write_astro_site(
        output_dir=output_dir,
        site=site,
        pages=pages,
        index=index,
        allow_delete_topics_dir=allow_delete_topics_dir,
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
