"""AutoDoc docsite public API.

This module is intentionally small: it exposes a stable import surface for CLI
and tests, while implementation details live in smaller modules.
"""

from __future__ import annotations

from chunkhound.autodoc.cleanup import (
    _build_cleanup_prompt,
    _build_cleanup_system_prompt,
    _cleanup_with_llm,
    _minimal_cleanup,
    _normalize_llm_output,
    _read_prompt_file,
    _taint_cleanup_system_guidance,
)
from chunkhound.autodoc.generator import cleanup_topics, generate_docsite
from chunkhound.autodoc.index_loader import find_index_file, load_topics, parse_index_file
from chunkhound.autodoc.models import (
    CleanupConfig,
    CodeMapperIndex,
    CodeMapperTopic,
    DocsitePage,
    DocsiteResult,
    DocsiteSite,
    GlossaryTerm,
    IndexTopicEntry,
    NavGroup,
)
from chunkhound.autodoc.references import (
    _apply_reference_normalization,
    _select_flat_references_for_cleaned_body,
    build_references_section,
    extract_sources_block,
    flatten_sources_block,
    strip_references_section,
)
from chunkhound.autodoc.site_writer import (
    _render_astro_config,
    _render_doc_layout,
    _render_index_metadata,
    _render_index_page,
    _render_search_index,
    _render_topic_page,
    write_astro_assets_only,
    write_astro_site,
)

__all__: list[str] = [
    "CleanupConfig",
    "CodeMapperIndex",
    "CodeMapperTopic",
    "DocsitePage",
    "DocsiteResult",
    "DocsiteSite",
    "GlossaryTerm",
    "IndexTopicEntry",
    "NavGroup",
    "_apply_reference_normalization",
    "_build_cleanup_prompt",
    "_build_cleanup_system_prompt",
    "_cleanup_with_llm",
    "_minimal_cleanup",
    "_normalize_llm_output",
    "_read_prompt_file",
    "_render_astro_config",
    "_render_doc_layout",
    "_render_index_metadata",
    "_render_index_page",
    "_render_search_index",
    "_render_topic_page",
    "_select_flat_references_for_cleaned_body",
    "_taint_cleanup_system_guidance",
    "build_references_section",
    "cleanup_topics",
    "extract_sources_block",
    "find_index_file",
    "flatten_sources_block",
    "generate_docsite",
    "load_topics",
    "parse_index_file",
    "strip_references_section",
    "write_astro_assets_only",
    "write_astro_site",
]
