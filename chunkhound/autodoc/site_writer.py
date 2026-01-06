from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from chunkhound.autodoc.models import (
    CodeMapperIndex,
    DocsitePage,
    DocsiteSite,
    GlossaryTerm,
    NavGroup,
)
from chunkhound.autodoc.site_writer_dirs import (
    ensure_astro_asset_dirs,
    ensure_astro_site_dirs,
)
from chunkhound.autodoc.site_writer_metadata import (
    _render_index_metadata as _metadata_render_index_metadata,
)
from chunkhound.autodoc.site_writer_renderers import (
    _render_astro_config,
    _render_doc_layout,
    _render_favicon_bytes,
    _render_font_bytes,
    _render_font_license_text,
    _render_global_css,
    _render_glossary_page,
    _render_index_page,
    _render_nav_json,
    _render_package_json,
    _render_readme,
    _render_search_index,
    _render_site_json,
    _render_topic_page,
    _render_tsconfig,
)


def write_astro_assets_only(*, output_dir: Path) -> None:
    """
    Update only the Astro runtime assets in an already-generated docsite.

    Intended for iterating on UI/layout without rewriting topic markdown pages.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    asset_dirs = ensure_astro_asset_dirs(output_dir=output_dir)

    site = _load_site_from_existing(output_dir)
    _write_common_configs(output_dir=output_dir, site=site)
    _write_common_assets(
        layouts_dir=asset_dirs.layouts_dir,
        styles_dir=asset_dirs.styles_dir,
        public_dir=asset_dirs.public_dir,
    )


def write_astro_site(
    *,
    output_dir: Path,
    site: DocsiteSite,
    pages: list[DocsitePage],
    index: CodeMapperIndex,
    allow_delete_topics_dir: bool,
    nav_groups: list[NavGroup] | None = None,
    glossary_terms: list[GlossaryTerm] | None = None,
    homepage_overview: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_common_configs(output_dir=output_dir, site=site)

    site_dirs = ensure_astro_site_dirs(
        output_dir=output_dir,
        clean_topics_dir=True,
        allow_delete_topics_dir=allow_delete_topics_dir,
    )

    _write_text(site_dirs.data_dir / "site.json", _render_site_json(site))
    _write_text(site_dirs.data_dir / "search.json", _render_search_index(pages))
    _write_common_assets(
        layouts_dir=site_dirs.layouts_dir,
        styles_dir=site_dirs.styles_dir,
        public_dir=site_dirs.public_dir,
    )

    nav_payload: list[NavGroup]
    if nav_groups:
        nav_payload = nav_groups
    else:
        nav_payload = [NavGroup(title="Topics", slugs=[page.slug for page in pages])]
    _write_text(site_dirs.data_dir / "nav.json", _render_nav_json(nav_payload))

    _write_text(
        site_dirs.pages_dir / "index.md",
        _render_index_page(
            site=site, pages=pages, index=index, overview_markdown=homepage_overview
        ),
    )

    glossary_path = site_dirs.pages_dir / "glossary.md"
    if glossary_terms:
        _write_text(glossary_path, _render_glossary_page(glossary_terms))
    elif glossary_path.exists():
        glossary_path.unlink()

    for page in pages:
        _write_text(
            site_dirs.topics_dir / f"{page.slug}.md",
            _render_topic_page(page),
        )


def _write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _write_bytes(path: Path, content: bytes) -> None:
    path.write_bytes(content)


def _write_common_configs(*, output_dir: Path, site: DocsiteSite | None) -> None:
    _write_text(output_dir / "astro.config.mjs", _render_astro_config())
    _write_text(output_dir / "tsconfig.json", _render_tsconfig())
    if site is not None:
        _write_text(output_dir / "package.json", _render_package_json(site))
        _write_text(output_dir / "README.md", _render_readme(site))


def _write_common_assets(
    *,
    layouts_dir: Path,
    styles_dir: Path,
    public_dir: Path,
) -> None:
    _write_text(layouts_dir / "DocLayout.astro", _render_doc_layout())
    _write_text(styles_dir / "global.css", _render_global_css())
    _write_bytes(public_dir / "favicon.ico", _render_favicon_bytes())

    fonts_dir = public_dir / "fonts"
    licenses_dir = fonts_dir / "licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)

    for filename in (
        "SourceSans3VF-Upright.ttf.woff2",
        "SourceSans3VF-Italic.ttf.woff2",
        "DMSerifDisplay-Regular.ttf",
        "DMSerifDisplay-Italic.ttf",
    ):
        _write_bytes(fonts_dir / filename, _render_font_bytes(filename))

    for filename in (
        "SourceSans3-LICENSE.md",
        "DMSerif-OFL.txt",
        "DMSerif-LICENSE.txt",
    ):
        _write_text(licenses_dir / filename, _render_font_license_text(filename))


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


def _render_index_metadata(index: CodeMapperIndex) -> list[str]:
    return _metadata_render_index_metadata(index)
