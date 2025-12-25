from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import rmtree
from typing import Callable, Iterable

from chunkhound.code_mapper.utils import safe_scope_label
from chunkhound.interfaces.llm_provider import LLMProvider
from chunkhound.llm_manager import LLMManager

_INDEX_PATTERNS = (
    "*_code_mapper_index.md",
    "*_autodoc_index.md",
)

_TOPIC_LINK_RE = re.compile(
    r"^\s*\d+\.\s+\[(?P<title>.+?)\]\((?P<filename>.+?)\)\s*$"
)
_SOURCES_HEADING_RE = re.compile(r"^##\s+Sources\s*$", re.IGNORECASE)
_REFERENCES_HEADING_RE = re.compile(r"^##\s+References\s*$", re.IGNORECASE)
_TREE_LINE_RE = re.compile(r"^(?P<prefix>.*?)[├└]──\s+(?P<content>.+)$")
_FILE_LINE_RE = re.compile(
    r"^\[(?P<ref>\d+)\]\s+(?P<name>.+?)(?:\s+\((?P<details>.+)\))?$"
)

_FAVICON_ICO_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNg"
    "YAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


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
            "No AutoDoc index file found (expected "
            + ", ".join(pattern_list)
            + ")."
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


async def cleanup_topics(
    topics: list[CodeMapperTopic],
    llm_manager: LLMManager | None,
    config: CleanupConfig,
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
            )
        )

    return pages


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
        log_info=log_info,
        log_warning=log_warning,
    )

    site = DocsiteSite(
        title=site_title or _default_site_title(index.scope_label),
        tagline=site_tagline
        or "Approachable documentation generated from AutoDoc output.",
        scope_label=index.scope_label,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_dir=str(input_dir),
        topic_count=len(pages),
    )

    write_astro_site(
        output_dir=output_dir,
        site=site,
        pages=pages,
        index=index,
    )

    return DocsiteResult(
        output_dir=output_dir,
        pages=pages,
        index=index,
        missing_topics=missing,
    )


def write_astro_site(
    *,
    output_dir: Path,
    site: DocsiteSite,
    pages: list[DocsitePage],
    index: CodeMapperIndex,
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

    _write_text(
        pages_dir / "index.md",
        _render_index_page(site=site, pages=pages, index=index),
    )

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
    return base64.b64decode(_FAVICON_ICO_BASE64)


def _render_package_json(site: DocsiteSite) -> str:
    safe_name = safe_scope_label(site.scope_label).replace("_", "-")
    package_name = f"chunkhound-docs-{safe_name}" if safe_name else "chunkhound-docs"
    return "\n".join(
        [
            "{",
            f'  "name": "{package_name}",',
            '  "version": "0.0.0",',
            '  "type": "module",',
            '  "private": true,',
            '  "scripts": {',
            '    "dev": "astro dev",',
            '    "build": "astro build",',
            '    "preview": "astro preview"',
            "  },",
            '  "dependencies": {',
            '    "astro": "^4.0.0",',
            '    "mermaid": "^10.9.0",',
            '    "remark-gfm": "^4.0.0"',
            "  }",
            "}",
            "",
        ]
    )


def _render_astro_config() -> str:
    return "\n".join(
        [
            "import { defineConfig } from \"astro/config\";",
            "import remarkGfm from \"remark-gfm\";",
            "",
            "export default defineConfig({",
            "  site: 'http://localhost:4321',",
            "  devToolbar: { enabled: false },",
            "  markdown: {",
            "    remarkPlugins: [remarkGfm],",
            "    shikiConfig: {",
            "      theme: 'github-light'",
            "    }",
            "  }",
            "});",
            "",
        ]
    )


def _render_tsconfig() -> str:
    return "\n".join(
        [
            "{",
            '  "extends": "astro/tsconfigs/base",',
            '  "compilerOptions": {',
            '    "types": ["astro/client"]',
            "  }",
            "}",
            "",
        ]
    )


def _render_readme(site: DocsiteSite) -> str:
    return "\n".join(
        [
            f"# {site.title}",
            "",
            "Generated docs site built from AutoDoc output.",
            "",
            "## Quick start",
            "",
            "```bash",
            "npm install",
            "npm run dev",
            "```",
            "",
        ]
    )


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


def _render_doc_layout() -> str:
    return "\n".join(
        [
            "---",
            "import '../styles/global.css';",
            "import site from '../data/site.json';",
            "import searchIndexData from '../data/search.json';",
            "const { title, description } = Astro.props;",
            "const pages = await Astro.glob('../pages/topics/*.md');",
            "const nav = pages",
            "  .map((page) => ({",
            "    title: page.frontmatter.title ?? page.url.split('/').pop(),",
            "    order: page.frontmatter.order ?? 9999,",
            "    url: page.url,",
            "  }))",
            "  .sort((a, b) => a.order - b.order);",
            "const pageTitle = title ? `${title} - ${site.title}` : site.title;",
            "const normalizePath = (value) => {",
            "  if (!value || value === '/') return '/';",
            "  return value.replace(/\\/+$/, '');",
            "};",
            "const current = normalizePath(Astro.url.pathname);",
            "const isActive = (url) => current === normalizePath(url);",
            "const activeTopic = nav.find((item) => isActive(item.url));",
            "const currentIndex = nav.findIndex((item) => isActive(item.url));",
            "const prevTopic = currentIndex > 0 ? nav[currentIndex - 1] : null;",
            "const nextTopic =",
            "  currentIndex >= 0 && currentIndex < nav.length - 1",
            "    ? nav[currentIndex + 1]",
            "    : null;",
            "const breadcrumb = activeTopic ? `Topics / ${activeTopic.title}` : 'Overview';",
            "---",
            "",
            "<!doctype html>",
            "<html lang=\"en\">",
            "  <head>",
            "    <meta charset=\"utf-8\" />",
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
            "    <link rel=\"icon\" href=\"/favicon.ico\" />",
            "    <title>{pageTitle}</title>",
            "  </head>",
            "  <body>",
            "    <div class=\"shell\">",
            "      <header class=\"site-header\">",
            "        <div class=\"header-grid\">",
            "          <button",
            "            class=\"nav-toggle\"",
            "            type=\"button\"",
            "            data-nav-toggle",
            "            aria-controls=\"site-nav\"",
            "            aria-expanded=\"false\"",
            "          >",
            "            <span class=\"nav-toggle-icon\" aria-hidden=\"true\"></span>",
            "            Menu",
            "          </button>",
            "          <div class=\"header-titles\">",
            "            <span class=\"site-kicker\">AutoDoc Docs</span>",
            "            <h1 class=\"site-title\">{site.title}</h1>",
            "            <p class=\"site-tagline\">{site.tagline}</p>",
            "          </div>",
            "          <div class=\"header-search\">",
            "            <label class=\"sr-only\" for=\"global-search\">Search docs</label>",
            "            <input",
            "              id=\"global-search\"",
            "              class=\"search-input\"",
            "              type=\"search\"",
            "              placeholder=\"Search docs\"",
            "              autocomplete=\"off\"",
            "              data-search-input",
            "            />",
            "            <div class=\"search-results\" data-search-results hidden></div>",
            "          </div>",
            "        </div>",
            "      </header>",
            "      <div class=\"site-body\">",
            "        <div class=\"nav-scrim\" data-nav-scrim></div>",
            "        <aside class=\"site-nav\" id=\"site-nav\" aria-label=\"Topics\">",
            "          <div class=\"nav-title\">Topics</div>",
            "          <label class=\"sr-only\" for=\"nav-filter\">Filter topics</label>",
            "          <input",
            "            id=\"nav-filter\"",
            "            class=\"nav-filter\"",
            "            type=\"search\"",
            "            placeholder=\"Filter topics\"",
            "            autocomplete=\"off\"",
            "            data-nav-filter",
            "          />",
            "          <nav>",
            "            <a",
            "              href=\"/\"",
            "              class={isActive('/') ? 'active' : ''}",
            "              aria-current={isActive('/') ? 'page' : undefined}",
            "            >",
            "              Overview",
            "            </a>",
            "            {nav.map((item) => (",
            "              <a",
            "                href={item.url}",
            "                class={isActive(item.url) ? 'active' : ''}",
            "                aria-current={isActive(item.url) ? 'page' : undefined}",
            "                data-topic",
            "              >",
            "                {item.title}",
            "              </a>",
            "            ))}",
            "          </nav>",
            "        </aside>",
            "        <main class=\"site-main\">",
            "          <div class=\"breadcrumbs\">{breadcrumb}</div>",
            "          <div class=\"page-header\">",
            "            {title && <h2>{title}</h2>}",
            "            {description && <p>{description}</p>}",
            "          </div>",
            "          {(activeTopic && (prevTopic || nextTopic)) && (",
            "            <div class=\"page-nav page-nav-top\">",
            "              {prevTopic ? (",
            "                <a class=\"nav-prev\" href={prevTopic.url}>",
            "                  &larr; Previous: {prevTopic.title}",
            "                </a>",
            "              ) : (",
            "                <span></span>",
            "              )}",
            "              {nextTopic ? (",
            "                <a class=\"nav-next\" href={nextTopic.url}>",
            "                  Next: {nextTopic.title} &rarr;",
            "                </a>",
            "              ) : (",
            "                <span></span>",
            "              )}",
            "            </div>",
            "          )}",
            "          <article class=\"page-content\">",
            "            <slot />",
            "          </article>",
            "          {(activeTopic && (prevTopic || nextTopic)) && (",
            "            <div class=\"page-nav page-nav-bottom\">",
            "              {prevTopic ? (",
            "                <a class=\"nav-prev\" href={prevTopic.url}>",
            "                  &larr; Previous: {prevTopic.title}",
            "                </a>",
            "              ) : (",
            "                <span></span>",
            "              )}",
            "              {nextTopic ? (",
            "                <a class=\"nav-next\" href={nextTopic.url}>",
            "                  Next: {nextTopic.title} &rarr;",
            "                </a>",
            "              ) : (",
            "                <span></span>",
            "              )}",
            "            </div>",
            "          )}",
            "        </main>",
            "        <aside class=\"page-toc\" aria-label=\"On this page\">",
            "          <div class=\"toc-title\">On this page</div>",
            "          <nav class=\"toc-list\" data-toc></nav>",
            "        </aside>",
            "      </div>",
            "      <footer class=\"site-footer\">",
            "        <span>",
            "          Generated by{' '}",
            "          <a",
            "            class=\"footer-link\"",
            "            href=\"https://github.com/chunkhound/chunkhound\"",
            "            target=\"_blank\"",
            "            rel=\"noreferrer\"",
            "          >",
            "            ChunkHound",
            "          </a>",
            "        </span>",
            "        <span>Updated {new Date(site.generatedAt).toLocaleString()}</span>",
            "      </footer>",
            "    </div>",
            "    <script",
            "      type=\"application/json\"",
            "      id=\"search-index\"",
            "      set:html={JSON.stringify(searchIndexData)}",
            "    ></script>",
            "    <script type=\"module\">",
            "      const searchIndex = (() => {",
            "        const raw = document.getElementById('search-index');",
            "        if (!raw) return [];",
            "        try {",
            "          return JSON.parse(raw.textContent || '[]');",
            "        } catch (err) {",
            "          return [];",
            "        }",
            "      })();",
            "      const navFilter = document.querySelector('[data-nav-filter]');",
            "      const navLinks = Array.from(",
            "        document.querySelectorAll('.site-nav nav a[data-topic]')",
            "      );",
            "      if (navFilter) {",
            "        navFilter.addEventListener('input', (event) => {",
            "          const query = event.target.value.trim().toLowerCase();",
            "          navLinks.forEach((link) => {",
            "            const text = link.textContent?.toLowerCase() ?? '';",
            "            link.hidden = query.length > 0 && !text.includes(query);",
            "          });",
            "        });",
            "      }",
            "      const searchInput = document.querySelector('[data-search-input]');",
            "      const searchResults = document.querySelector('[data-search-results]');",
            "      const searchLimit = 8;",
            "      const clearResults = () => {",
            "        if (!searchResults) return;",
            "        searchResults.innerHTML = '';",
            "        searchResults.hidden = true;",
            "      };",
            "      const renderResults = (items, query) => {",
            "        if (!searchResults) return;",
            "        searchResults.innerHTML = '';",
            "        if (!items.length) {",
            "          const empty = document.createElement('div');",
            "          empty.className = 'search-empty';",
            "          empty.textContent = `No results for \"${query}\"`;",
            "          searchResults.appendChild(empty);",
            "          searchResults.hidden = false;",
            "          return;",
            "        }",
            "        items.forEach((item) => {",
            "          const entry = document.createElement('a');",
            "          entry.className = 'search-result';",
            "          entry.href = item.url;",
            "          entry.innerHTML = `<span>${item.title}</span><span>${item.snippet}</span>`;",
            "          searchResults.appendChild(entry);",
            "        });",
            "        searchResults.hidden = false;",
            "      };",
            "      const buildSnippet = (text, tokens) => {",
            "        if (!text) return '';",
            "        const lower = text.toLowerCase();",
            "        let index = -1;",
            "        for (const token of tokens) {",
            "          const match = lower.indexOf(token);",
            "          if (match !== -1) {",
            "            index = match;",
            "            break;",
            "          }",
            "        }",
            "        if (index === -1) return text.slice(0, 140);",
            "        const start = Math.max(0, index - 40);",
            "        return text.slice(start, start + 140);",
            "      };",
            "      if (searchInput && searchResults) {",
            "        searchInput.addEventListener('input', (event) => {",
            "          const query = event.target.value.trim().toLowerCase();",
            "          if (!query) {",
            "            clearResults();",
            "            return;",
            "          }",
            "          const tokens = query.split(/\\s+/).filter(Boolean);",
            "          const matches = searchIndex.filter((item) => {",
            "            const haystack = `${item.title} ${item.description} ${item.body}`.toLowerCase();",
            "            return tokens.every((token) => haystack.includes(token));",
            "          });",
            "          const results = matches.slice(0, searchLimit).map((item) => ({",
            "            ...item,",
            "            snippet: buildSnippet(item.body || item.description || '', tokens)",
            "          }));",
            "          renderResults(results, event.target.value.trim());",
            "        });",
            "        document.addEventListener('click', (event) => {",
            "          if (!searchResults.contains(event.target) && event.target !== searchInput) {",
            "            clearResults();",
            "          }",
            "        });",
            "        if (searchInput.value.trim()) {",
            "          searchInput.dispatchEvent(new Event('input', { bubbles: true }));",
            "        } else {",
            "          clearResults();",
            "        }",
            "      }",
            "      const tocContainer = document.querySelector('[data-toc]');",
            "      const tocNodes = Array.from(",
            "        document.querySelectorAll('.page-content h2, .page-content h3, .page-content h4, .page-content p')",
            "      );",
            "      const slugify = (text) =>",
            "        text",
            "          .toLowerCase()",
            "          .trim()",
            "          .replace(/[^a-z0-9]+/g, '-')",
            "          .replace(/^-+|-+$/g, '');",
            "      const usedIds = new Set();",
            "      const ensureUniqueId = (base) => {",
            "        let candidate = base;",
            "        let count = 1;",
            "        while (usedIds.has(candidate) || !candidate) {",
            "          candidate = `${base}-${count}`;",
            "          count += 1;",
            "        }",
            "        usedIds.add(candidate);",
            "        return candidate;",
            "      };",
            "      if (tocContainer && tocNodes.length) {",
            "        tocNodes.forEach((node) => {",
            "          let text = '';",
            "          let level = 'h2';",
            "          let allowCopy = false;",
            "          let copyTarget = node;",
            "          if (node.tagName === 'H2' || node.tagName === 'H3') {",
            "            text = node.textContent?.trim() ?? '';",
            "            level = node.tagName.toLowerCase();",
            "            allowCopy = true;",
            "          } else if (node.tagName === 'H4') {",
            "            text = node.textContent?.trim() ?? '';",
            "            level = 'h4';",
            "            allowCopy = true;",
            "          } else if (node.tagName === 'P') {",
            "            const strong = node.querySelector('strong');",
            "            if (!strong || strong !== node.firstElementChild) {",
            "              return;",
            "            }",
            "            text = strong.textContent?.trim() ?? '';",
            "            if (!text) {",
            "              return;",
            "            }",
            "            text = text.replace(/:$/, '').trim();",
            "            level = 'h3';",
            "            allowCopy = true;",
            "            copyTarget = strong;",
            "          } else {",
            "            return;",
            "          }",
            "          if (!text) {",
            "            return;",
            "          }",
            "          const baseId = node.id || slugify(text);",
            "          const id = ensureUniqueId(baseId);",
            "          node.id = id;",
            "          const tocLink = document.createElement('a');",
            "          tocLink.href = `#${id}`;",
            "          tocLink.textContent = text || 'Section';",
            "          if (level === 'h3') {",
            "            tocLink.className = 'toc-link toc-h3';",
            "          } else if (level === 'h4') {",
            "            tocLink.className = 'toc-link toc-h4';",
            "          } else {",
            "            tocLink.className = 'toc-link';",
            "          }",
            "          tocContainer.appendChild(tocLink);",
            "          if (!allowCopy) {",
            "            return;",
            "          }",
            "          const copyButton = document.createElement('button');",
            "          copyButton.type = 'button';",
            "          copyButton.className = 'heading-link';",
            "          copyButton.setAttribute('aria-label', 'Copy link to section');",
            "          copyButton.textContent = '#';",
            "          copyButton.addEventListener('click', async () => {",
            "            const url = `${window.location.origin}${window.location.pathname}#${id}`;",
            "            try {",
            "              await navigator.clipboard.writeText(url);",
            "              copyButton.textContent = '✓';",
            "              setTimeout(() => {",
            "                copyButton.textContent = '#';",
            "              }, 1200);",
            "            } catch (err) {",
            "              window.location.hash = id;",
            "            }",
            "          });",
            "          copyTarget.appendChild(copyButton);",
            "        });",
            "      } else if (tocContainer) {",
            "        tocContainer.innerHTML = '<span class=\"toc-empty\">No sections</span>';",
            "      }",
            "      const navToggle = document.querySelector('[data-nav-toggle]');",
            "      const navScrim = document.querySelector('[data-nav-scrim]');",
            "      const toggleNav = (open) => {",
            "        document.body.classList.toggle('nav-open', open);",
            "        if (navToggle) {",
            "          navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');",
            "        }",
            "      };",
            "      if (navToggle) {",
            "        navToggle.addEventListener('click', () => {",
            "          toggleNav(!document.body.classList.contains('nav-open'));",
            "        });",
            "      }",
            "      if (navScrim) {",
            "        navScrim.addEventListener('click', () => toggleNav(false));",
            "      }",
            "      document.addEventListener('keydown', (event) => {",
            "        if (event.key === 'Escape') {",
            "          toggleNav(false);",
            "        }",
            "      });",
            "      const loadMermaid = async () => {",
            "        try {",
            "          const mod = await import(",
            "            'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs'",
            "          );",
            "          return mod.default ?? mod;",
            "        } catch (err) {",
            "          console.error('Failed to load Mermaid', err);",
            "          return null;",
            "        }",
            "      };",
            "      const initMermaid = async () => {",
            "        const blocks = Array.from(",
            "          document.querySelectorAll('pre[data-language=\"mermaid\"]')",
            "        );",
            "        if (!blocks.length) return;",
            "        const mermaid = await loadMermaid();",
            "        if (!mermaid) return;",
            "        blocks.forEach((block) => {",
            "          const code = block.querySelector('code');",
            "          const container = document.createElement('div');",
            "          container.className = 'mermaid';",
            "          const raw = code?.textContent || block.textContent || '';",
            "          let sanitized = raw.trim().replace(/\\r\\n/g, '\\n');",
            "          const normalizeLabelNewlines = (input) => {",
            "            let out = '';",
            "            let inLabel = false;",
            "            for (let i = 0; i < input.length; i += 1) {",
            "              const ch = input[i];",
            "              if (ch === '[' || ch === '{') inLabel = true;",
            "              if ((ch === ']' || ch === '}') && inLabel) inLabel = false;",
            "              if (ch === '\\n' && inLabel) {",
            "                out += ' ';",
            "                continue;",
            "              }",
            "              out += ch;",
            "            }",
            "            return out;",
            "          };",
            "          const stripParensInLabels = (input) => {",
            "            let out = '';",
            "            let inLabel = false;",
            "            for (let i = 0; i < input.length; i += 1) {",
            "              const ch = input[i];",
            "              if (ch === '[' || ch === '{') inLabel = true;",
            "              if ((ch === ']' || ch === '}') && inLabel) inLabel = false;",
            "              if (inLabel && (ch === '(' || ch === ')')) {",
            "                continue;",
            "              }",
            "              out += ch;",
            "            }",
            "            return out;",
            "          };",
            "          const convertNodeParens = (input) => {",
            "            return input.replace(/(^|\\s|-->\\s*|--\\s*|==>\\s*)([A-Za-z0-9_]+)\\(([^)]*)\\)/g, '$1$2[$3]');",
            "          };",
            "          sanitized = normalizeLabelNewlines(sanitized);",
            "          sanitized = stripParensInLabels(sanitized);",
            "          sanitized = convertNodeParens(sanitized);",
            "          sanitized = sanitized.replace(/&&/g, 'and');",
            "          sanitized = sanitized.replace(/&/g, 'and');",
            "          sanitized = sanitized.replace(/\\//g, ' or ');",
            "          sanitized = sanitized.replace(/[()]/g, '');",
            "          container.textContent = sanitized.trim();",
            "          block.replaceWith(container);",
            "        });",
            "        const enhanceMermaid = () => {",
            "          const diagrams = Array.from(document.querySelectorAll('.mermaid'));",
            "          diagrams.forEach((diagram) => {",
            "            if (diagram.closest('figure')) {",
            "              return;",
            "            }",
            "            const figure = document.createElement('figure');",
            "            figure.className = 'mermaid-figure';",
            "            figure.setAttribute('role', 'img');",
            "            figure.setAttribute('aria-label', 'Diagram');",
            "            const caption = document.createElement('figcaption');",
            "            caption.textContent = 'Diagram';",
            "            diagram.parentNode?.insertBefore(figure, diagram);",
            "            figure.appendChild(diagram);",
            "            figure.appendChild(caption);",
            "            figure.addEventListener('click', () => {",
            "              const svg = diagram.querySelector('svg');",
            "              if (!svg) return;",
            "              const serializer = new XMLSerializer();",
            "              const svgText = serializer.serializeToString(svg);",
            "              const blob = new Blob([svgText], { type: 'image/svg+xml' });",
            "              const url = URL.createObjectURL(blob);",
            "              window.open(url, '_blank', 'noopener');",
            "              setTimeout(() => URL.revokeObjectURL(url), 1000);",
            "            });",
            "          });",
            "        };",
            "        mermaid.initialize({",
            "          startOnLoad: false,",
            "          theme: 'base',",
            "          themeVariables: {",
            "            primaryColor: '#fff3e9',",
            "            primaryTextColor: '#1c1c1c',",
            "            primaryBorderColor: '#b56a46',",
            "            lineColor: '#8b3d1c',",
            "            secondaryColor: '#f2d6c5',",
            "            tertiaryColor: '#f2efe9'",
            "          }",
            "        });",
            "        mermaid.run({ querySelector: '.mermaid' });",
            "        setTimeout(enhanceMermaid, 0);",
            "      };",
            "      if (document.readyState === 'loading') {",
            "        document.addEventListener('DOMContentLoaded', initMermaid);",
            "      } else {",
            "        initMermaid();",
            "      }",
            "    </script>",
            "  </body>",
            "</html>",
            "",
        ]
    )


def _render_global_css() -> str:
    return "\n".join(
        [
            "@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Source+Sans+3:wght@400;600;700&display=swap');",
            "",
            ":root {",
            "  --font-display: 'DM Serif Display', 'Times New Roman', serif;",
            "  --font-body: 'Source Sans 3', 'Segoe UI', sans-serif;",
            "  --ink: #1c1c1c;",
            "  --muted: #5b5b63;",
            "  --accent: #cc6b3d;",
            "  --accent-strong: #8b3d1c;",
            "  --accent-soft: #f6e2d7;",
            "  --panel: #fff9f3;",
            "  --border: #e7d6c8;",
            "}",
            "",
            "* {",
            "  box-sizing: border-box;",
            "}",
            "",
            "body {",
            "  margin: 0;",
            "  font-family: var(--font-body);",
            "  color: var(--ink);",
            "  background: radial-gradient(circle at top left, #fff1e6 0%, #fefaf7 45%, #f7f2ee 100%);",
            "  min-height: 100vh;",
            "}",
            "",
            "a {",
            "  color: var(--accent-strong);",
            "}",
            "",
            "a:hover {",
            "  color: #6f2f14;",
            "}",
            "",
            "a:focus-visible,",
            "button:focus-visible,",
            "input:focus-visible {",
            "  outline: 2px solid var(--accent);",
            "  outline-offset: 2px;",
            "}",
            "",
            ".sr-only {",
            "  position: absolute;",
            "  width: 1px;",
            "  height: 1px;",
            "  padding: 0;",
            "  margin: -1px;",
            "  overflow: hidden;",
            "  clip: rect(0, 0, 0, 0);",
            "  border: 0;",
            "}",
            "",
            ".shell {",
            "  display: flex;",
            "  flex-direction: column;",
            "  min-height: 100vh;",
            "}",
            "",
            ".site-header {",
            "  padding: 2.5rem 6vw 1.5rem;",
            "  border-bottom: 1px solid var(--border);",
            "  background: linear-gradient(120deg, #fff7ef 0%, #f7ede6 100%);",
            "}",
            "",
            ".header-grid {",
            "  display: grid;",
            "  grid-template-columns: auto 1fr minmax(220px, 320px);",
            "  gap: 1.5rem;",
            "  align-items: center;",
            "}",
            "",
            ".header-titles {",
            "  display: flex;",
            "  flex-direction: column;",
            "  gap: 0.2rem;",
            "}",
            "",
            ".header-search {",
            "  position: relative;",
            "}",
            "",
            ".search-input {",
            "  width: 100%;",
            "  padding: 0.6rem 0.8rem;",
            "  border-radius: 0.6rem;",
            "  border: 1px solid var(--border);",
            "  background: white;",
            "  font-size: 0.95rem;",
            "}",
            "",
            ".search-results {",
            "  position: absolute;",
            "  top: calc(100% + 0.5rem);",
            "  left: 0;",
            "  right: 0;",
            "  background: white;",
            "  border: 1px solid var(--border);",
            "  border-radius: 0.75rem;",
            "  box-shadow: 0 14px 30px rgba(43, 33, 24, 0.12);",
            "  padding: 0.5rem;",
            "  display: flex;",
            "  flex-direction: column;",
            "  gap: 0.35rem;",
            "  max-height: 320px;",
            "  overflow-y: auto;",
            "  z-index: 20;",
            "}",
            "",
            ".search-results[hidden] {",
            "  display: none;",
            "}",
            "",
            ".search-result {",
            "  display: flex;",
            "  flex-direction: column;",
            "  gap: 0.2rem;",
            "  padding: 0.45rem 0.6rem;",
            "  border-radius: 0.5rem;",
            "  text-decoration: none;",
            "  color: var(--ink);",
            "  background: #fff;",
            "}",
            "",
            ".search-result span:first-child {",
            "  font-weight: 600;",
            "}",
            "",
            ".search-result span:last-child {",
            "  font-size: 0.85rem;",
            "  color: var(--muted);",
            "}",
            "",
            ".search-result:hover {",
            "  background: var(--accent-soft);",
            "}",
            "",
            ".search-empty {",
            "  padding: 0.75rem;",
            "  color: var(--muted);",
            "  font-size: 0.9rem;",
            "}",
            "",
            ".site-kicker {",
            "  text-transform: uppercase;",
            "  font-size: 0.75rem;",
            "  letter-spacing: 0.2em;",
            "  color: var(--muted);",
            "}",
            "",
            ".site-title {",
            "  font-family: var(--font-display);",
            "  margin: 0.25rem 0 0.25rem;",
            "  font-size: clamp(2rem, 3vw, 3rem);",
            "}",
            "",
            ".site-tagline {",
            "  margin: 0;",
            "  color: var(--muted);",
            "  max-width: 60ch;",
            "}",
            "",
            ".nav-toggle {",
            "  display: none;",
            "  align-items: center;",
            "  gap: 0.5rem;",
            "  border-radius: 0.6rem;",
            "  border: 1px solid var(--border);",
            "  background: white;",
            "  padding: 0.4rem 0.7rem;",
            "  font-size: 0.85rem;",
            "  font-weight: 600;",
            "  color: var(--ink);",
            "}",
            "",
            ".nav-toggle-icon {",
            "  width: 1.2rem;",
            "  height: 2px;",
            "  background: var(--ink);",
            "  position: relative;",
            "  display: inline-block;",
            "}",
            "",
            ".nav-toggle-icon::before,",
            ".nav-toggle-icon::after {",
            "  content: '';",
            "  position: absolute;",
            "  left: 0;",
            "  width: 1.2rem;",
            "  height: 2px;",
            "  background: var(--ink);",
            "}",
            "",
            ".nav-toggle-icon::before {",
            "  top: -6px;",
            "}",
            "",
            ".nav-toggle-icon::after {",
            "  top: 6px;",
            "}",
            "",
            ".site-body {",
            "  display: grid;",
            "  grid-template-columns: minmax(220px, 260px) minmax(0, 1fr) minmax(200px, 240px);",
            "  gap: 2rem;",
            "  padding: 2rem 6vw 3rem;",
            "  flex: 1;",
            "}",
            "",
            ".nav-scrim {",
            "  display: none;",
            "}",
            "",
            ".site-nav {",
            "  background: var(--panel);",
            "  border: 1px solid var(--border);",
            "  border-radius: 1rem;",
            "  padding: 1.5rem;",
            "  position: sticky;",
            "  top: 1.5rem;",
            "  align-self: start;",
            "  height: fit-content;",
            "  max-height: calc(100vh - 3rem);",
            "  overflow: auto;",
            "}",
            "",
            ".nav-title {",
            "  font-weight: 600;",
            "  font-size: 0.85rem;",
            "  text-transform: uppercase;",
            "  letter-spacing: 0.12em;",
            "  color: var(--muted);",
            "  margin-bottom: 1rem;",
            "}",
            "",
            ".nav-filter {",
            "  width: 100%;",
            "  padding: 0.5rem 0.65rem;",
            "  border-radius: 0.6rem;",
            "  border: 1px solid var(--border);",
            "  background: white;",
            "  margin-bottom: 0.75rem;",
            "  font-size: 0.9rem;",
            "}",
            "",
            ".site-nav nav {",
            "  display: flex;",
            "  flex-direction: column;",
            "  gap: 0.6rem;",
            "}",
            "",
            ".site-nav a {",
            "  color: var(--ink);",
            "  text-decoration: none;",
            "  font-size: 0.95rem;",
            "  padding: 0.35rem 0.5rem;",
            "  border-radius: 0.5rem;",
            "  border-left: 3px solid transparent;",
            "}",
            "",
            ".site-nav a.active,",
            ".site-nav a:hover {",
            "  background: var(--accent-soft);",
            "  color: #702f12;",
            "  border-left-color: var(--accent);",
            "}",
            "",
            ".site-main {",
            "  background: white;",
            "  border-radius: 1.25rem;",
            "  padding: 2rem 2.5rem;",
            "  box-shadow: 0 20px 40px rgba(43, 33, 24, 0.08);",
            "  animation: rise 0.6s ease-out;",
            "}",
            "",
            ".breadcrumbs {",
            "  font-size: 0.85rem;",
            "  color: var(--muted);",
            "  margin-bottom: 0.75rem;",
            "}",
            "",
            ".page-header h2 {",
            "  font-family: var(--font-display);",
            "  margin: 0 0 0.4rem;",
            "  font-size: clamp(1.8rem, 2.6vw, 2.6rem);",
            "}",
            "",
            ".page-header p {",
            "  margin: 0 0 1.5rem;",
            "  color: var(--muted);",
            "}",
            "",
            ".page-content h2 {",
            "  margin-top: 2rem;",
            "  font-family: var(--font-display);",
            "  line-height: 1.2;",
            "  display: block;",
            "}",
            "",
            ".page-content h3 {",
            "  margin-top: 1.5rem;",
            "  line-height: 1.2;",
            "  display: block;",
            "}",
            "",
            ".page-content h4 {",
            "  margin-top: 1.25rem;",
            "  line-height: 1.2;",
            "  display: block;",
            "}",
            "",
            ".page-content .heading-link {",
            "  margin-left: 0.5rem;",
            "  opacity: 0;",
            "  border: 0;",
            "  background: none;",
            "  color: var(--muted);",
            "  border-radius: 0;",
            "  padding: 0;",
            "  min-width: 0;",
            "  width: auto;",
            "  height: auto;",
            "  font-size: 0.6em;",
            "  font-weight: 500;",
            "  line-height: 1;",
            "  cursor: pointer;",
            "  display: inline;",
            "  align-self: center;",
            "  vertical-align: middle;",
            "  flex-shrink: 0;",
            "}",
            "",
            ".page-content h2:hover .heading-link,",
            ".page-content h3:hover .heading-link,",
            ".page-content h4:hover .heading-link,",
            ".page-content p:hover .heading-link,",
            ".page-content .heading-link:focus-visible {",
            "  opacity: 1;",
            "  background: var(--accent-soft);",
            "  color: var(--accent-strong);",
            "  border-color: var(--border);",
            "}",
            "",
            ".page-content code {",
            "  background: #f4efe9;",
            "  padding: 0.1rem 0.3rem;",
            "  border-radius: 0.3rem;",
            "}",
            "",
            ".page-content pre {",
            "  background: #faf5f1;",
            "  padding: 1rem;",
            "  border-radius: 0.8rem;",
            "  overflow-x: auto;",
            "  line-height: 1.45;",
            "  tab-size: 2;",
            "}",
            "",
            ".page-content pre code {",
            "  background: transparent;",
            "  padding: 0;",
            "  border-radius: 0;",
            "  white-space: pre;",
            "  display: block;",
            "}",
            "",
            ".page-content blockquote {",
            "  margin: 1.5rem 0;",
            "  padding: 1rem 1.2rem;",
            "  border-left: 4px solid var(--accent);",
            "  background: #fff4ea;",
            "  border-radius: 0.6rem;",
            "  box-shadow: 0 10px 20px rgba(204, 107, 61, 0.12);",
            "}",
            "",
            ".page-content .generation-details {",
            "  margin-top: 1.5rem;",
            "  padding: 0.75rem 1rem;",
            "  border: 1px solid var(--border);",
            "  border-radius: 0.75rem;",
            "  background: #fffdfb;",
            "}",
            "",
            ".page-content .generation-details summary {",
            "  cursor: pointer;",
            "  font-weight: 600;",
            "  color: var(--accent-strong);",
            "}",
            "",
            ".page-content table {",
            "  width: 100%;",
            "  border-collapse: separate;",
            "  border-spacing: 0;",
            "  margin: 1.5rem 0;",
            "  font-size: 0.95rem;",
            "  border: 1px solid var(--border);",
            "  border-radius: 0.9rem;",
            "  background: #fffdfb;",
            "  box-shadow: 0 12px 24px rgba(43, 33, 24, 0.08);",
            "  overflow: hidden;",
            "}",
            "",
            ".page-content table th,",
            ".page-content table td {",
            "  border: 0;",
            "  border-bottom: 1px solid var(--border);",
            "  border-right: 1px solid var(--border);",
            "  padding: 0.75rem 0.95rem;",
            "  text-align: left;",
            "  vertical-align: top;",
            "}",
            "",
            ".page-content table th {",
            "  background: #fff0e4;",
            "  font-weight: 600;",
            "  border-bottom: 1px solid var(--border);",
            "}",
            "",
            ".page-content table tr > :last-child {",
            "  border-right: 0;",
            "}",
            "",
            ".page-content table tbody tr:nth-child(even) td {",
            "  background: #fff9f3;",
            "}",
            "",
            ".page-content table tbody tr:hover td {",
            "  background: #fff0e4;",
            "}",
            "",
            ".page-content table tbody tr:last-child td {",
            "  border-bottom: 0;",
            "}",
            "",
            ".page-content table td code {",
            "  white-space: normal;",
            "  overflow-wrap: anywhere;",
            "}",
            "",
            ".page-content .mermaid-figure {",
            "  background: #fff9f4;",
            "  border: 1px solid var(--border);",
            "  border-radius: 0.9rem;",
            "  padding: 1rem;",
            "  margin: 1.5rem 0;",
            "  box-shadow: 0 16px 30px rgba(204, 107, 61, 0.12);",
            "  cursor: zoom-in;",
            "}",
            "",
            ".page-content .mermaid-figure figcaption {",
            "  font-size: 0.8rem;",
            "  color: var(--muted);",
            "  margin-top: 0.5rem;",
            "  text-align: center;",
            "}",
            "",
            ".page-content .mermaid {",
            "  background: transparent;",
            "}",
            "",
            ".page-toc {",
            "  background: var(--panel);",
            "  border: 1px solid var(--border);",
            "  border-radius: 1rem;",
            "  padding: 1.25rem;",
            "  position: sticky;",
            "  top: 1.5rem;",
            "  align-self: start;",
            "  height: fit-content;",
            "}",
            "",
            ".toc-title {",
            "  font-weight: 600;",
            "  font-size: 0.85rem;",
            "  text-transform: uppercase;",
            "  letter-spacing: 0.12em;",
            "  color: var(--muted);",
            "  margin-bottom: 0.75rem;",
            "}",
            "",
            ".toc-list {",
            "  display: flex;",
            "  flex-direction: column;",
            "  gap: 0.5rem;",
            "}",
            "",
            ".toc-link {",
            "  color: var(--ink);",
            "  text-decoration: none;",
            "  font-size: 0.9rem;",
            "  display: block;",
            "  line-height: 1.35;",
            "  white-space: normal;",
            "  overflow-wrap: break-word;",
            "  word-break: normal;",
            "  hyphens: auto;",
            "}",
            "",
            ".toc-link.toc-h3 {",
            "  padding-left: 0.75rem;",
            "  font-size: 0.85rem;",
            "  color: var(--muted);",
            "}",
            "",
            ".toc-link.toc-h4 {",
            "  padding-left: 1.5rem;",
            "  font-size: 0.82rem;",
            "  color: var(--muted);",
            "}",
            "",
            ".toc-empty {",
            "  color: var(--muted);",
            "  font-size: 0.85rem;",
            "}",
            "",
            ".page-nav {",
            "  display: flex;",
            "  justify-content: space-between;",
            "  gap: 1rem;",
            "  margin: 1rem 0;",
            "  font-size: 0.9rem;",
            "}",
            "",
            ".page-nav a {",
            "  text-decoration: none;",
            "  padding: 0.4rem 0.6rem;",
            "  border-radius: 0.6rem;",
            "  border: 1px solid var(--border);",
            "  background: #fff7f0;",
            "}",
            "",
            ".page-nav a:hover {",
            "  background: var(--accent-soft);",
            "}",
            "",
            ".site-footer {",
            "  border-top: 1px solid var(--border);",
            "  padding: 1.25rem 6vw;",
            "  display: flex;",
            "  justify-content: space-between;",
            "  font-size: 0.85rem;",
            "  color: var(--muted);",
            "  background: #fff9f3;",
            "}",
            "",
            ".site-footer a {",
            "  color: var(--accent-strong);",
            "  text-decoration: none;",
            "}",
            "",
            ".site-footer a:hover {",
            "  text-decoration: underline;",
            "}",
            "",
            "@keyframes rise {",
            "  from {",
            "    opacity: 0;",
            "    transform: translateY(16px);",
            "  }",
            "  to {",
            "    opacity: 1;",
            "    transform: translateY(0);",
            "  }",
            "}",
            "",
            "@media (max-width: 1100px) {",
            "  .site-body {",
            "    grid-template-columns: minmax(220px, 260px) 1fr;",
            "  }",
            "  .page-toc {",
            "    display: none;",
            "  }",
            "}",
            "",
            "@media (max-width: 900px) {",
            "  .header-grid {",
            "    grid-template-columns: auto 1fr;",
            "  }",
            "  .header-search {",
            "    grid-column: 1 / -1;",
            "  }",
            "  .nav-toggle {",
            "    display: inline-flex;",
            "  }",
            "  .site-body {",
            "    grid-template-columns: 1fr;",
            "  }",
            "  .site-nav {",
            "    position: fixed;",
            "    inset: 0 auto 0 0;",
            "    width: min(80vw, 320px);",
            "    transform: translateX(-110%);",
            "    transition: transform 0.25s ease;",
            "    z-index: 30;",
            "  }",
            "  .site-main {",
            "    padding: 1.5rem;",
            "  }",
            "  .nav-scrim {",
            "    display: block;",
            "    position: fixed;",
            "    inset: 0;",
            "    background: rgba(28, 28, 28, 0.35);",
            "    opacity: 0;",
            "    pointer-events: none;",
            "    transition: opacity 0.2s ease;",
            "    z-index: 20;",
            "  }",
            "  body.nav-open {",
            "    overflow: hidden;",
            "  }",
            "  body.nav-open .site-nav {",
            "    transform: translateX(0);",
            "  }",
            "  body.nav-open .nav-scrim {",
            "    opacity: 1;",
            "    pointer-events: auto;",
            "  }",
            "}",
            "",
        ]
    )


def _render_index_page(
    *,
    site: DocsiteSite,
    pages: list[DocsitePage],
    index: CodeMapperIndex,
) -> str:
    topic_lines = [
        f"{page.order}. [{page.title}](topics/{page.slug}/)"
        for page in pages
    ]

    metadata_lines = _render_index_metadata(index)

    body_parts = [
        "Welcome to the generated AutoDoc documentation site.",
        "",
        "## Topics",
        "",
        *topic_lines,
    ]

    if metadata_lines:
        body_parts.extend(
            [
                "",
                "<details class=\"generation-details\">",
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
    )


def _render_page_frontmatter(
    *,
    layout: str,
    title: str,
    description: str,
    order: int | None,
    body: str,
) -> str:
    lines = [
        "---",
        f"layout: {layout}",
        f"title: \"{_escape_yaml(title)}\"",
        f"description: \"{_escape_yaml(description)}\"",
    ]
    if order is not None:
        lines.append(f"order: {order}")
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
        assembly_provider = llm_config.get("assembly_synthesis_provider")
        if assembly_provider:
            lines.append(f"- Assembly provider: {assembly_provider}")
        assembly_model = llm_config.get("assembly_synthesis_model")
        if assembly_model:
            lines.append(f"- Assembly model: {assembly_model}")
        assembly_effort = llm_config.get("assembly_reasoning_effort")
        if assembly_effort:
            lines.append(f"- Assembly reasoning effort: {assembly_effort}")

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


async def _cleanup_with_llm(
    *,
    topics: list[CodeMapperTopic],
    provider: LLMProvider,
    config: CleanupConfig,
    log_info: Callable[[str], None] | None,
    log_warning: Callable[[str], None] | None,
) -> list[str]:
    system_prompt = (
        "You are a senior technical writer polishing engineering documentation. "
        "Make the writing approachable without losing precision."
    )

    prompts = [
        _build_cleanup_prompt(topic.title, topic.body_markdown)
        for topic in topics
    ]

    responses: list[str] = []
    for batch in _chunked(prompts, config.batch_size):
        if log_info:
            log_info(f"Running cleanup batch with {len(batch)} topic(s).")
        try:
            batch_responses = await provider.batch_complete(
                batch,
                system=system_prompt,
                max_completion_tokens=config.max_completion_tokens,
            )
        except Exception as exc:
            if log_warning:
                log_warning(
                    "LLM cleanup batch failed; falling back to minimal cleanup. "
                    f"Error: {exc}"
                )
            return [_minimal_cleanup(topic) for topic in topics]

        responses.extend([resp.content.strip() for resp in batch_responses])

    cleaned: list[str] = []
    for topic, response in zip(topics, responses, strict=False):
        if not response:
            cleaned.append(_minimal_cleanup(topic))
        else:
            cleaned.append(_normalize_llm_output(response))
    return cleaned


def _build_cleanup_prompt(title: str, body: str) -> str:
    return "\n".join(
        [
            "Rewrite the documentation section below as a polished, friendly doc page.",
            "Requirements:",
            "- Do NOT add new facts or speculation.",
            "- Keep citations like [1] exactly as-is.",
            "- Preserve code identifiers and inline code formatting.",
            "- Start with a short \"Overview\" section.",
            "- Use level-2 headings (##). Do NOT include a level-1 heading.",
            "- Do NOT include recommendations, follow-ups, or next steps.",
            "- Use visuals when clearly helpful: include a Mermaid diagram in a ```mermaid code fence and/or a table/callout to clarify complex flows. Skip visuals if they would be redundant.",
            "- If the content includes a \"Sources\" section, rename it to \"References\".",
            "- Remove duplicate title lines or redundant bold title repeats.",
            "- Keep the length roughly similar to the input.",
            "- Mermaid rules: keep node labels on a single line (no raw newlines inside [brackets] or {diamonds}). Prefer simple labels and avoid unusual punctuation.",
            "Mermaid examples (copy the style, not the content):",
            "```mermaid",
            "flowchart TD",
            "  A[CLI or Env Inputs] --> B[Config Resolution]",
            "  B --> C[Service Registry]",
            "  B --> D{Embeddings required?}",
            "  D -- yes --> E[Embedding Provider]",
            "  D -- no --> F[Search Only Mode]",
            "```",
            "```mermaid",
            "flowchart LR",
            "  A[Request] --> B[Validate]",
            "  B --> C{Valid?}",
            "  C -- yes --> D[Execute]",
            "  C -- no --> E[Error]",
            "```",
            "```mermaid",
            "flowchart TD",
            "  A[CLI entry] --> B[Config builder]",
            "  B --> C[validate_for_command]",
            "  C --> D{Config valid?}",
            "  D -- yes --> E[MCP server start]",
            "  D -- no --> F[CLI error]",
            "```",
            "",
            f"Title: {title}",
            "",
            "Input markdown:",
            body.strip(),
        ]
    )


def _normalize_llm_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = _strip_first_heading(cleaned)
    cleaned = _ensure_overview_heading(cleaned)
    return cleaned.strip()


def _minimal_cleanup(topic: CodeMapperTopic) -> str:
    return _ensure_overview_heading(topic.body_markdown.strip())


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
    lines = ["## References", "", *flat_items]
    return "\n".join(lines).strip()


def _apply_reference_normalization(body: str, sources_block: str | None) -> str:
    cleaned = strip_references_section(body)
    if not sources_block:
        return cleaned.strip()
    flat_items = flatten_sources_block(sources_block)
    references_block = build_references_section(flat_items)
    if not references_block:
        return cleaned.strip()
    if cleaned.strip():
        return cleaned.strip() + "\n\n" + references_block
    return references_block


def _strip_markdown_for_search(markdown: str) -> str:
    text = re.sub(r"```.*?```", " ", markdown, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_description(markdown: str, limit: int = 180) -> str:
    paragraph_lines: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            if paragraph_lines:
                break
            continue
        if stripped.startswith("#"):
            continue
        if stripped.lower() in {"overview", "**overview**"}:
            continue
        if _is_list_item(stripped):
            paragraph_lines.append(_strip_list_marker(stripped))
            break
        paragraph_lines.append(stripped)
    paragraph = " ".join(paragraph_lines).strip()
    if not paragraph:
        return "Key workflows and responsibilities summarized for this topic."
    paragraph = re.sub(r"\s+", " ", paragraph)
    if len(paragraph) <= limit:
        return paragraph
    return paragraph[: limit - 3].rstrip() + "..."


def _strip_metadata_block(text: str) -> tuple[str | None, str]:
    stripped = text.lstrip()
    if not stripped.startswith("<!--"):
        return None, text
    start = text.find("<!--")
    end = text.find("-->", start + 4)
    if end == -1:
        return None, text
    metadata = text[start + 4 : end].strip()
    remainder = text[end + 3 :]
    return metadata, remainder.lstrip("\n")


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped
    return None


def _heading_text(heading: str) -> str:
    return heading.lstrip("# ").strip()


def _strip_first_heading(text: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("# "):
            return "\n".join(lines[idx + 1 :]).lstrip()
    return text




def _ensure_overview_heading(text: str) -> str:
    lines = text.splitlines()
    first_content_idx: int | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            return text
        if stripped.startswith("**Overview**"):
            remainder = stripped[len("**Overview**") :].strip()
            remainder = remainder.lstrip("-").lstrip(":").strip()
            new_lines = lines[:idx] + ["## Overview"]
            if remainder:
                new_lines.append(remainder)
            new_lines.extend(lines[idx + 1 :])
            return "\n".join(new_lines).lstrip()
        lowered = stripped.lower()
        if lowered.startswith("overview"):
            remainder = stripped[len("overview") :].strip()
            remainder = remainder.lstrip("-").lstrip(":").strip()
            new_lines = lines[:idx] + ["## Overview"]
            if remainder:
                new_lines.append(remainder)
            new_lines.extend(lines[idx + 1 :])
            return "\n".join(new_lines).lstrip()
        first_content_idx = idx
        break
    if first_content_idx is None:
        return text
    new_lines = lines[:first_content_idx] + ["## Overview", ""] + lines[
        first_content_idx:
    ]
    return "\n".join(new_lines).lstrip()


def _is_list_item(line: str) -> bool:
    if line.startswith(("-", "*", "+")):
        return True
    return bool(re.match(r"\\d+\\.\\s+", line))


def _strip_list_marker(line: str) -> str:
    if line.startswith(("-", "*", "+")):
        return line[1:].strip()
    return re.sub(r"^\\d+\\.\\s+", "", line).strip()


def _remove_duplicate_title_line(text: str, title: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    removed = False
    for line in lines:
        stripped = line.strip()
        if not stripped and not cleaned:
            continue
        if not removed and stripped in {
            f"**{title}**",
            f"**{title.rstrip()}**",
        }:
            removed = True
            continue
        cleaned.append(line)
    return "\n".join(cleaned).lstrip()


def _scope_from_heading(heading: str) -> str:
    cleaned = _heading_text(heading)
    if " for " in cleaned:
        return cleaned.split(" for ", 1)[1].strip()
    return cleaned


def _default_site_title(scope_label: str) -> str:
    if scope_label and scope_label != "/":
        return f"AutoDoc - {scope_label}"
    return "AutoDoc Documentation"


def _slugify_title(title: str, order: int) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", title.strip().lower()).strip("-")
    if not text:
        text = "topic"
    return f"{order:02d}-{text}"


def _escape_yaml(value: str) -> str:
    return value.replace("\"", "\\\"")


def _escape_json(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', "\\\"")


def _chunked(items: Iterable[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [list(items)]
    batch: list[str] = []
    batches: list[list[str]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches
