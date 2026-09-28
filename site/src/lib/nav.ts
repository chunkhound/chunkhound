/**
 * Canonical site-navigation model — the single source of truth for:
 *  - every documentation guide (DOCS_PAGES): the docs sidebar, the docs
 *    landing page, and the top-nav shortcuts all derive from this list;
 *  - the marketing top nav (NAV_TABS).
 *
 * Nav philosophy: the top nav is a global nav. It links to the few real
 * destinations that matter — never to same-page scroll anchors, which merely
 * duplicate scrolling and cannot reflect reading position. The full guide
 * index belongs in the docs sidebar, not the top nav.
 */
export type DocsPage = {
  title: string;
  href: string;
  /** Phosphor web-component name; rendered via PhosphorIcon. */
  icon: string;
  /** Sidebar/hub blurb. */
  description: string;
  /**
   * Longer, search-optimized meta description for the page's `<head>`. Only set
   * it when the short `description` would undersell the page in search results;
   * otherwise DocsLayout falls back to `description`, so the sidebar/hub/llms.txt
   * blurb stays the single source for short meta copy.
   */
  seoDescription?: string;
  /** Feature this guide as a top-nav destination on marketing pages. */
  inTopNav?: boolean;
};

/** Documentation guides, priority-ordered. */
export const DOCS_PAGES: DocsPage[] = [
  {
    title: 'Getting Started',
    href: '/docs/getting-started/',
    icon: 'ph-rocket-launch',
    description:
      'Install ChunkHound, choose a setup route, index your code, and verify search end to end.',
    inTopNav: true,
  },
  {
    title: 'Architecture',
    href: '/docs/architecture/',
    icon: 'ph-tree-structure',
    description:
      'Why the engine is shaped this way: structural chunking, exact and semantic retrieval, and cited synthesis over a local index.',
    seoDescription:
      'Why the ChunkHound engine is shaped this way: structural chunking, exact and semantic retrieval, an adaptive rerank cutoff, multi-hop exploration, and map-reduce synthesis, over a local code index.',
    inTopNav: true,
  },
  {
    title: 'Configuration',
    href: '/docs/configuration/',
    icon: 'ph-sliders-horizontal',
    description:
      'Configure embedding providers, database backends, and indexing behavior.',
  },
  {
    title: 'CLI Reference',
    href: '/docs/cli-reference/',
    icon: 'ph-terminal-window',
    description: 'Complete reference for all ChunkHound CLI commands and flags.',
  },
  {
    title: 'Changelog',
    href: '/docs/changelog/',
    icon: 'ph-clock-counter-clockwise',
    description: 'Release history and breaking changes for ChunkHound.',
  },
  {
    title: 'Contributing',
    href: '/docs/contributing/',
    icon: 'ph-git-pull-request',
    description:
      'How to contribute to ChunkHound — a 100% AI-generated codebase with AI-agent code review.',
  },
];

/** Trailing-slash-insensitive compare: real destination pages mark active
 * on an exact pathname match. Prefix matching would wrongly light one tab
 * on a sibling page; scrollspy is intentionally absent — the nav links to
 * pages, not anchors. Shared by Nav.astro and DocsNav.astro. */
export function normalizePath(path: string): string {
  return path.replace(/\/+$/, "");
}

export function isActivePath(currentPath: string, href: string): boolean {
  return normalizePath(currentPath) === normalizePath(href);
}

/** Docs hub landing page metadata (/docs/). */
export const DOCS_HOME = {
  title: 'Documentation',
  href: '/docs/',
  description:
    'Install, configure, operate, and understand ChunkHound — from getting-started guides to architecture, CLI reference, and configuration.',
};

export type NavTab = {
  label: string;
  href: string;
};

/**
 * Top nav = the docs hub plus the critical guides, highest-value first.
 * A tab is active on an exact pathname match to its href (see Nav.astro).
 */
export const NAV_TABS: NavTab[] = [
  ...DOCS_PAGES.filter((page) => page.inTopNav).map((page) => ({
    label: page.title,
    href: page.href,
  })),
];
