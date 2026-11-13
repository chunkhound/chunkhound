/**
 * Canonical site-navigation model — the single source of truth for:
 *  - the guide groups (DOCS_SECTIONS): the docs sidebar and the docs landing
 *    page group by product, and the flat guide list derives from them;
 *  - every documentation guide (DOCS_PAGES): featured top-nav shortcuts,
 *    page meta copy, and llms.txt derive from the derived flat list;
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

/**
 * A product-level guide group. Sections own every grouping decision (sidebar
 * landmarks, docs-hub headings) so a page is never tagged twice.
 */
export type DocsSection = {
  /** Anchor for the visible group title; the sidebar nav names itself with it. */
  id: string;
  /** Brand casing as authored — CSS must never uppercase it. */
  title: string;
  pages: DocsPage[];
};

/** Guide groups, priority-ordered; the main project precedes the sister one. */
export const DOCS_SECTIONS: DocsSection[] = [
  {
    id: 'chunkhound',
    title: 'ChunkHound',
    pages: [
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
        title: 'Enterprise Setup',
        href: '/docs/enterprise/',
        icon: 'ph-buildings',
        description:
          'Run ChunkHound across a fleet with one object store for remote configuration and usage analytics.',
        seoDescription:
          'Deploy ChunkHound across an enterprise fleet: pull remote configuration from a private object store, push opt-in usage analytics to the same bucket, and roll out by ring.',
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
        description:
          'Complete reference for all ChunkHound CLI commands and flags.',
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
    ],
  },
  {
    id: 'pi-schematic',
    title: 'pi-schematic',
    pages: [
      {
        title: 'Software Factory',
        href: '/docs/pi-schematic/',
        icon: 'ph-blueprint',
        description:
          'A young project from the ChunkHound team: read-only turns, model routing for skills and sub-agents, and workflows you write in code — optionally grounded by ChunkHound.',
        seoDescription:
          'pi-schematic turns a barebones Pi session into a repeatable process: readonly turns, Model Groups routing skills and sub-agents, and workflow control planes written in code. Optional grounding from ChunkHound — a local index with file and line citations.',
      },
      {
        title: 'Workflows',
        href: '/docs/workflows/',
        icon: 'ph-code',
        description:
          'Use a pi-schematic workflow to inspect code against a spec, plan changes, run tests, and control when work stops.',
      },
    ],
  },
];

/**
 * Flat guide list for consumers that ignore grouping (meta copy, llms.txt,
 * route checks). Derived — never edit it directly.
 */
export const DOCS_PAGES: DocsPage[] = DOCS_SECTIONS.flatMap(
  (section) => section.pages,
);

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
    'Install, configure, operate, and understand ChunkHound — plus pi-schematic, the layer that makes agent work repeatable.',
};

export type NavTab = {
  label: string;
  href: string;
};

/**
 * Top nav = the featured guides, highest-value first.
 * A tab is active on an exact pathname match to its href (see Nav.astro).
 */
export const NAV_TABS: NavTab[] = [
  ...DOCS_PAGES.filter((page) => page.inTopNav).map((page) => ({
    label: page.title,
    href: page.href,
  })),
];
