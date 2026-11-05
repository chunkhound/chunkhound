/**
 * Generate the agent-readable substrate: site/public/llms.txt (curated index)
 * and site/public/llms-full.txt (concatenated docs Markdown).
 *
 * WHY: agents consume llms.txt as a table of contents and llms-full.txt when
 * they want the full docs without crawling HTML. Building both at prepare time
 * keeps them fresh with the docs and changelog. Markdown-only docs pages are
 * flattened to prose: presentation HTML (figures, icons, callout shells) is for
 * the rendered site only and would duplicate facts in the agent corpus. .astro
 * pages (getting-started, contributing) cannot be flattened to Markdown, so they
 * are linked from llms.txt instead.
 *
 * Output goes to site/public/ so `astro build` serves it at /llms.txt and
 * /llms-full.txt. Like the generated OG PNGs, the files are gitignored.
 *
 * Docs section derives from site/src/lib/nav.ts (DOCS_PAGES/DOCS_HOME) — the
 * same canonical list the sidebar and top nav render from — so titles,
 * descriptions, and ordering cannot drift between llms.txt and the site.
 * Env override: CHUNKHOUND_ROOT — set to repo root for hermetic testing.
 * Mode: pass --check to exit non-zero on drift without writing (CI / tests).
 * Ordering: must run after sync:changelog (llms-full.txt includes it).
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import astroConfig from "../astro.config.mjs";
import { DOCS_HOME, DOCS_PAGES } from "../src/lib/nav.ts";
import { loadPositioning, repoRoot } from "./lib/repo-context.mjs";

const root = repoRoot();

const POSITIONING_FIELDS = ["description", "subheadline"];
const PUBLIC_DIR = resolve(root, "site/public");

// Single source of truth for the public origin: astro.config.mjs `site`.
const SITE_URL = astroConfig.site;
if (typeof SITE_URL !== "string" || !SITE_URL.startsWith("https://")) {
  throw new Error(`astro.config.mjs has no valid site URL (got ${SITE_URL})`);
}

// llms-full.txt concatenates Markdown-backed docs pages as prose. .astro pages
// (getting-started, contributing) cannot be flattened to Markdown, so they are
// linked from llms.txt instead (see header). Hrefs come from nav.ts, so this
// is only the .md-capable subset — titles/descriptions stay in nav.ts.
const MARKDOWN_DOCS_HREFS = [
  "/docs/configuration/",
  "/docs/enterprise/",
  "/docs/cli-reference/",
  "/docs/changelog/",
];

const CHECK_MODE = process.argv.includes("--check");

const positioning = loadPositioning(root, POSITIONING_FIELDS);

// positioning.json is the single source of truth for both agent-facing
// paragraphs; fail loudly when either is absent or blank.
function requiredPositioningField(name) {
  const value = positioning[name];
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`positioning.json is missing required field "${name}"`);
  }
  return value.trim();
}

const summary = requiredPositioningField("description");
const subheadline = requiredPositioningField("subheadline");

// Strip YAML frontmatter so agents get clean Markdown. Malformed frontmatter
// aborts the run: a missing or non-line-anchored closing fence must never let
// YAML leak into the agent-facing corpus (the old slice(0) fallback did).
// LF- and CRLF-tolerant: a CRLF file must strip identically or the leak
// contract breaks for Windows checkouts.
function stripFrontmatter(md) {
  const openingFence = /^---\r?\n/.exec(md);
  if (!openingFence) return md;
  const end = md.indexOf("\n---", 3);
  if (end === -1) throw new Error("Malformed frontmatter: missing closing ---");
  const afterFence = end + 4; // first char after the closing "---"
  const next = md[afterFence];
  // The closing fence must be its own line: `---` at EOF or followed by a
  // newline (\r is tolerated; a following \n is then consumed by trimStart).
  if (next !== undefined && next !== "\n" && next !== "\r") {
    throw new Error("Malformed frontmatter: closing --- is not on its own line");
  }
  // Page with frontmatter but no body: nothing to emit.
  return next === undefined ? "" : md.slice(afterFence + 1).trimStart();
}

// Tags that delimit blocks of visible text. Unwrapping them restores the line
// breaks HTML would have rendered, so unwrapped prose reads as paragraphs.
const BLOCK_TAGS = new Set([
  "address", "article", "aside", "blockquote", "br", "caption", "div",
  "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
  "header", "hr", "li", "ol", "p", "pre", "section", "table", "tbody",
  "td", "tfoot", "th", "thead", "tr", "ul",
]);

// Only these entities are decoded, so literal text such as `&foo;` survives.
const ENTITIES = { "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&amp;": "&" };

// Presentation markup is for the rendered site only. Agents get the visible
// prose: raw SVG paths, class names, and sr-only duplicates add no facts.
function flattenProse(prose) {
  const unwrapped = prose
    .replace(/^[ \t]+(?=<)/gm, "") // HTML source indentation is not prose indentation
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<svg\b[\s\S]*?<\/svg>/gi, "")
    .replace(/<(\w+)\b[^>]*\bclass="[^"]*\bsr-only\b[^"]*"[^>]*>[\s\S]*?<\/\1>/gi, "")
    .replace(/<\/?([a-zA-Z][\w-]*)\b[^>]*>/g, (_, name) =>
      BLOCK_TAGS.has(name.toLowerCase()) ? "\n" : "",
    );
  return unwrapped
    .replace(/&(?:lt|gt|quot|amp|#39);/g, (entity) => ENTITIES[entity])
    .replace(/\n{3,}/g, "\n\n");
}

// Length of the backtick run starting at `start`.
function backtickRun(text, start) {
  let length = 1;
  while (text[start + length] === "`") length++;
  return length;
}

// Index of the backtick run of exactly `run` backticks closing a span, or -1.
// Bounded to the current line: an unmatched backtick must not swallow the
// next paragraph's span.
function closingBackticks(text, from, run) {
  const lineEnd = text.indexOf("\n", from);
  const limit = lineEnd === -1 ? text.length : lineEnd;
  for (let i = from; i < limit; i++) {
    if (text[i] !== "`") continue;
    const length = backtickRun(text, i);
    if (length === run) return i;
    i += length - 1;
  }
  return -1;
}

// Inline code is literal Markdown content, not presentation markup: `<pid>`
// inside a span is a placeholder, so flattenProse would corrupt it. Swap each
// span for a collision-free sentinel, then restore it verbatim afterwards.
function protectInlineCode(prose, spans) {
  let out = "";
  let cursor = 0;
  while (cursor < prose.length) {
    const open = prose.indexOf("`", cursor);
    if (open === -1) break;
    const run = backtickRun(prose, open);
    const close = closingBackticks(prose, open + run, run);
    if (close === -1) {
      out += prose.slice(cursor, open + run);
      cursor = open + run;
      continue;
    }
    out += prose.slice(cursor, open);
    out += `\u0000${spans.length}\u0000`;
    spans.push(prose.slice(open, close + run));
    cursor = close + run;
  }
  return out + prose.slice(cursor);
}

function flattenMarkdownProse(prose) {
  const spans = [];
  const flattened = flattenProse(protectInlineCode(prose, spans));
  return flattened.replace(/\u0000(\d+)\u0000/g, (_, index) => spans[Number(index)]);
}

// Fenced code is opaque: an HTML sample inside a fence must stay byte-identical.
// split() with a capturing group puts fenced segments at odd indexes.
function flattenPresentation(md) {
  return md
    .split(/(```[\s\S]*?```|~~~[\s\S]*?~~~)/g)
    .map((segment, index) =>
      index % 2 === 1 ? segment : flattenMarkdownProse(segment),
    )
    .join("");
}

const DOCS_MD = MARKDOWN_DOCS_HREFS.map((href) => {
  const file = `${href.replace(/^\/docs\//, "").replace(/\/$/, "")}.md`;
  return { file, href };
});

for (const doc of DOCS_MD) {
  const path = resolve(root, "site/src/pages/docs", doc.file);
  if (!existsSync(path)) {
    // changelog.md is generated by sync-changelog.mjs; the others are committed.
    throw new Error(`Docs page not found: ${path}. If it is changelog.md, run sync:changelog first.`);
  }
  doc.content = flattenPresentation(
    stripFrontmatter(readFileSync(path, "utf-8")),
  ).trimEnd();
}

function docsEntry(href) {
  const page = DOCS_PAGES.find((candidate) => candidate.href === href);
  if (!page) throw new Error(`nav.ts has no docs page for ${href}`);
  return `- [${page.title}](${SITE_URL}${page.href}): ${page.description}`;
}

// Docs section derived from nav.ts so llms.txt cannot drift from the site.
// The MCP anchor and the docs hub are agent-specific additions nav.ts does
// not model; everything else reuses nav.ts titles/descriptions verbatim.
const docsSection = [
  `- [${DOCS_HOME.title}](${SITE_URL}${DOCS_HOME.href}): ${DOCS_HOME.description}`,
  docsEntry("/docs/getting-started/"),
  `- [MCP Integration](${SITE_URL}/docs/getting-started/#mcp): Expose ChunkHound to Claude Code, Cursor, or any MCP client.`,
  ...DOCS_PAGES.filter((page) => page.href !== "/docs/getting-started/").map(
    (page) => docsEntry(page.href),
  ),
];

const llmsTxt = [
  "# ChunkHound",
  "",
  `> ${summary}`,
  "",
  subheadline,
  "",
  "## Docs",
  "",
  ...docsSection,
  "",
  "## Project",
  "",
  "- [GitHub](https://github.com/chunkhound/chunkhound): Source code, issues, and install instructions.",
  "- [PyPI](https://pypi.org/project/chunkhound/): Published package (`uv tool install chunkhound`).",
  "",
].join("\n");

const llmsFullTxt = [
  "# ChunkHound",
  "",
  `> ${summary}`,
  "",
  `Full documentation in Markdown for AI agents. Human-rendered docs: ${SITE_URL}/docs/`,
  "",
  "---",
  "",
  ...DOCS_MD.flatMap((doc) => [doc.content, "", "---", ""]),
].join("\n");

const OUTPUTS = [
  { path: resolve(PUBLIC_DIR, "llms.txt"), content: llmsTxt },
  { path: resolve(PUBLIC_DIR, "llms-full.txt"), content: llmsFullTxt },
];

if (CHECK_MODE) {
  const drifted = OUTPUTS.filter(
    (out) => !existsSync(out.path) || readFileSync(out.path, "utf-8") !== out.content
  );
  if (drifted.length === 0) {
    console.log("\u2713 llms.txt and llms-full.txt are in sync");
    process.exit(0);
  }
  console.error(
    `\u2717 Drifted: ${drifted.map((d) => d.path).join(", ")} \u2014 run \`npm --prefix site run generate:llms-txt\``
  );
  process.exit(1);
}

for (const out of OUTPUTS) {
  writeFileSync(out.path, out.content);
  console.log(`\u2713 Wrote ${out.path}`);
}
