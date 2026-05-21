/**
 * Sync the root CHANGELOG.md into the site's changelog page.
 *
 * Reads the root CHANGELOG.md, prepends the Astro frontmatter needed by the
 * site layout, and writes to site/src/pages/docs/changelog.md.
 *
 * Package scripts run this before dev/build/preview so served docs do not drift
 * from the repository changelog.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "../..");

const ROOT_CHANGELOG = resolve(repoRoot, "CHANGELOG.md");
const SITE_CHANGELOG = resolve(repoRoot, "site/src/pages/docs/changelog.md");

const FRONTMATTER = `---
layout: ../../layouts/DocsLayout.astro
title: "Changelog"
description: "Release history and breaking changes for ChunkHound."
order: 4
section: "manual"
---

`;

const content = readFileSync(ROOT_CHANGELOG, "utf-8");
writeFileSync(SITE_CHANGELOG, FRONTMATTER + content);
console.log(`✓ Synced site changelog from ${ROOT_CHANGELOG}`);
