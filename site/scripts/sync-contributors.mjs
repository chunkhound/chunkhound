/**
 * Resolve the contributor count into site/src/lib/contributors.json.
 *
 * WHY build-time, not component-time: the receipt is a marketing claim, so it
 * must be the same number the tests see and Pages deploys. One fetch per CI
 * run, before tests. A failed fetch fails the build — a stale or zero receipt
 * is worse than a red run.
 *
 * Env:
 *   CHUNKHOUND_CONTRIBUTORS_JSON  API-shaped pages snapshot (hermetic tests)
 *   CHUNKHOUND_CONTRIBUTORS_OUT   output path override (hermetic tests)
 *   GITHUB_TOKEN / GH_TOKEN       60/h unauthenticated -> 1000/h authenticated
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildContributorRecord,
  nextPageUrl,
  selectContributors,
} from "./lib/contributors.mjs";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
// Relative env paths resolve against the repo root, not the caller's cwd
// (`npm --prefix site` runs with cwd=site).
const ROOT = resolve(SCRIPT_DIR, "../..");
const OUTPUT = process.env.CHUNKHOUND_CONTRIBUTORS_OUT
  ? resolve(ROOT, process.env.CHUNKHOUND_CONTRIBUTORS_OUT)
  : resolve(SCRIPT_DIR, "../src/lib/contributors.json");
const SNAPSHOT = process.env.CHUNKHOUND_CONTRIBUTORS_JSON;
const API =
  "https://api.github.com/repos/chunkhound/chunkhound/contributors?per_page=100";
const MAX_ATTEMPTS = 3;
const REQUEST_TIMEOUT_MS = 10_000;
const MAX_WAIT_MS = 30_000;

const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN;

/** Milliseconds to wait for a retryable response, or null to fail fast. */
function retryDelay(response) {
  const retryAfter = Number(response.headers.get("retry-after"));
  // WHY quota-only: a stale reset header rides along with transient 5xx/403s.
  const remaining = response.headers.get("x-ratelimit-remaining");
  const reset = Number(remaining === "0" ? response.headers.get("x-ratelimit-reset") : NaN);
  const wait = Number.isFinite(retryAfter) && retryAfter > 0
    ? retryAfter * 1000
    : Number.isFinite(reset) && reset > 0
      ? reset * 1000 - Date.now()
      : 2000;
  return wait > 0 && wait <= MAX_WAIT_MS ? wait : null;
}

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: {
      accept: "application/vnd.github+json",
      "x-github-api-version": "2026-03-10",
      "user-agent": "chunkhound-site-build",
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  if (response.ok) {
    return { body: await response.json(), link: response.headers.get("link") };
  }
  const retryable =
    response.status === 403 || response.status === 429 || response.status >= 500;
  if (!retryable) {
    throw new Error(`GitHub contributors API returned ${response.status}`);
  }
  return { retryStatus: response.status, retryDelay: retryDelay(response) };
}

async function fetchPages() {
  const pages = [];
  const visited = new Set();
  let url = API;
  while (url) {
    // WHY: a self-referential rel="next" would fetch forever; fail loudly.
    if (visited.has(url)) {
      throw new Error(`pagination repeated ${url}; set CHUNKHOUND_CONTRIBUTORS_JSON.`);
    }
    visited.add(url);
    let attempt = 0;
    let result;
    for (;;) {
      result = await fetchJson(url);
      if (!result.retryStatus) {
        break;
      }
      attempt += 1;
      if (attempt >= MAX_ATTEMPTS || result.retryDelay === null) {
        throw new Error(
          `GitHub contributors API returned ${result.retryStatus}; ` +
            "rate limit or outage. Retry later, or set CHUNKHOUND_CONTRIBUTORS_JSON.",
        );
      }
      await new Promise((done) => setTimeout(done, result.retryDelay));
    }
    if (!Array.isArray(result.body)) {
      throw new Error("GitHub contributors API returned a non-array payload");
    }
    pages.push(result.body);
    url = nextPageUrl(result.link);
  }
  return pages;
}

function loadPages() {
  if (!SNAPSHOT) {
    return null;
  }
  return JSON.parse(readFileSync(resolve(ROOT, SNAPSHOT), "utf-8"));
}

const pages = loadPages() ?? (await fetchPages());
const record = buildContributorRecord(selectContributors(pages));

mkdirSync(dirname(OUTPUT), { recursive: true });
writeFileSync(OUTPUT, `${JSON.stringify(record, null, 2)}\n`, "utf-8");
console.log(`✓ ${record.count} contributors (as of ${record.asOf}) -> ${OUTPUT}`);
