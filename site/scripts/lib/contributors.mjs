/**
 * Contributor-count shaping — the single definition of "contributor" for the
 * site receipt.
 *
 * GitHub offers two disagreeing numbers: the REST endpoint (linked accounts
 * with commits, bot accounts omitted) and the UI graph (adds co-authors and bot
 * accounts). The receipt markets people, so the definition is the REST one,
 * with an explicit agent denylist because GitHub classifies automation
 * inconsistently.
 */

/** Next-page URL from an RFC 5988 Link header, or null on the last page. */
export function nextPageUrl(linkHeader) {
  if (!linkHeader) {
    return null;
  }
  for (const part of linkHeader.split(",")) {
    const match = part.match(/<([^>]+)>\s*;\s*rel="next"/);
    if (match) {
      return match[1];
    }
  }
  return null;
}

/** Where a reader can verify the number. */
export const CONTRIBUTORS_SOURCE =
  "https://github.com/chunkhound/chunkhound/graphs/contributors";

/** Agent accounts that must never inflate a "people" claim. */
export const EXCLUDED_LOGINS = new Set([
  "copilot",
  "claude",
  "codex",
  "cursoragent",
  "dependabot[bot]",
  "github-actions[bot]",
]);

/**
 * Collapse paginated REST pages into one entry per human account.
 *
 * Deduplication is by immutable account id, not login: logins can be reused,
 * ids cannot. Bot type is skipped, an unknown type is an error (a silently
 * skipped account would undercount), and anon entries cannot appear because
 * the caller never requests `anon=1`.
 */
export function selectContributors(pages) {
  if (!Array.isArray(pages)) {
    throw new Error("contributor payload must be an array of pages");
  }
  const byId = new Map();
  for (const page of pages) {
    if (!Array.isArray(page)) {
      throw new Error("each contributor page must be an array");
    }
    for (const entry of page) {
      if (!entry || typeof entry !== "object") {
        throw new Error("contributor entry must be an object");
      }
      if (entry.type === "Bot") {
        continue;
      }
      if (entry.type !== "User") {
        throw new Error(`unknown contributor type: ${String(entry.type)}`);
      }
      if (!Number.isInteger(entry.id) || typeof entry.login !== "string") {
        throw new Error("contributor entry is missing its account id or login");
      }
      if (EXCLUDED_LOGINS.has(entry.login.toLowerCase())) {
        continue;
      }
      byId.set(entry.id, entry);
    }
  }
  return [...byId.values()];
}

/** Build the record the site imports. An empty set is a failure, not a zero. */
export function buildContributorRecord(selected, now = new Date()) {
  if (!Array.isArray(selected) || selected.length === 0) {
    throw new Error("resolved zero contributors; refusing to publish a zero receipt");
  }
  return {
    count: selected.length,
    asOf: now.toISOString(),
    source: CONTRIBUTORS_SOURCE,
  };
}
