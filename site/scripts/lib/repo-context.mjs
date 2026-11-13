/**
 * Shared repo-context scaffolding for the site/ generator scripts.
 *
 * Both scripts resolve the repo root via CHUNKHOUND_ROOT (hermetic testing)
 * and read site/src/lib/positioning.json — the canonical positioning source.
 *
 * package.json engines >=22.18 is required: generate-llms-txt.mjs imports
 * ../src/lib/nav.ts, which only runs via Node's built-in type-stripping.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPTS_DIR = dirname(fileURLToPath(import.meta.url));

/** Repo root, overridable via CHUNKHOUND_ROOT for hermetic test runs. */
export function repoRoot() {
  return process.env.CHUNKHOUND_ROOT
    ? resolve(process.env.CHUNKHOUND_ROOT)
    : resolve(SCRIPTS_DIR, "../../..");
}

/** The canonical positioning object; missing required fields fail loudly. */
export function loadPositioning(root, requiredFields) {
  const positioningFile = resolve(root, "site/src/lib/positioning.json");
  if (!existsSync(positioningFile)) {
    throw new Error(`site/src/lib/positioning.json not found at ${positioningFile}`);
  }
  const positioning = JSON.parse(readFileSync(positioningFile, "utf-8"));
  for (const field of requiredFields) {
    if (typeof positioning[field] !== "string" || positioning[field].trim() === "") {
      throw new Error(`positioning.json is missing required field "${field}"`);
    }
  }
  return positioning;
}
