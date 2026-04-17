import { execSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

// Walk up from cwd looking for the repo root (the directory that contains
// `chunkhound/_version.py` *or* a `.git` folder). We can't rely on
// `import.meta.url` because Vite bundles this module into the consuming page
// at `site/dist/pages/...`, which would resolve to `site/` instead of the
// repo root.
function findRepoRoot(): string {
    let dir = process.cwd();
    for (let i = 0; i < 10; i++) {
        if (
            existsSync(path.join(dir, "chunkhound", "_version.py")) ||
            existsSync(path.join(dir, "pyproject.toml")) ||
            existsSync(path.join(dir, ".git"))
        ) {
            return dir;
        }
        const parent = path.dirname(dir);
        if (parent === dir) break;
        dir = parent;
    }
    return process.cwd();
}

const repoRoot = findRepoRoot();

export function getChunkhoundVersion(): string {
    // 1. Build-generated _version.py (most precise; matches pip install)
    const versionFile = path.join(repoRoot, "chunkhound/_version.py");
    if (existsSync(versionFile)) {
        const m = readFileSync(versionFile, "utf8")
            .match(/__version__\s*=\s*version\s*=\s*['"]([^'"]+)['"]/);
        if (m) return cleanDevSuffix(m[1]);
    }
    // 2. Latest git tag (works in any checkout with history)
    try {
        const tag = execSync("git describe --tags --abbrev=0", {
            cwd: repoRoot,
            stdio: ["ignore", "pipe", "ignore"],
        })
            .toString()
            .trim();
        if (tag) return tag.replace(/^v/, "");
    } catch {
        // ignore — fall through to explicit failure
    }
    throw new Error(
        "Unable to resolve ChunkHound version for docs build: no chunkhound/_version.py and no Git tags available."
    );
}

function cleanDevSuffix(v: string): string {
    // Strip ".devN+gHASH.dYYYYMMDD" so docs show "4.1.0b1" not the dev tag
    return v.replace(/\.dev\d+.*$/, "");
}
