from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
NPM: str = shutil.which("npm") or "npm"


def run_tsx_raw(script: str, **kwargs) -> subprocess.CompletedProcess:
    """Write script to a temp .mts file in ROOT and run via npm exec tsx.

    Returns the raw CompletedProcess. Accepts subprocess.run kwargs except
    capture_output, text, and cwd (already set). Typical usage: check=False
    or env=... to support callers that need non-zero exit handling.
    """
    # Temp file placed in ROOT so relative imports like './site/src/...' resolve.
    # Inline -e breaks on Windows: npm.CMD (batch file) treats newlines as
    # command separators, truncating the script to an empty string.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mts", dir=ROOT, delete=False, encoding="utf-8"
    ) as f:
        temp_path = pathlib.Path(f.name)
        f.write(script)
    try:
        return subprocess.run(
            [NPM, "exec", "--prefix", "site", "--", "tsx", str(temp_path)],
            capture_output=True,
            text=True,
            cwd=ROOT,
            **kwargs,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def run_tsx_json(script: str) -> dict:
    """Execute a repo-local tsx snippet from the site workspace and parse JSON."""
    return json.loads(run_tsx_raw(script, check=True).stdout)
