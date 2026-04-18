from __future__ import annotations

import json
import pathlib
import shutil
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[2]
NPM: str = shutil.which("npm") or "npm"


def run_tsx_json(script: str) -> dict:
    """Execute a repo-local tsx snippet from the site workspace and parse JSON."""
    result = subprocess.run(
        [
            NPM,
            "exec",
            "--prefix",
            "site",
            "--",
            "tsx",
            "--input-type=module",
            "-e",
            script,
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return json.loads(result.stdout)
