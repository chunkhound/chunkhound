import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.site.tsx_runner import NPM, sanitized_subprocess_env

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"
DIST_INDEX = DIST / "index.html"


@pytest.fixture(scope="session", autouse=True)
def built_site() -> None:
    """Reuse an existing built site, or build once for the test session."""
    if os.environ.get("CHUNKHOUND_USE_EXISTING_SITE_DIST") == "1":
        if not DIST.exists():
            raise AssertionError(f"Expected prebuilt site artifact at {DIST}")
        return

    if DIST_INDEX.exists():
        return

    result = subprocess.run(
        [NPM, "run", "build", "--prefix", "site"],
        cwd=ROOT,
        env=sanitized_subprocess_env(),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        result.check_returncode()
