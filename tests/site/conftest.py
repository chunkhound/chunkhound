import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.site.tsx_runner import NPM, sanitized_subprocess_env

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"


@pytest.fixture(scope="session", autouse=True)
def built_site() -> None:
    """
    Build the site once per test session.
    Set CHUNKHOUND_USE_EXISTING_SITE_DIST=1
    to skip and reuse an existing dist.
    """
    if os.environ.get("CHUNKHOUND_USE_EXISTING_SITE_DIST") == "1":
        if not DIST.exists():
            raise RuntimeError(
                f"Expected prebuilt site at {DIST}. "
                f"Run 'npm run build --prefix site' first, "
                "or unset CHUNKHOUND_USE_EXISTING_SITE_DIST."
            )
        print(f"Reusing existing site dist at {DIST}")
        return

    print(f"Building site for tests... ({DIST})")
    result = subprocess.run(
        [NPM, "run", "build", "--prefix", "site"],
        cwd=ROOT,
        env=sanitized_subprocess_env(),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write("=== Site build failed ===\n")
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        result.check_returncode()
    if not (DIST / "index.html").exists():
        raise RuntimeError(
            f"Build succeeded but {DIST / 'index.html'} was not produced"
        )
