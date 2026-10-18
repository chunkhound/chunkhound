import subprocess
import sys
from pathlib import Path

# filelock coordinates site builds across xdist workers to prevent race
# conditions when multiple workers try to build the site simultaneously.
import pytest
from filelock import FileLock

from tests.helpers.isolation import ARTIFACTS_KEY
from tests.site.tsx_runner import NPM, sanitized_subprocess_env

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"


def _require_site_dist() -> None:
    if not (DIST / "index.html").is_file():
        raise RuntimeError(
            f"Expected built site at {DIST / 'index.html'}. "
            "Run 'npm run build --prefix site' first, or unset "
            "CHUNKHOUND_USE_EXISTING_SITE_DIST."
        )


def _build_site() -> None:
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
    _require_site_dist()


@pytest.fixture(scope="session", autouse=True)
def built_site(test_run_controls, pytestconfig) -> None:
    """Publish one successful build to all workers; stale dist is not a run marker."""
    if test_run_controls.get("CHUNKHOUND_USE_EXISTING_SITE_DIST") == "1":
        _require_site_dist()
        return

    # The controller owns this unique directory and passes it to every worker.
    state = pytestconfig.stash[ARTIFACTS_KEY] / "site-build"
    state.mkdir(parents=True, exist_ok=True)
    ready = state / "ready"
    with FileLock(state / "build.lock", timeout=300):
        if not ready.exists():
            _build_site()
            ready.touch()
        _require_site_dist()
