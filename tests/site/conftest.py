import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.site.browser_helpers import SITE_INSTALL_COMMAND, skip_or_fail
from tests.site.process_runner import run_text_process
from tests.site.tsx_runner import NPM, isolated_subprocess_env

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"
_SITE_NODE_MIN = (22, 18)
_SITE_TEST_TIMEOUT_S = 700


def pytest_collection_modifyitems(config, items) -> None:
    """Let the site build's 600s subprocess cap fit inside the test timeout."""
    for item in items:
        if Path(__file__).resolve().parent in Path(item.path).resolve().parents:
            item.add_marker(pytest.mark.timeout(_SITE_TEST_TIMEOUT_S))


def _node_version() -> tuple[int, int] | None:
    node = shutil.which("node")
    if node is None:
        return None
    try:
        raw = run_text_process([node, "--version"], timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    match = raw.lstrip("v").split(".", 2)
    if len(match) < 2 or not all(part.isdigit() for part in match[:2]):
        return None
    return int(match[0]), int(match[1])


def _require_site_toolchain() -> None:
    """Skip the site group with an actionable reason, not a build traceback."""
    version = _node_version()
    missing = []
    if shutil.which("npm") is None:
        missing.append("npm")
    if version is None:
        missing.append("Node >=22.18")
    elif version < _SITE_NODE_MIN:
        missing.append(f"Node >=22.18 (found {version[0]}.{version[1]})")
    if not (ROOT / "site" / "node_modules").is_dir():
        missing.append("site/node_modules (run 'npm ci --prefix site')")
    if missing:
        skip_or_fail(
            "Site tests need " + ", ".join(missing) + f". Run: {SITE_INSTALL_COMMAND}"
        )


@pytest.fixture(scope="session", autouse=True)
def built_site() -> None:
    """Build the site once per test session.

    Set CHUNKHOUND_USE_EXISTING_SITE_DIST=1 to skip and reuse an existing dist.
    Under reuse, gitignored generated sources (site/public/llms*.txt,
    og-image-*.png) are absent: tests must read site/dist/** or git-tracked
    files, never generated site/public files.
    """
    _require_site_toolchain()
    if os.environ.get("CHUNKHOUND_USE_EXISTING_SITE_DIST") == "1":
        if not DIST.exists():
            skip_or_fail(
                f"Site tests need a prebuilt site at {DIST}. "
                "Run 'npm run build --prefix site' first, or unset "
                "CHUNKHOUND_USE_EXISTING_SITE_DIST."
            )
        print(f"Reusing existing site dist at {DIST}")
        return

    print(f"Building site for tests... ({DIST})")
    # Cache dir is deleted on context exit; a stalled build fails, not hangs CI.
    with isolated_subprocess_env() as env:
        try:
            result = run_text_process(
                [NPM, "run", "build", "--prefix", "site"],
                cwd=ROOT,
                env=env,
                timeout=600,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("site build timed out after 600s") from e
    if result.returncode != 0:
        sys.stderr.write("=== Site build failed ===\n")
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        result.check_returncode()
    if not (DIST / "index.html").exists():
        raise RuntimeError(
            f"Build succeeded but {DIST / 'index.html'} was not produced"
        )
