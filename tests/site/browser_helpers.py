"""Run Chromium contracts through the site's locked JavaScript dependency."""

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import NoReturn

from tests.helpers.required_tool import skip_or_fail as _skip_or_fail
from tests.site.tsx_runner import ROOT, run_tsx_json, run_tsx_raw

_PLAYWRIGHT = ROOT / "site" / "node_modules" / "playwright" / "index.mjs"
_REQUIRE_SITE_ENV = "CHUNKHOUND_REQUIRE_SITE"
SITE_INSTALL_COMMAND = (
    "npm ci --prefix site && npm exec --prefix site -- playwright install chromium"
)


def skip_or_fail(reason: str) -> NoReturn:
    """Skip locally; fail where CI requires the toolchain to be present."""
    _skip_or_fail(reason, require_env=_REQUIRE_SITE_ENV)


@lru_cache(maxsize=1)
def _chromium_probe() -> tuple[bool, str]:
    """Report (available, missing reason); raise when present tools are broken.

    Only definitive answers cross this boundary, so lru_cache can keep them.
    It never caches exceptions: a broken probe must fail, not silently skip.
    """
    if shutil.which("node") is None or shutil.which("npm") is None:
        return False, "Node.js and npm are not installed"
    if not _PLAYWRIGHT.exists():
        return False, "site JavaScript dependencies are not installed"
    script = (
        f"import {{ chromium }} from '{_PLAYWRIGHT.as_uri()}';\n"
        "console.log(chromium.executablePath());\n"
    )
    # run_tsx_raw raises RuntimeError on timeout; let that fail the test.
    result = run_tsx_raw(script, timeout=30)
    if result.returncode != 0:
        raise AssertionError(f"Chromium probe failed:\n{result.stderr}")
    path = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not path or not Path(path).exists():
        return False, "Playwright Chromium is not installed"
    return True, ""


def require_chromium() -> None:
    """Skip with the install command when Chromium is absent; fail if required."""
    available, missing = _chromium_probe()
    if not available:
        skip_or_fail(f"Browser tests need {missing}. Run: {SITE_INSTALL_COMMAND}")


_IMPORTS = """
import { homepageProbe, pageProbe, ctaSnapshot, themeSnapshot }
  from './site/../tests/site/browser_helpers.mjs';
"""


def probe(script: str) -> dict:
    """Run one Chromium probe and report a browser-side failure as a test failure.

    Every probe ships the same real markup, scripts and CSS, so launch and error
    plumbing has exactly one owner here.
    """
    require_chromium()
    try:
        return run_tsx_json(_IMPORTS + script)
    except subprocess.CalledProcessError as error:
        raise AssertionError(f"Chromium contract failed:\n{error.stderr}") from error


homepage_probe = probe


def page_probe(path: str, widths: list[int], body: str) -> dict:
    """Measure `path` at every width; `body` runs inside the page callback."""
    return probe(
        f"console.log(JSON.stringify(await pageProbe({path!r}, {widths!r},"
        f" async page => {{\n{body}\n}})));"
    )
