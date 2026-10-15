"""Run Chromium contracts through the site's locked JavaScript dependency."""

import subprocess

from tests.site.tsx_runner import run_tsx_json


def homepage_probe(script: str) -> dict:
    imports = """
import { homepageProbe, ctaSnapshot, themeSnapshot }
  from './site/../tests/site/browser_helpers.mjs';
"""
    try:
        return run_tsx_json(imports + script)
    except subprocess.CalledProcessError as error:
        raise AssertionError(f"Chromium contract failed:\n{error.stderr}") from error
