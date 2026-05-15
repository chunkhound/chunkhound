"""Smoke tests: deck page renders without build errors."""

import shutil
import subprocess

import pytest


def test_deck_astro_build_succeeds():
    """The deck page must compile during Astro build.

    This smoke test catches: missing component imports, broken template
    syntax, CSS parse errors, and JavaScript frontmatter issues — all of
    which are silent during development but surface at build time.
    """
    # On Windows, npx is npx.cmd and may not be on PATH for subprocess.
    # Use shutil.which() to find it cross-platform.
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if npx is None:
        # Node.js/npm not installed — skip gracefully on CI runners
        # that don't have it in PATH.
        pytest.skip("npx not found on PATH")
        return

    result = subprocess.run(
        [npx, "astro", "build"],
        cwd="site",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"astro build failed for deck:\n{result.stderr[-2000:]}"
    )
