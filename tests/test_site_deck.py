"""Smoke tests: deck page renders without build errors."""

import subprocess


def test_deck_astro_build_succeeds():
    """The deck page must compile during Astro build.

    This smoke test catches: missing component imports, broken template
    syntax, CSS parse errors, and JavaScript frontmatter issues — all of
    which are silent during development but surface at build time.
    """
    result = subprocess.run(
        ["npx", "astro", "build"],
        cwd="site",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"astro build failed for deck:\n{result.stderr[-2000:]}"
    )
