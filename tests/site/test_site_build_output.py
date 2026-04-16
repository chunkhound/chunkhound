import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"


def test_site_build_outputs_platform_aware_onboarding() -> None:
    subprocess.run(
        ["npm", "run", "build", "--prefix", "site"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    homepage = (DIST / "index.html").read_text()
    getting_started = (DIST / "docs" / "getting-started" / "index.html").read_text()
    cli_reference = (DIST / "docs" / "cli-reference" / "index.html").read_text()
    configuration = (DIST / "docs" / "configuration" / "index.html").read_text()

    assert "macOS/Linux" in homepage
    assert "PowerShell" in homepage
    assert "data-platform-option" in homepage
    assert "/docs/getting-started/" in homepage
    assert 'aria-label="Setup configurator"' in homepage
    assert "data-platform-code" in getting_started
    assert "platform-code-block" in getting_started
    assert "code-header" in getting_started
    assert "install.ps1" in getting_started
    assert "Expected output" in getting_started
    assert "code-panel" in homepage
    assert getting_started.count("platform-code-block") >= 2
    assert getting_started.index("platform-code-block") < getting_started.index(
        "code-panel"
    )
    assert "chunkhound autodoc map-output/ --out-dir docs-site/" in cli_reference
    assert "chunkhound autodoc --assets-only --out-dir docs-site/" in cli_reference
    assert "chunkhound autodoc --out-dir site/" not in cli_reference
    assert (
        "Complete reference for all ChunkHound CLI commands and flags."
        in cli_reference
    )
    assert (
        "Configure embedding providers, database backends, and indexing behavior."
        in configuration
    )
    assert '<nav class="nav-tabs"' not in homepage
    assert "cdn.jsdelivr.net" not in getting_started
    assert "cdn.jsdelivr.net" not in configuration
