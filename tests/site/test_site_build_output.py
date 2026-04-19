import re
import subprocess
import tempfile
from pathlib import Path

from tests.site.tsx_runner import run_tsx_raw

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"
VERSION_FILE = ROOT / "chunkhound" / "_version.py"


def _clean_dev_suffix(version: str) -> str:
    return version.split(".dev", 1)[0]


def _expected_docs_version() -> str:
    if VERSION_FILE.exists():
        match = re.search(
            r"__version__\s*=\s*version\s*=\s*['\"]([^'\"]+)['\"]",
            VERSION_FILE.read_text(encoding="utf-8"),
        )
        if match is None:
            raise AssertionError("Could not parse chunkhound/_version.py version")
        return _clean_dev_suffix(match.group(1))

    git_describe = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return git_describe.stdout.strip().removeprefix("v")


def _extract_astro_code_block_after_marker(html: str, marker: str) -> str:
    marker_index = html.find(marker)
    assert marker_index != -1, f"Missing marker {marker!r}"

    pre_index = html.find('<pre class="astro-code', marker_index)
    assert pre_index != -1, f"Missing astro-code block after {marker!r}"

    end_index = html.find("</pre>", pre_index)
    assert end_index != -1, f"Missing closing </pre> after {marker!r}"

    return html[pre_index : end_index + len("</pre>")]


def test_site_build_outputs_platform_aware_onboarding() -> None:
    homepage = (DIST / "index.html").read_text(encoding="utf-8")
    getting_started = (DIST / "docs" / "getting-started" / "index.html").read_text(encoding="utf-8")
    cli_reference = (DIST / "docs" / "cli-reference" / "index.html").read_text(encoding="utf-8")
    configuration = (DIST / "docs" / "configuration" / "index.html").read_text(encoding="utf-8")
    assert "macOS/Linux" in homepage
    assert "PowerShell" in homepage
    assert re.search(
        r'<script[^>]+src="https://cloud\.umami\.is/script\.js"[^>]+data-website-id="[a-f0-9-]+"',
        homepage,
    ), "Umami analytics script missing from homepage"
    assert "data-platform-option" in homepage
    assert "/docs/getting-started/" in homepage
    assert 'aria-label="Setup configurator"' in homepage
    assert "data-platform-code" in getting_started
    assert re.search(
        r'<script[^>]+src="https://cloud\.umami\.is/script\.js"[^>]+data-website-id="[a-f0-9-]+"',
        getting_started,
    ), "Umami analytics script missing from getting_started"
    assert "platform-code-block" in getting_started
    assert "code-header" in getting_started
    # Astro still emits Shiki's light/dark CSS variables even though the site
    # stylesheet intentionally renders code blocks with the dark token set.
    platform_code_block = _extract_astro_code_block_after_marker(
        getting_started, 'data-platform-code="posix"'
    )
    doc_code_block = _extract_astro_code_block_after_marker(
        getting_started, 'data-copy="chunkhound --version"'
    )
    for code_block in (platform_code_block, doc_code_block):
        assert "astro-code-themes" in code_block
        assert "--shiki-light:" in code_block
        assert "--shiki-dark:" in code_block
    assert "install.ps1" in getting_started
    assert "Expected output" in getting_started
    assert f"chunkhound {_expected_docs_version()}" in getting_started
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


def test_version_helper_fails_without_version_file_or_git_tags() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        script = f"""
import process from "node:process";
(async () => {{
  process.chdir({temp_dir!r});

  try {{
    const {{ getChunkhoundVersion }} = await import({(ROOT / "site" / "src" / "lib" / "version.ts").as_uri()!r});
    console.log(getChunkhoundVersion());
  }} catch (error) {{
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }}
}})();
"""
        result = run_tsx_raw(script, check=False)

    assert result.returncode == 1
    assert (
        "Unable to resolve ChunkHound version for docs build" in result.stderr
        or "Unable to resolve ChunkHound version for docs build" in result.stdout
    )
