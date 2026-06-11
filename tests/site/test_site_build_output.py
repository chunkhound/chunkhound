import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.site.png_helpers import png_dimensions
from tests.site.tsx_runner import run_tsx_raw, sanitized_subprocess_env

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"
VERSION_FILE = ROOT / "chunkhound" / "_version.py"
VERSION_RESOLUTION_FAILURE = "Unable to resolve ChunkHound version for docs build"
NUMERIC_LANGUAGE_CLAIM = re.compile(r"\b\d+\+?\s+languages\b", re.IGNORECASE)


def _clean_dev_suffix(version: str) -> str:
    return version.split(".dev", 1)[0]


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _create_tagged_repo(repo_dir: Path, version_tag: str) -> None:
    _run(["git", "init"], repo_dir)
    _run(["git", "config", "user.name", "ChunkHound Tests"], repo_dir)
    _run(["git", "config", "user.email", "tests@chunkhound.invalid"], repo_dir)
    (repo_dir / "README.md").write_text("test\n", encoding="utf-8")
    _run(["git", "add", "README.md"], repo_dir)
    _run(["git", "commit", "-m", "initial"], repo_dir)
    _run(["git", "tag", version_tag], repo_dir)


def _expected_docs_version(
    root: Path = ROOT,
    version_file: Path = VERSION_FILE,
) -> str:
    env_version = os.environ.get("CHUNKHOUND_DOCS_VERSION", "").strip()
    if env_version:
        return _normalize_version(env_version)

    if version_file.exists():
        match = re.search(
            r"__version__\s*=\s*version\s*=\s*['\"]([^'\"]+)['\"]",
            version_file.read_text(encoding="utf-8"),
        )
        if match is None:
            raise AssertionError("Could not parse chunkhound/_version.py version")
        return _normalize_version(match.group(1))

    git_describe = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return _normalize_version(git_describe.stdout.strip())


def _normalize_version(version: str) -> str:
    return _clean_dev_suffix(version).removeprefix("v")


def _write_version_file(repo_dir: Path, version: str) -> Path:
    version_file = repo_dir / "chunkhound" / "_version.py"
    version_file.parent.mkdir()
    version_file.write_text(
        f"__version__ = version = {version!r}\n",
        encoding="utf-8",
    )
    return version_file


def _expected_changelog_markers(
    changelog_path: Path = ROOT / "CHANGELOG.md",
) -> tuple[str, str]:
    version = None
    section = None

    for line in changelog_path.read_text(encoding="utf-8").splitlines():
        if version is None:
            match = re.match(r"## \[([^\]]+)\] - ", line)
            if match:
                version = match.group(1)
            continue

        match = re.match(r"### (.+)", line)
        if match:
            section = match.group(1)
            break

    assert version is not None, "Missing released version heading in CHANGELOG.md"
    assert section is not None, "Missing section heading in CHANGELOG.md"
    return version, section


def _run_version_helper(
    repo_dir: Path, env: dict[str, str]
) -> subprocess.CompletedProcess:
    version_module_uri = (ROOT / "site" / "src" / "lib" / "version.ts").as_uri()
    script = f"""
import process from "node:process";

(async () => {{
  process.chdir({str(repo_dir)!r});

  try {{
    const {{ getChunkhoundVersion }} = await import({version_module_uri!r});
    console.log(getChunkhoundVersion());
  }} catch (error) {{
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }}
}})();
"""
    return run_tsx_raw(script, check=False, env=env)


def _extract_astro_code_block_after_marker(html: str, marker: str) -> str:
    marker_index = html.find(marker)
    assert marker_index != -1, f"Missing marker {marker!r}"

    pre_index = html.find('<pre class="astro-code', marker_index)
    assert pre_index != -1, f"Missing astro-code block after {marker!r}"

    end_index = html.find("</pre>", pre_index)
    assert end_index != -1, f"Missing closing </pre> after {marker!r}"

    return html[pre_index : end_index + len("</pre>")]


def _meta_tag_content(html: str, attr_name: str, attr_value: str) -> str | None:
    """Extract content from the first <meta> tag matching attr_name=attr_value.

    Only handles double-quoted attributes (sufficient for Astro HTML output).
    Returns the first match if multiple tags share the same identifier.
    """
    for match in re.finditer(r"<meta\s+[^>]*>", html):
        attributes = dict(
            re.findall(r'([^\s=/>]+)\s*=\s*"([^"]*)"', match.group(0))
        )
        if attributes.get(attr_name) == attr_value:
            return attributes.get("content")
    return None


def _title_content(html: str) -> str:
    match = re.search(r"<title>(.*?)</title>", html, flags=re.DOTALL)
    assert match is not None, "Missing <title> tag"
    return match.group(1)


def test_site_build_outputs_platform_aware_onboarding() -> None:
    homepage = (DIST / "index.html").read_text(encoding="utf-8")
    getting_started = (DIST / "docs" / "getting-started" / "index.html").read_text(
        encoding="utf-8"
    )
    cli_reference = (DIST / "docs" / "cli-reference" / "index.html").read_text(
        encoding="utf-8"
    )
    configuration = (DIST / "docs" / "configuration" / "index.html").read_text(
        encoding="utf-8"
    )
    docs_home = (DIST / "docs" / "getting-started" / "index.html").read_text(
        encoding="utf-8"
    )
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
    # Wordmark-text SVGs (stacked variant in hero)
    assert "/wordmark-text.svg" in homepage
    assert "/wordmark-text-dark.svg" in homepage
    # UseCases component present
    assert 'id="use-cases"' in homepage
    assert "Research before editing" in homepage
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
    assert 'src="/logo.svg"' in homepage
    assert 'src="/logo-light.svg"' in homepage
    assert "code-panel" in homepage
    assert getting_started.count("platform-code-block") >= 2
    assert getting_started.index("platform-code-block") < getting_started.index(
        "code-panel"
    )
    assert "chunkhound autodoc map-output/ --out-dir docs-site/" in cli_reference
    assert "chunkhound autodoc --assets-only --out-dir docs-site/" in cli_reference
    assert "chunkhound autodoc --out-dir site/" not in cli_reference
    assert "Complete reference for all ChunkHound CLI commands" in cli_reference
    assert "embedding providers, database backends, and indexing behavior" in configuration
    assert '<nav class="nav-tabs"' not in homepage
    sidebar_tag = re.search(r'<aside class="docs-sidebar"[^>]*>', docs_home)
    assert sidebar_tag is not None
    assert 'role="dialog"' not in sidebar_tag.group(0)
    assert 'aria-modal="true"' not in sidebar_tag.group(0)
    assert 'tabindex="-1"' not in sidebar_tag.group(0)
    assert "cdn.jsdelivr.net" not in getting_started
    assert "cdn.jsdelivr.net" not in configuration


@pytest.mark.parametrize(
    ("scenario", "expected_version"),
    [
        ("env_only", "4.1.0b1"),
        ("env_over_file_and_git", "4.1.0b2"),
        ("version_file_only", "4.2.0b1"),
        ("file_over_git", "4.2.1"),
        ("git_tag_only", "4.3.0rc1"),
        ("no_sources", None),
    ],
)
def test_version_helper_contract(scenario: str, expected_version: str | None) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_dir = Path(temp_dir)
        env = sanitized_subprocess_env()

        if scenario == "env_only":
            env["CHUNKHOUND_DOCS_VERSION"] = "v4.1.0b1"
        elif scenario == "env_over_file_and_git":
            env["CHUNKHOUND_DOCS_VERSION"] = "v4.1.0b2"
            _write_version_file(repo_dir, "4.2.0b1.dev3")
            _create_tagged_repo(repo_dir, "v4.3.0rc1")
        elif scenario == "version_file_only":
            _write_version_file(repo_dir, "4.2.0b1.dev3")
        elif scenario == "file_over_git":
            _write_version_file(repo_dir, "4.2.1.dev2")
            _create_tagged_repo(repo_dir, "v4.3.0rc1")
        elif scenario == "git_tag_only":
            _create_tagged_repo(repo_dir, "v4.3.0rc1")
        elif scenario != "no_sources":
            raise AssertionError(f"Unhandled scenario {scenario}")

        result = _run_version_helper(repo_dir, env)
        combined_output = f"{result.stdout}\n{result.stderr}"

    if expected_version is not None:
        assert result.returncode == 0
        assert result.stdout.strip() == expected_version
        assert VERSION_RESOLUTION_FAILURE not in combined_output
    else:
        assert result.returncode != 0
        assert VERSION_RESOLUTION_FAILURE in combined_output


def test_homepage_and_readme_use_qualified_locality_and_language_claims() -> None:
    homepage = (DIST / "index.html").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    combined = f"{homepage}\n{readme}"

    assert "Local-first" in homepage
    assert "Local-first" in readme
    assert "Dozens of languages & file types" in homepage
    assert "Dozens of languages & file types" in readme
    assert "can run fully local" in homepage
    assert "Zero code egress" in homepage
    assert "with local providers" in homepage
    assert "local-provider option" in homepage
    assert "Semantic search requires an embedding provider" in readme
    assert "research requires an LLM provider and an embedding provider with reranking support" in readme
    assert "zero-code-egress" in readme
    assert "- [Docs](https://chunkhound.ai/docs/getting-started/)" in readme

    for absolute_claim in (
        "100% local",
        "no code leaving your network",
        "Your code never leaves",
        "0 bytes sent",
    ):
        assert absolute_claim not in combined
    assert NUMERIC_LANGUAGE_CLAIM.search(combined) is None
    assert "Your entire engineering context, deeply understood" in homepage
    assert "Your entire engineering context, deeply understood." in readme
    assert "Cited answers · Git history research · Pinpoint web research" in homepage
    assert "Cited answers · Git history research · Pinpoint web research" in readme


def test_built_site_has_og_meta_tags() -> None:
    """Built homepage includes correct OG and Twitter Card meta tags."""
    homepage = (DIST / "index.html").read_text(encoding="utf-8")
    expected_title = "ChunkHound — Your engineering context, deeply understood"
    expected_description = (
        "ChunkHound gives AI agents cited context across current code, git history, "
        "and technical web research — for safer edits, clearer reviews, and "
        "release-ready summaries. Local-first. MIT licensed."
    )

    assert _title_content(homepage) == expected_title
    assert _meta_tag_content(homepage, "name", "description") == expected_description
    assert _meta_tag_content(homepage, "property", "og:title") == expected_title
    assert (
        _meta_tag_content(homepage, "property", "og:description")
        == expected_description
    )
    assert _meta_tag_content(homepage, "name", "twitter:title") == expected_title
    assert (
        _meta_tag_content(homepage, "name", "twitter:description")
        == expected_description
    )

    # Meta tag checks must ignore serializer attribute order.
    og_image = _meta_tag_content(homepage, "property", "og:image")
    assert og_image is not None, "Missing og:image meta tag"
    assert og_image.startswith("https://"), (
        f"OG image URL should be absolute: {og_image}"
    )
    assert og_image.endswith("/og-image-dark.png")

    for prop, expected in [
        ("og:image:type", "image/png"),
        ("og:image:width", "1200"),
        ("og:image:height", "630"),
        ("og:type", "website"),
    ]:
        content = _meta_tag_content(homepage, "property", prop)
        assert content is not None, f"Missing meta tag: {prop}"
        assert content == expected

    tw_image = _meta_tag_content(homepage, "name", "twitter:image")
    assert tw_image is not None, "Missing twitter:image meta tag"
    assert tw_image.endswith("/og-image-dark.png")

    tw_card = _meta_tag_content(homepage, "name", "twitter:card")
    assert tw_card is not None, "Missing twitter:card meta tag"
    assert tw_card == "summary_large_image"


def test_readme_branding_assets_exist() -> None:
    assert (ROOT / "site" / "public" / "wordmark-text.svg").exists()
    assert (ROOT / "site" / "public" / "wordmark-text-dark.svg").exists()
    for name in ("og-image-dark.svg", "og-image-light.svg"):
        assert "Your entire engineering context, deeply understood" in (
            ROOT / "site" / "public" / name
        ).read_text(encoding="utf-8")


def test_built_site_has_changelog_page() -> None:
    """Changelog page is built from the current root changelog content."""
    changelog = (DIST / "docs" / "changelog" / "index.html").read_text(encoding="utf-8")
    version, section = _expected_changelog_markers()

    assert version in changelog
    assert section in changelog


def test_built_docs_pages_render_toc_links_server_side() -> None:
    for page, anchors in {
        "getting-started": (
            "#install",
            "#index-and-verify",
            "#use-it-from-your-agent",
            "#example-prompts",
            "#mcp",
            "#where-to-next",
        ),
        "contributing": (
            "#getting-started",
            "#development-workflow",
            "#the-review-process",
            "#what-makes-a-good-pr",
        ),
        "configuration": (
            "#configuration-file",
            "#configuration-precedence",
            "#embedding-providers",
            "#advanced-routing",
        ),
        "cli-reference": (
            "#chunkhound-index",
            "#chunkhound-search",
            "#chunkhound-research",
            "#common-flags",
        ),
        "changelog": (
            "#unreleased",
            "#breaking-changes",
            "#added",
            "#changed",
        ),
    }.items():
        html = (DIST / "docs" / page / "index.html").read_text(encoding="utf-8")

        assert '<nav class="toc-list" data-toc>' in html
        for anchor in anchors:
            assert f'href="{anchor}"' in html, f"Missing TOC anchor {anchor} on {page}"


def test_built_site_has_og_png_assets() -> None:
    """OG PNG images exist in dist/ with correct 1200x630 dimensions."""
    for name in ("og-image-dark.png", "og-image-light.png"):
        png_path = DIST / name
        assert png_path.exists(), f"{name} missing from dist/"
        assert png_path.stat().st_size > 5000, (
            f"{name} is too small ({png_path.stat().st_size} bytes)"
        )

        width, height = png_dimensions(png_path)
        assert width == 1200, f"{name} width is {width}, expected 1200"
        assert height == 630, f"{name} height is {height}, expected 630"
