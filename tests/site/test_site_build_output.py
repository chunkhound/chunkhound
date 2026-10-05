import html
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.site.html_helpers import (
    NUMERIC_LANGUAGE_CLAIM,
    canonical_href,
    meta_tag_content,
)
from tests.site.png_helpers import png_dimensions
from tests.site.process_runner import run_text_process
from tests.site.tsx_runner import isolated_subprocess_env, run_tsx_json, run_tsx_raw

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"

VERSION_FILE = ROOT / "chunkhound" / "_version.py"
VERSION_RESOLUTION_FAILURE = "Unable to resolve ChunkHound version for docs build"


def _clean_dev_suffix(version: str) -> str:
    return version.split(".dev", 1)[0]


def _run(command: list[str], cwd: Path) -> None:
    run_text_process(command, cwd=cwd, check=True)


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

    git_describe = run_text_process(
        ["git", "describe", "--tags", "--abbrev=0"],
        cwd=root,
        check=True,
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


# Each stage's server-rendered default option: the row it checks on first paint,
# and therefore the row whose reveal carries the prerequisite chips.
_DEFAULT_OPTION = {
    "retrieval": "voyageai",
    "research": "vercel",
    "agent": "pi",
}


def _stage_prerequisites(html: str, stage: str) -> list[tuple[str, str]]:
    """(anchor attributes, inner HTML) for one stage's default row reveal.

    The detail lives in the row itself (`ul.option-requirements` inside
    `.option-reveal`), so the stage's rendered prerequisites are the default
    option's, keyed by `data-option-requirements`.
    """
    match = re.search(
        rf'<ul[^>]*data-option-requirements="{_DEFAULT_OPTION[stage]}"[^>]*>'
        rf"(.*?)</ul>",
        html,
        re.DOTALL,
    )
    assert match is not None, f"Missing prerequisite list for stage {stage!r}"
    anchors = re.findall(r"<a\s([^>]*)>(.*?)</a>", match.group(1), re.DOTALL)
    assert anchors, f"Stage {stage!r} renders no prerequisite links"
    return anchors


def _title_content(html: str) -> str:
    match = re.search(r"<title>(.*?)</title>", html, flags=re.DOTALL)
    assert match is not None, "Missing <title> tag"
    return match.group(1)


def test_homepage_layout_has_expected_sections_and_order() -> None:
    homepage = (DIST / "index.html").read_text(encoding="utf-8")

    assert re.search(
        r'<script[^>]+src="https://cloud\.umami\.is/script\.js"[^>]+data-website-id="[a-f0-9-]+"',
        homepage,
    ), "Umami analytics script missing from homepage"
    assert "/docs/getting-started/" in homepage
    # Single full-bleed terminal surface: the page's single h1 leads the hero,
    # inside the same surface as the live session region (declared before it) —
    # never a separate intro band above a detached proof stage.
    assert homepage.count("<h1") == 1
    assert (
        'aria-label="Example AI agent conversation powered by ChunkHound"' in homepage
    )
    assert homepage.index("<h1") < homepage.index(
        'aria-label="Example AI agent conversation'
    )
    assert "hero-intro" not in homepage
    assert "hero-stage" not in homepage
    assert "/wordmark-text.svg" not in homepage
    # Trust quote relocated from the hero into the scale section's proof strip.
    assert homepage.index('id="scale"') < homepage.index("trust-quote")
    # Homepage leads with scale before the unified repo-and-web research story.
    section_ids = (
        "scale",
        "research",
        "questions",
        "enterprise",
        "configurator",
        "start",
    )
    positions = [homepage.index(f'id="{section_id}"') for section_id in section_ids]
    assert positions == sorted(positions)
    # Top nav leads with the critical destination pages, never same-page
    # scroll anchors, which merely duplicate scrolling (see site/src/lib/nav.ts).
    nav_tabs = re.search(r'<div class="nav-tabs"[^>]*>(.*?)</div>', homepage, re.DOTALL)
    assert nav_tabs is not None, "top nav tabs container missing"
    assert re.findall(r'href="([^"]+)"', nav_tabs.group(1)) == [
        "/docs/getting-started/",
        "/docs/architecture/",
    ]
    assert "remote embedding, reranking, and LLM providers" in homepage
    assert 'id="agent-scale"' not in homepage
    assert 'id="use-cases"' not in homepage
    assert 'src="/logo.svg"' in homepage
    assert 'src="/logo-light.svg"' in homepage
    assert '<nav class="nav-tabs"' not in homepage


def test_homepage_hero_is_one_terminal_surface() -> None:
    """The hero is one full-bleed terminal surface: the full-width h1 leads it,
    then the run's window — the session control in the seam row above the live
    region, all inside .terminal-card — with no separate intro band or detached
    proof stage."""
    homepage = (DIST / "index.html").read_text(encoding="utf-8")

    h1 = homepage.index("<h1")
    card = homepage.index('class="terminal-card"')
    control = homepage.index('class="terminal-control"')
    region = homepage.index('aria-label="Example AI agent conversation')
    assert h1 < card < control < region
    assert "hero-intro" not in homepage
    assert "hero-stage" not in homepage

    # The surface itself owns the full-bleed code background. The hero's CSS
    # was extracted to site/src/styles/hero-terminal.css (global, no Astro
    # scoping attribute), so match the plain `.hero` rule.
    css = "".join(
        bundle.read_text(encoding="utf-8")
        for bundle in (DIST / "_astro").glob("*.css")
    )
    assert re.search(
        r"\.hero\{[^}]*background:\s*var\(--code-bg\)", css
    ), "the hero section must own the full-bleed code surface"


_BAND_RULE = re.compile(
    r"main\s*>\s*section:nth-of-type\(\s*(?:even|2n)\s*\)"
    r"(?::not\(\s*\.brand-surface\s*\))?\s*\{[^}]*--bg-band[^}]*\}"
)


def test_homepage_section_banding_alternates_positionally() -> None:
    """Adjacent homepage sections never share a background: odd positions
    (the hero leads) stay on the page tone and even positions carry
    --bg-band, so the section rhythm alternates.

    The rule must stay positional. A per-component flag set silently drifts
    out of parity as sections are inserted or reordered — the regression that
    shipped two adjacent bands.
    """
    homepage = (DIST / "index.html").read_text(encoding="utf-8")
    first_section = homepage.index("<section", homepage.index("<main"))
    assert 'class="hero"' in homepage[first_section : first_section + 200], (
        "homepage must lead with the hero on the page tone"
    )

    # The closing act must be the LAST section inside <main>: it is the one
    # section exempt from the band (it sits on the brand surface), so it must be
    # excluded positionally rather than sitting in the middle of the rhythm.
    main_body = homepage[homepage.index("<main") : homepage.index("</main>")]
    assert 'id="start"' in main_body[main_body.rindex("<section") :], (
        "the closing activation section must be the last section in <main>"
    )

    css = "".join(
        bundle.read_text(encoding="utf-8")
        for bundle in (DIST / "_astro").glob("*.css")
    )
    assert _BAND_RULE.search(css), (
        "Homepage banding must be positional "
        "(main > section:nth-of-type(even):not(.brand-surface) "
        "{ background: var(--bg-band) }) "
        "so adjacent sections can never share a tone"
    )


def test_homepage_closing_section_uses_the_brand_surface() -> None:
    """The closing act is the one section exempt from positional banding: it sits
    on the brand surface (the page's dark bookend with the hero) so the hound
    mark is licensed and the CTA fill cannot invert between themes."""
    homepage = (DIST / "index.html").read_text(encoding="utf-8")
    match = re.search(r'<section id="start"[^>]*>', homepage)
    assert match is not None, "homepage needs a closing activation section"
    assert "brand-surface" in match.group(0), (
        "the closing section must declare the brand surface"
    )

    css = "".join(
        bundle.read_text(encoding="utf-8")
        for bundle in (DIST / "_astro").glob("*.css")
    )
    assert re.search(
        r"\.brand-surface\s*\{[^}]*background:\s*var\(--brand-surface\)", css
    ), "the brand surface must be a shared global recipe, not a section one-off"


def test_configurator_ui_has_reranker_and_prerequisites() -> None:
    homepage = (DIST / "index.html").read_text(encoding="utf-8")

    assert "macOS/Linux" in homepage
    assert "PowerShell" in homepage
    assert "data-platform-option" in homepage
    assert 'aria-label="Setup configurator"' in homepage
    # The configurator code panel exposes its copy affordance to users
    # (announced by assistive tech) rather than matching internal markup classes.
    assert 'aria-label="Copy setup commands"' in homepage
    # Install-command copy button announces failures too: it carries a status
    # live region (the hero pill is the page's only copy control).
    assert homepage.count('aria-label="Copy install command"') >= 1
    assert homepage.count('data-copy-status') >= 2
    assert "Reranking" in homepage
    assert 'data-rerank-state="included"' in homepage
    assert "Included by VoyageAI." in homepage
    assert "Customize reranker" in homepage
    assert "Reranker endpoint URL" in homepage
    assert "Reranker API format" in homepage
    # The reranker UI is nested inside the retrieval stage panel, before the
    # research stage panel begins (StageDetail.astro + Configurator.astro).
    rerank_index = homepage.index("data-rerank-endpoint")
    assert homepage.index('data-stage-panel="retrieval"') < rerank_index
    assert rerank_index < homepage.index('<h4 id="reranking-heading"')
    assert rerank_index < homepage.index('data-stage-panel="research"')
    # The reranker is a per-provider override rendered inside the default checked
    # row's reveal (StageDetail's slot), so on first paint it reads as part of
    # that row — and sits inside the height-locked list, where its height cannot
    # move the card. configurator.ts relocates the single element on later picks.
    assert homepage.index('data-option-requirements="voyageai"') < rerank_index
    assert rerank_index < homepage.index('data-retrieval="openai-embed"')
    assert homepage.count("data-rerank-endpoint") == 1
    reranker_format_help = (
        "TEI uses the model configured on its server. "
        "Cohere and Voyage require a model."
    )
    assert reranker_format_help in homepage
    assert (
        "Voyage is for native Voyage-compatible endpoints such as MongoDB Atlas"
        in homepage
    )
    assert "top_k" in homepage
    # Prerequisites render into each stage's default row reveal as new-tab
    # external links carrying a decorative logo and a visible label.
    expected_prerequisites = {
        "retrieval": (
            "https://dashboard.voyageai.com/api-keys",
            "VoyageAI API key",
        ),
        "research": (
            "https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai-gateway%2Fapi-keys&title=AI+Gateway+API+Keys",
            "Vercel AI Gateway API key",
        ),
        "agent": ("https://pi.dev", "Pi 1.0+"),
    }
    for stage, (href, label) in expected_prerequisites.items():
        anchors = _stage_prerequisites(homepage, stage)
        visible_labels = [re.sub(r"<[^>]+>", "", body).strip() for _, body in anchors]
        assert visible_labels[0] == label, f"Unexpected prerequisites for {stage!r}"
        assert f'href="{href}"' in html.unescape(anchors[0][0])
        for attrs, body in anchors:
            assert 'target="_blank"' in attrs
            assert 'rel="noopener noreferrer"' in attrs
            assert "<svg" in body, f"Stage {stage!r} prerequisite link missing logo"
    assert (
        'href="https://dashboard.voyageai.com/organization/tos"'
        in _stage_prerequisites(homepage, "retrieval")[1][0]
    )
    assert (
        "Opt out of VoyageAI training for true privacy"
        in _stage_prerequisites(homepage, "retrieval")[1][1]
    )
    assert 'class="prerequisite-item prerequisite-item-optional"' in homepage
    assert "Pi 1.0+" in homepage


def test_getting_started_docs_render_platform_code_and_setup() -> None:
    getting_started = (DIST / "docs" / "getting-started" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "data-platform-code" in getting_started
    assert re.search(
        r'<script[^>]+src="https://cloud\.umami\.is/script\.js"[^>]+data-website-id="[a-f0-9-]+"',
        getting_started,
    ), "Umami analytics script missing from getting_started"
    # Platform blocks render via the shared PlatformCodeBlock component;
    # assert its user-facing affordances (copy button + live status region),
    # not styling classes.
    assert "data-copy-status" in getting_started
    assert 'aria-label="Copy" data-copy=' in getting_started
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
    assert "For a fully local deployment, we recommend" in getting_started
    assert 'href="https://pi.dev"' in getting_started
    assert "pi install npm:pi-mcp-adapter" not in getting_started
    assert "Pi 1.0 and later include MCP support" in getting_started
    assert "pi mcp list" in getting_started
    visible_text = " ".join(re.sub(r"<[^>]+>", "", getting_started).split())
    for instruction in (
        "pi remove npm:pi-mcp-adapter",
        "--local",
        "grant project trust",
        "In the Pi session, run /mcp",
        "ChunkHound is connected",
    ):
        assert instruction in visible_text
    assert f"chunkhound {_expected_docs_version()}" in getting_started
    # Platform shell-chooser code blocks render twice (install + configurator),
    # the first one before the configurator's copyable generated commands.
    shell_tabs = 'aria-label="Choose your shell"'
    assert getting_started.count(shell_tabs) >= 2
    assert getting_started.index(shell_tabs) < getting_started.index(
        'aria-label="Copy setup commands"'
    )
    # Shell tabs follow the ARIA tabs pattern only where panels exist
    # (install block uses idPrefix="install"); the configurator's single-panel
    # switcher (default idPrefix="platform") must not emit dangling aria-controls.
    for platform in ("posix", "powershell"):
        assert f'id="install-tab-{platform}"' in getting_started
        assert f'aria-labelledby="install-tab-{platform}"' in getting_started
        assert f'id="install-panel-{platform}"' in getting_started
        assert f'id="platform-tab-{platform}"' in getting_started
    assert 'role="tabpanel"' in getting_started
    assert getting_started.count('aria-controls="install-panel-') == 2
    assert 'data-platform-mode="tabs"' in getting_started
    assert 'data-platform-mode="switcher"' in getting_started
    assert "cdn.jsdelivr.net" not in getting_started


def test_cli_reference_and_configuration_docs_render_expected_content() -> None:
    cli_reference = (DIST / "docs" / "cli-reference" / "index.html").read_text(
        encoding="utf-8"
    )
    configuration = (DIST / "docs" / "configuration" / "index.html").read_text(
        encoding="utf-8"
    )
    docs_home = (DIST / "docs" / "getting-started" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "chunkhound autodoc map-output/ --out-dir docs-site/" in cli_reference
    assert "chunkhound autodoc --assets-only --out-dir docs-site/" in cli_reference
    assert "chunkhound autodoc --out-dir site/" not in cli_reference
    assert "Complete reference for all ChunkHound CLI commands" in cli_reference
    assert (
        "embedding providers, database backends, and indexing behavior" in configuration
    )
    assert "http://localhost:8001/v1/rerank" in configuration
    assert "Qwen3ForSequenceClassification" in configuration
    sidebar_tag = re.search(r'<aside class="docs-sidebar"[^>]*>', docs_home)
    assert sidebar_tag is not None
    assert 'role="dialog"' not in sidebar_tag.group(0)
    assert 'aria-modal="true"' not in sidebar_tag.group(0)
    assert 'tabindex="-1"' not in sidebar_tag.group(0)
    assert "cdn.jsdelivr.net" not in configuration


def test_closing_install_command_is_one_click_target() -> None:
    """The command text must live inside the copy button. Rendering the field as
    a div with a separate icon-only button left the command click-dead at an
    18px target. The closing act owns the install command (the hero no longer
    carries it), so the copy control must ship inside #start."""
    homepage = (DIST / "index.html").read_text(encoding="utf-8")

    button = re.search(
        r'<button[^>]*data-copy="uv tool install chunkhound"[^>]*>\s*'
        r"<code[^>]*>uv tool install chunkhound</code>",
        homepage,
    )
    assert button, "install command text must live inside the copy button"
    assert "copy-btn" in button.group(0) and "install-command" in button.group(0), (
        "the command button must keep the copy-handler hook classes"
    )


def test_homepage_hero_links_to_the_configurator() -> None:
    """The configurator is not a top-nav destination, so the hero must carry
    its own route to it."""
    homepage = (DIST / "index.html").read_text(encoding="utf-8")

    assert homepage.count('href="/#configurator"') >= 1


def test_homepage_publishes_the_install_path_and_closing_claims() -> None:
    """The install route and the ownership claim are both published, now split
    on purpose: the closing section owns the single route to getting started,
    the footer sign-off owns the claim — so neither restates the other."""
    homepage = (DIST / "index.html").read_text(encoding="utf-8")

    assert "Own the runtime. Own the index. Choose the models." in homepage
    assert "uv tool install chunkhound" in homepage
    assert 'href="/docs/getting-started/"' in homepage
    assert "What do you need to understand next?" not in homepage
    assert "Free and open source, supported by its community." not in homepage


def test_homepage_closing_section_is_the_single_activation_cta() -> None:
    """The page must close on exactly one action: the last <main> section sends
    the reader to the getting-started guide and offers no competing path, while
    still answering the last objections (cost, privacy, licence)."""
    homepage = (DIST / "index.html").read_text(encoding="utf-8")
    match = re.search(r'<section id="start".*?</section>', homepage, re.DOTALL)
    assert match is not None, "homepage needs a closing activation section"
    closing = match.group(0)

    assert re.search(r'href="/docs/getting-started/"[^>]*>\s*Get started', closing)
    for fact in (
        "No license or per-seat fee",
        "Nothing leaves your machine",
        "MIT licensed",
    ):
        assert fact in closing, f"closing section must answer: {fact}"
    # Exactly one action: any second link would split the close.
    assert closing.count("<a ") == 1


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
        with isolated_subprocess_env() as env:
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


def test_homepage_research_connects_repo_and_web_for_real_questions() -> None:
    homepage = (DIST / "index.html").read_text(encoding="utf-8")
    research_match = re.search(
        r'<section id="research".*?</section>', homepage, re.DOTALL
    )
    questions_match = re.search(
        r'<section id="questions".*?</section>', homepage, re.DOTALL
    )
    assert research_match is not None
    assert questions_match is not None
    research = research_match.group(0)
    questions = questions_match.group(0)

    # The repo-and-web knowledge story: both halves merge into one cited answer.
    assert "Give agents the codebase and the browser tab." in research
    for source in (
        "Current code and call paths",
        "Git history and diffs",
        "Official docs and API references",
        "GitHub issues and release notes",
        "Tutorials and working examples",
        "Benchmarks and industry best practices",
    ):
        assert source in research
    assert "INSIDE YOUR REPO" in research
    assert "OPEN IN THE BROWSER" in research
    assert "One cited answer grounded in both" in research

    # The questions the capability answers are their own section after it.
    assert homepage.index('id="research"') < homepage.index('id="questions"')
    assert "Research the problem without bouncing between tools." in questions
    assert "Why is this breaking?" in questions
    assert "Full model-processing privacy requires local" in questions


def test_homepage_enterprise_story_focuses_on_current_user_value() -> None:
    homepage = (DIST / "index.html").read_text(encoding="utf-8")
    match = re.search(r'<section id="enterprise".*?</section>', homepage, re.DOTALL)
    assert match is not None
    enterprise = match.group(0)

    for promise in (
        "Keep the whole codebase in scope",
        "Own the deployment and index",
        "Choose the model boundary",
    ):
        assert promise in enterprise

    for current_value in (
        "without manually assembling context",
        "project database on infrastructure you control",
        "Remote providers receive request content",
    ):
        assert current_value in enterprise

    assert "roadmap" not in homepage.lower()
    assert 'href="/docs/architecture/"' in homepage


def test_homepage_and_readme_state_scale_ownership_and_privacy_boundaries() -> None:
    homepage = (DIST / "index.html").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    combined = f"{homepage}\n{readme}"

    for claim in (
        "Enterprise-scale engineering research. On your laptop.",
        "Own the runtime. Own the index. Choose the models.",
        "MIT licensed",
    ):
        assert claim in homepage
        assert claim in readme

    # Scale figure is surface-consistent: homepage and README both state the
    # same claim (this is a consistency check, not a factuality assertion).
    assert "180-million-line" in homepage
    assert "180M</dt>" in homepage
    assert "180 million lines" in readme

    assert "DuckDuckGo and source websites" in homepage
    assert "DuckDuckGo and source websites" in readme
    assert "local embedding, reranking, and LLM providers" in homepage
    assert "local embedding, reranking, and LLM providers" in readme
    assert "external APIs and hardware may cost money" in readme
    assert "Semantic search requires an embedding provider" in readme
    research_provider_requirement = (
        "research requires an LLM provider and an embedding provider "
        "with reranking support"
    )
    assert research_provider_requirement in readme
    assert "- [Docs](https://chunkhound.ai/docs/getting-started/)" in readme

    for absolute_claim in (
        "100% local",
        "no code leaving your network",
        "Your code never leaves",
        "0 bytes sent",
        "0 bytes leave",
        "zero egress",
        "never leaves",
    ):
        assert absolute_claim not in combined
    assert NUMERIC_LANGUAGE_CLAIM.search(combined) is None


def test_built_site_has_og_meta_tags() -> None:
    """Built homepage includes correct OG and Twitter Card meta tags."""
    homepage = (DIST / "index.html").read_text(encoding="utf-8")
    expected_title = "ChunkHound — Enterprise-scale engineering research on your laptop"
    expected_description = (
        "Free, open-source engineering research for code, git history, rendered web "
        "pages, and technical documents—local-first, enterprise-scale, and cited."
    )

    assert _title_content(homepage) == expected_title
    assert canonical_href(homepage) == "https://chunkhound.ai/"
    assert meta_tag_content(homepage, "name", "description") == expected_description
    assert meta_tag_content(homepage, "property", "og:url") == "https://chunkhound.ai/"
    assert meta_tag_content(homepage, "property", "og:title") == expected_title
    assert (
        meta_tag_content(homepage, "property", "og:description") == expected_description
    )
    assert meta_tag_content(homepage, "name", "twitter:title") == expected_title
    assert (
        meta_tag_content(homepage, "name", "twitter:description")
        == expected_description
    )

    # Meta tag checks must ignore serializer attribute order.
    og_image = meta_tag_content(homepage, "property", "og:image")
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
        content = meta_tag_content(homepage, "property", prop)
        assert content is not None, f"Missing meta tag: {prop}"
        assert content == expected

    tw_image = meta_tag_content(homepage, "name", "twitter:image")
    assert tw_image is not None, "Missing twitter:image meta tag"
    assert tw_image.endswith("/og-image-dark.png")

    tw_card = meta_tag_content(homepage, "name", "twitter:card")
    assert tw_card is not None, "Missing twitter:card meta tag"
    assert tw_card == "summary_large_image"


def test_built_docs_meta_descriptions_follow_navigation() -> None:
    nav = run_tsx_json(
        """import { DOCS_PAGES } from './site/src/lib/nav.ts';
console.log(JSON.stringify({ pages: DOCS_PAGES }));"""
    )
    # nav.ts is the SSOT for docs meta copy (DocsLayout resolves it by path), so
    # a docs route absent from DOCS_PAGES silently falls back to its own prop.
    # Assert the built docs tree and the nav list agree, not just the listed pages.
    built = {
        f"/{path.parent.relative_to(DIST).as_posix()}/"
        for path in (DIST / "docs").glob("*/index.html")
    }
    assert built == {page["href"] for page in nav["pages"]}, (
        "DOCS_PAGES and the built docs tree disagree"
    )
    for page in nav["pages"]:
        html = (DIST / page["href"].strip("/") / "index.html").read_text(
            encoding="utf-8"
        )
        expected = page.get("seoDescription", page["description"])
        assert meta_tag_content(html, "name", "description") == expected, page["href"]


def test_readme_branding_assets_exist() -> None:
    assert (ROOT / "site" / "public" / "wordmark-text.svg").exists()
    assert (ROOT / "site" / "public" / "wordmark-text-dark.svg").exists()
    for name in ("og-image-dark.svg", "og-image-light.svg"):
        assert "Enterprise-scale engineering research. On your laptop." in (
            ROOT / "site" / "public" / name
        ).read_text(encoding="utf-8")


def test_social_preview_accent_dot_matches_wordmark_spacing() -> None:
    """Accent dot sits at cx=595 after the wordmark in every site OG SVG.

    The dot belongs to the wordmark lockup, independent of the centered tagline.
    site/public/ is the source of truth; brand/ copies were removed to avoid drift.
    """
    for name in ("og-image-dark.svg", "og-image-light.svg"):
        svg = (ROOT / "site" / "public" / name).read_text(encoding="utf-8")
        assert '<circle cx="595" cy="64" r="8"' in svg, name
        assert '<circle cx="597" cy="64" r="8"' not in svg, name


def test_built_site_serves_agent_discovery_files() -> None:
    """Agent discovery files are served with their required content.

    llms.txt/llms-full.txt are gitignored build products, so they do not exist
    in a checkout (e.g. CI's artifact-reuse job). Assert the served file's
    contract, never byte-equality with a generated source.
    """
    contracts = {
        "llms.txt": ("# ChunkHound\n", "## Docs\n"),
        "llms-full.txt": ("Full documentation in Markdown for AI agents.",),
        "robots.txt": ("User-agent: *\nAllow: /",),
    }

    for name, markers in contracts.items():
        built = DIST / name
        assert built.is_file(), f"Missing built agent discovery file: {name}"
        content = built.read_text(encoding="utf-8")
        for marker in markers:
            assert marker in content, f"{name} missing {marker!r}"

    # llms.txt spec: Markdown with exactly one H1.
    llms = (DIST / "llms.txt").read_text(encoding="utf-8")
    assert len(re.findall(r"^# ", llms, re.MULTILINE)) == 1


def test_built_site_serves_public_assets_verbatim() -> None:
    """Astro copies committed public/ assets into dist byte-for-byte.

    Uses tracked files only: generated public/ assets are absent in the
    artifact-reuse job, so they cannot prove verbatim passthrough there.
    """
    for name in ("robots.txt", "wordmark-text.svg", "favicon-dark.svg"):
        source = ROOT / "site" / "public" / name
        built = DIST / name
        assert built.is_file(), f"Missing built public asset: {name}"
        assert built.read_bytes() == source.read_bytes()


def test_built_site_has_changelog_page() -> None:
    """Changelog page is built from the current root changelog content."""
    changelog = (DIST / "docs" / "changelog" / "index.html").read_text(encoding="utf-8")
    version, section = _expected_changelog_markers()

    assert version in changelog
    assert section in changelog


def test_built_configuration_docs_render_markdown_tables() -> None:
    """GFM tables must survive the docs build (Astro 7 renders GFM natively;
    this guards against a future processor change dropping that default)."""
    configuration = (DIST / "docs" / "configuration" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "<table" in configuration


def test_built_configuration_docs_document_default_research_route() -> None:
    """The default research route (Vercel AI Gateway) is documented with the
    config the configurator emits."""
    configuration = (DIST / "docs" / "configuration" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "poolside/laguna-s-2.1" in configuration
    assert "Vercel AI Gateway API key" in configuration


def test_built_configuration_docs_state_provider_independence() -> None:
    """Docs must state that the LLM provider is independent of embeddings and
    name the recommended research model."""
    configuration = (DIST / "docs" / "configuration" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "configured independently of" in configuration
    assert "poolside/laguna-s-2.1" in configuration
    assert "qwen/qwen3.7-flash" in configuration
    assert "grok-4.3" in configuration
    assert "gemini-3.5-flash" in configuration


def test_built_docs_code_blocks_render_copy_status() -> None:
    """Markdown code blocks ship the copy-status live region, so copy
    failures are announced (platform blocks already carry their own)."""
    for page in ("getting-started", "contributing"):
        html = (DIST / "docs" / page / "index.html").read_text(encoding="utf-8")
        block_count = len(re.findall(r'class="code-block-md[ "]', html))
        assert block_count > 0, f"{page} has no markdown code blocks"
        # Platform blocks also carry statuses, so the count can only exceed
        # the markdown block count, never fall short of it.
        assert html.count("data-copy-status") >= block_count, (
            f"{page}: markdown code block missing [data-copy-status]"
        )


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


_BUNDLE_SRC = re.compile(r'<script[^>]*\bsrc="(/_astro/[^"]+\.js)"')


@pytest.mark.parametrize(
    "page",
    ("index.html", "docs/getting-started/index.html"),
)
def test_built_pages_ship_their_script_bundles(page: str) -> None:
    """Behavior tests import the site scripts directly and strip <script> from
    dist markup, so nothing else proves a built page actually loads its bundle.
    A page that lost its script tag would ship dead markup while those tests
    stayed green."""
    sources = _BUNDLE_SRC.findall((DIST / page).read_text(encoding="utf-8"))

    assert sources, f"{page} references no /_astro/*.js bundle"
    for source in sources:
        assert (DIST / source.lstrip("/")).is_file(), (
            f"{page} references missing bundle {source}"
        )
