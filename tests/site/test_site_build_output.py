import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.site.tsx_runner import run_tsx_raw, sanitized_subprocess_env

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"
VERSION_FILE = ROOT / "chunkhound" / "_version.py"
VERSION_RESOLUTION_FAILURE = (
    "Unable to resolve ChunkHound version for docs build"
)


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
