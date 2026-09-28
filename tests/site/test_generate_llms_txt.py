"""
Tests for site/scripts/generate-llms-txt.mjs (llms.txt + llms-full.txt).

The script derives the agent-facing summary from the canonical
site/src/lib/positioning.json and concatenates the docs Markdown into
llms-full.txt. CHUNKHOUND_ROOT redirects it at a fake repo for hermetic runs.
"""

import json
import pathlib
import re
import shutil
import subprocess
import tempfile

import pytest

from tests.site.process_runner import run_text_process
from tests.site.tsx_runner import NPM, ROOT, isolated_subprocess_env

GENERATE_SCRIPT = ROOT / "site" / "scripts" / "generate-llms-txt.mjs"

POSITIONING = {
    "description": "Canonical description used as the llms.txt summary.",
    "subheadline": "Canonical subheadline rendered as the llms.txt intro paragraph.",
}
LEGACY_SENTENCE = "Local-first semantic and regex code search"
DOCS = {
    "configuration.md": ("Configuration", "Config body line."),
    "cli-reference.md": ("CLI Reference", "CLI body line."),
    "changelog.md": ("Changelog", "Changelog body line."),
}


def _write_fake_repo(
    repo_root: pathlib.Path, positioning: dict | None = None
) -> None:
    """Create a fake repo with positioning.json, docs pages, and site/public."""
    lib_dir = repo_root / "site" / "src" / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "positioning.json").write_text(
        json.dumps(
            positioning if positioning is not None else POSITIONING, indent=2
        ),
        encoding="utf-8",
    )
    docs_dir = repo_root / "site" / "src" / "pages" / "docs"
    docs_dir.mkdir(parents=True)
    for name, (title, body) in DOCS.items():
        (docs_dir / name).write_text(
            f"---\ntitle: {title}\n---\n\n{body}\n", encoding="utf-8"
        )
    (repo_root / "site" / "public").mkdir(parents=True)


def _run_generate(
    repo_root: pathlib.Path, *extra: str
) -> subprocess.CompletedProcess:
    """Run generate-llms-txt.mjs against a fake repo directory."""
    with isolated_subprocess_env(CHUNKHOUND_ROOT=str(repo_root)) as env:
        return run_text_process(
            [
                NPM,
                "exec",
                "--prefix",
                "site",
                "--",
                "node",
                str(GENERATE_SCRIPT),
                *extra,
            ],
            cwd=ROOT,
            env=env,
        )


def _output(repo_root: pathlib.Path, name: str) -> str:
    return (repo_root / "site" / "public" / name).read_text(encoding="utf-8")


def _strip_frontmatter(text: str) -> str:
    """Body after the closing frontmatter fence — the slice llms-full.txt keeps."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[index + 1 :])
    return text


def test_every_markdown_docs_page_reaches_llms_full() -> None:
    """A new .md docs page must fail here, not silently vanish from
    llms-full.txt. Rather than parse the generator's flatten list, we run the
    real generator over the real docs and require every page's body to reach
    the output, so the check is on what the generator emits."""
    docs_dir = ROOT / "site" / "src" / "pages" / "docs"
    real_docs = {p.relative_to(docs_dir): p for p in docs_dir.rglob("*.md")}

    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        _write_fake_repo(repo_root)
        staged_docs = repo_root / "site" / "src" / "pages" / "docs"
        for relative, source in real_docs.items():
            staged = staged_docs / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        result = _run_generate(repo_root)
        assert result.returncode == 0, result.stderr
        full = _output(repo_root, "llms-full.txt")

    for relative, source in real_docs.items():
        body = _strip_frontmatter(source.read_text(encoding="utf-8")).strip()
        assert body, f"{relative} has no body to reach llms-full.txt"
        assert body in full, f"{relative} is missing from llms-full.txt"


def test_mcp_anchor_linked_by_llms_txt_exists_in_getting_started() -> None:
    """llms.txt hardcodes the MCP anchor; a heading rename must not ship a dead
    anchor to agents."""
    source = GENERATE_SCRIPT.read_text(encoding="utf-8")
    anchors = re.findall(r"/docs/getting-started/#([\w-]+)", source)
    assert anchors, "llms.txt no longer links the getting-started MCP anchor"
    page = (ROOT / "site/src/pages/docs/getting-started.astro").read_text(
        encoding="utf-8"
    )
    for anchor in anchors:
        assert f'id="{anchor}"' in page, (
            f'getting-started has no id="{anchor}" for the llms.txt link'
        )


def test_llms_txt_uses_positioning_copy_and_drops_legacy_sentence() -> None:
    """llms.txt carries both canonical paragraphs; the old hardcoded line is gone."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        _write_fake_repo(repo_root)

        result = _run_generate(repo_root)
        assert result.returncode == 0, result.stderr

        llms = _output(repo_root, "llms.txt")
        assert llms.startswith("# ChunkHound\n")
        assert f"> {POSITIONING['description']}" in llms
        assert POSITIONING["subheadline"] in llms
        assert LEGACY_SENTENCE not in llms


def test_llms_full_strips_frontmatter_and_concatenates_docs() -> None:
    """llms-full.txt is docs Markdown in curated order with frontmatter stripped."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        _write_fake_repo(repo_root)

        result = _run_generate(repo_root)
        assert result.returncode == 0, result.stderr

        full = _output(repo_root, "llms-full.txt")
        assert f"> {POSITIONING['description']}" in full
        for title, body in DOCS.values():
            assert body in full
            assert f"title: {title}" not in full
        assert (
            full.index("Config body line.")
            < full.index("CLI body line.")
            < full.index("Changelog body line.")
        )


def test_check_mode_detects_drift_without_writing() -> None:
    """--check is non-zero before generating and after tampering, zero when in sync."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        _write_fake_repo(repo_root)

        assert _run_generate(repo_root, "--check").returncode != 0

        assert _run_generate(repo_root).returncode == 0
        assert _run_generate(repo_root, "--check").returncode == 0

        (repo_root / "site" / "public" / "llms.txt").write_text(
            "tampered", encoding="utf-8"
        )
        tampered = _run_generate(repo_root, "--check")
        assert tampered.returncode != 0
        assert "Drifted" in tampered.stderr
        # Check mode must not repair the drifted file.
        assert _output(repo_root, "llms.txt") == "tampered"


@pytest.mark.parametrize("field", ["description", "subheadline"])
@pytest.mark.parametrize("value", [None, "   "])
def test_missing_or_blank_positioning_field_fails_loudly(
    field: str, value: str | None
) -> None:
    """positioning.json is the single source; absent or blank fields abort the run."""
    positioning = dict(POSITIONING)
    if value is None:
        positioning.pop(field)
    else:
        positioning[field] = value

    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        _write_fake_repo(repo_root, positioning)

        result = _run_generate(repo_root)
        assert result.returncode != 0
        assert field in result.stderr


def test_real_repo_inputs_generate_cleanly() -> None:
    """The generator succeeds on the real positioning.json and real docs pages,
    not just synthetic fixtures (changelog.md is build-generated → synthetic)."""
    real_positioning = ROOT / "site" / "src" / "lib" / "positioning.json"
    real_docs = ROOT / "site" / "src" / "pages" / "docs"
    canonical = json.loads(real_positioning.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        _write_fake_repo(repo_root)
        (repo_root / "site" / "src" / "lib" / "positioning.json").write_text(
            real_positioning.read_text(encoding="utf-8"), encoding="utf-8"
        )
        for name in ("configuration.md", "cli-reference.md"):
            (repo_root / "site" / "src" / "pages" / "docs" / name).write_text(
                (real_docs / name).read_text(encoding="utf-8"), encoding="utf-8"
            )

        result = _run_generate(repo_root)
        assert result.returncode == 0, result.stderr
        assert canonical["description"] in _output(repo_root, "llms.txt")


def test_missing_positioning_file_fails_loudly() -> None:
    """repo-context.mjs is the single source; a missing file aborts naming it."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        _write_fake_repo(repo_root)
        (repo_root / "site" / "src" / "lib" / "positioning.json").unlink()

        result = _run_generate(repo_root)
        assert result.returncode != 0
        assert "positioning.json" in result.stderr


def test_missing_docs_dir_fails_loudly() -> None:
    """A missing docs tree aborts naming the absent docs page (not silently)."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        _write_fake_repo(repo_root)
        shutil.rmtree(repo_root / "site" / "src" / "pages" / "docs")

        result = _run_generate(repo_root)
        assert result.returncode != 0
        assert "Docs page not found" in result.stderr


def test_frontmatter_without_trailing_newline_does_not_leak() -> None:
    """A page whose closing fence has no trailing newline is frontmatter-only:
    emit nothing, never fall back to slicing from index 0 and leaking YAML."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        _write_fake_repo(repo_root)
        (repo_root / "site" / "src" / "pages" / "docs" / "configuration.md").write_text(
            "---\ntitle: Configuration\n---", encoding="utf-8"
        )

        result = _run_generate(repo_root)
        assert result.returncode == 0, result.stderr
        full = _output(repo_root, "llms-full.txt")
        assert "title: Configuration" not in full


def test_unterminated_frontmatter_fails_loudly() -> None:
    """A page opening with `---` but never closing it aborts with a clear error
    instead of emitting raw YAML into the agent-facing corpus."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        _write_fake_repo(repo_root)
        (repo_root / "site" / "src" / "pages" / "docs" / "configuration.md").write_text(
            "---\ntitle: Configuration\n\nbody without a closing fence",
            encoding="utf-8",
        )

        result = _run_generate(repo_root)
        assert result.returncode != 0
        assert "rontmatter" in result.stderr
