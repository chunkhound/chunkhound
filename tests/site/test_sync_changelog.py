"""
Tests for the site/scripts/sync-changelog.mjs build script.

The script reads the root CHANGELOG.md, prepends Astro frontmatter,
and writes to site/src/pages/docs/changelog.md.
"""

import pathlib
import subprocess
import tempfile

from tests.site.tsx_runner import ROOT, NPM, sanitized_subprocess_env

SYNC_SCRIPT = ROOT / "site" / "scripts" / "sync-changelog.mjs"

CHANGELOG_CONTENT = """# Changelog

## [5.1.0] - 2026-05-20

### Added
- New feature

[5.1.0]: https://github.com/chunkhound/chunkhound/releases/tag/v5.1.0
"""

EXPECTED_FRONTMATTER_START = """---
layout: ../../layouts/DocsLayout.astro
title: "Changelog"
description: "Release history and breaking changes for ChunkHound."
order: 4
section: "manual"
---

"""


def _run_sync(repo_root: pathlib.Path) -> subprocess.CompletedProcess:
    """Run sync-changelog.mjs against a fake repo directory."""
    env = sanitized_subprocess_env(CHUNKHOUND_ROOT=str(repo_root))
    return subprocess.run(
        [NPM, "exec", "--prefix", "site", "--", "node", str(SYNC_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )


def test_sync_prepends_frontmatter() -> None:
    """Frontmatter is prepended, original content preserved."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        # Create root CHANGELOG.md
        (repo_root / "CHANGELOG.md").write_text(CHANGELOG_CONTENT, encoding="utf-8")
        # Create expected output dir structure
        output_dir = repo_root / "site" / "src" / "pages" / "docs"
        output_dir.mkdir(parents=True)

        result = _run_sync(repo_root)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_file = output_dir / "changelog.md"
        assert output_file.exists(), f"Output not written to {output_file}"
        output_text = output_file.read_text(encoding="utf-8")

        assert output_text == EXPECTED_FRONTMATTER_START + CHANGELOG_CONTENT


def test_sync_missing_source_errors() -> None:
    """Script exits non-zero with descriptive error when CHANGELOG.md missing."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        output_dir = repo_root / "site" / "src" / "pages" / "docs"
        output_dir.mkdir(parents=True)

        result = _run_sync(repo_root)
        assert result.returncode != 0
        assert "CHANGELOG.md" in result.stderr
        assert "not found" in result.stderr


def test_sync_empty_source_outputs_frontmatter_only() -> None:
    """Empty CHANGELOG.md produces only the frontmatter block."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        (repo_root / "CHANGELOG.md").write_text("", encoding="utf-8")
        output_dir = repo_root / "site" / "src" / "pages" / "docs"
        output_dir.mkdir(parents=True)

        result = _run_sync(repo_root)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_file = output_dir / "changelog.md"
        output_text = output_file.read_text(encoding="utf-8")
        assert output_text == EXPECTED_FRONTMATTER_START
