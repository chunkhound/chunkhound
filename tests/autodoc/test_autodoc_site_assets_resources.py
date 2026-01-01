from __future__ import annotations

from pathlib import Path

from chunkhound.autodoc.site_writer import write_astro_assets_only
from chunkhound.autodoc.template_loader import load_bytes, load_text


def test_template_loader_reads_packaged_assets() -> None:
    layout = load_text("src/layouts/DocLayout.astro")
    css = load_text("src/styles/global.css")
    favicon = load_bytes("public/favicon.ico")

    assert "navData" in layout
    assert css.strip()
    assert favicon


def test_template_loader_rejects_path_traversal() -> None:
    try:
        load_text("../README.md")
    except ValueError as exc:
        assert "traverse" in str(exc)
    else:
        raise AssertionError("Expected ValueError for path traversal")


def test_template_templates_contain_placeholders() -> None:
    assert "{{TITLE}}" in load_text("README.md")
    assert "{{PACKAGE_NAME}}" in load_text("package.json")


def test_write_astro_assets_only_writes_packaged_favicon_bytes(tmp_path: Path) -> None:
    output_dir = tmp_path / "site"
    data_dir = output_dir / "src" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "site.json").write_text(
        (
            "{\n"
            '  "title": "My Docs",\n'
            '  "tagline": "Tagline",\n'
            '  "scopeLabel": "/",\n'
            '  "generatedAt": "2025-12-22T00:00:00Z",\n'
            '  "sourceDir": ".",\n'
            '  "topicCount": 1\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    write_astro_assets_only(output_dir=output_dir)

    assert (output_dir / "public" / "favicon.ico").read_bytes() == load_bytes(
        "public/favicon.ico"
    )

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    package_json = (output_dir / "package.json").read_text(encoding="utf-8")

    assert "{{TITLE}}" not in readme
    assert "My Docs" in readme

    assert "{{PACKAGE_NAME}}" not in package_json
    assert '"name": "chunkhound-docs' in package_json
