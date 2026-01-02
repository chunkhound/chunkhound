import argparse
import json
from pathlib import Path

from chunkhound.api.cli.parsers.autodoc_parser import add_autodoc_subparser
from chunkhound.autodoc.site_writer import write_astro_assets_only


def test_autodoc_parser_accepts_assets_only_flag() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_autodoc_subparser(subparsers)

    args = parser.parse_args(["autodoc", ".", "--out-dir", "site", "--assets-only"])

    assert args.command == "autodoc"
    assert args.assets_only is True


def test_autodoc_parser_defaults_audience_to_balanced() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_autodoc_subparser(subparsers)

    args = parser.parse_args(["autodoc", ".", "--out-dir", "site"])

    assert args.command == "autodoc"
    assert args.audience == "balanced"


def test_autodoc_parser_accepts_audience_numeric_and_named_values() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_autodoc_subparser(subparsers)

    base = ["autodoc", ".", "--out-dir", "site", "--audience"]

    assert parser.parse_args([*base, "1"]).audience == "technical"
    assert parser.parse_args([*base, "technical"]).audience == "technical"
    assert parser.parse_args([*base, "2"]).audience == "balanced"
    assert parser.parse_args([*base, "balanced"]).audience == "balanced"
    assert parser.parse_args([*base, "3"]).audience == "end-user"
    assert parser.parse_args([*base, "end-user"]).audience == "end-user"
    assert parser.parse_args([*base, "end_user"]).audience == "end-user"

def test_autodoc_parser_accepts_map_audience() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_autodoc_subparser(subparsers)

    args = parser.parse_args(
        ["autodoc", ".", "--out-dir", "site", "--map-audience", "3"]
    )
    assert args.map_audience == "end-user"


def test_write_astro_assets_only_preserves_topic_pages(tmp_path: Path) -> None:
    output_dir = tmp_path / "site"
    topics_dir = output_dir / "src" / "pages" / "topics"
    data_dir = output_dir / "src" / "data"
    topics_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    (data_dir / "site.json").write_text(
        json.dumps(
            {
                "title": "Test Site",
                "tagline": "Tagline",
                "scopeLabel": "/",
                "generatedAt": "2025-12-22T00:00:00Z",
                "sourceDir": str(tmp_path),
                "topicCount": 1,
            }
        ),
        encoding="utf-8",
    )

    topic_path = topics_dir / "topic.md"
    topic_path.write_text("original topic content", encoding="utf-8")

    layout_path = output_dir / "src" / "layouts" / "DocLayout.astro"
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    layout_path.write_text("old layout", encoding="utf-8")

    write_astro_assets_only(output_dir=output_dir)

    assert topic_path.read_text(encoding="utf-8") == "original topic content"
    assert layout_path.read_text(encoding="utf-8") != "old layout"
    assert "navData" in layout_path.read_text(encoding="utf-8")
    assert (output_dir / "astro.config.mjs").read_text(encoding="utf-8").strip()
    assert (output_dir / "tsconfig.json").read_text(encoding="utf-8").strip()
    assert (output_dir / "src" / "styles" / "global.css").read_text(
        encoding="utf-8"
    ).strip()
    assert (output_dir / "public" / "favicon.ico").read_bytes()


def test_write_astro_assets_only_preserves_package_and_readme_without_site_json(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "site"
    (output_dir / "src" / "pages" / "topics").mkdir(parents=True)

    package_json_path = output_dir / "package.json"
    readme_path = output_dir / "README.md"
    package_json_path.write_text("sentinel package.json", encoding="utf-8")
    readme_path.write_text("sentinel README", encoding="utf-8")

    write_astro_assets_only(output_dir=output_dir)

    assert package_json_path.read_text(encoding="utf-8") == "sentinel package.json"
    assert readme_path.read_text(encoding="utf-8") == "sentinel README"
