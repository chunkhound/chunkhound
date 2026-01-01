from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree


@dataclass(frozen=True)
class AstroAssetDirs:
    layouts_dir: Path
    styles_dir: Path
    public_dir: Path


@dataclass(frozen=True)
class AstroSiteDirs:
    pages_dir: Path
    topics_dir: Path
    layouts_dir: Path
    styles_dir: Path
    data_dir: Path
    public_dir: Path


def ensure_astro_asset_dirs(*, output_dir: Path) -> AstroAssetDirs:
    src_dir = output_dir / "src"
    layouts_dir = src_dir / "layouts"
    styles_dir = src_dir / "styles"
    public_dir = output_dir / "public"

    for path in (layouts_dir, styles_dir, public_dir):
        path.mkdir(parents=True, exist_ok=True)

    return AstroAssetDirs(
        layouts_dir=layouts_dir,
        styles_dir=styles_dir,
        public_dir=public_dir,
    )


def ensure_astro_site_dirs(
    *,
    output_dir: Path,
    clean_topics_dir: bool,
    allow_delete_topics_dir: bool,
) -> AstroSiteDirs:
    src_dir = output_dir / "src"
    pages_dir = src_dir / "pages"
    topics_dir = pages_dir / "topics"
    layouts_dir = src_dir / "layouts"
    styles_dir = src_dir / "styles"
    data_dir = src_dir / "data"
    public_dir = output_dir / "public"

    if clean_topics_dir and topics_dir.exists():
        if not allow_delete_topics_dir:
            raise RuntimeError(
                "Refusing to delete existing topic pages directory without explicit "
                "confirmation. Use `chunkhound autodoc --force` (or confirm the "
                "interactive prompt) to allow removing: "
                f"{topics_dir}"
            )
        rmtree(topics_dir)

    for path in (pages_dir, topics_dir, layouts_dir, styles_dir, data_dir, public_dir):
        path.mkdir(parents=True, exist_ok=True)

    return AstroSiteDirs(
        pages_dir=pages_dir,
        topics_dir=topics_dir,
        layouts_dir=layouts_dir,
        styles_dir=styles_dir,
        data_dir=data_dir,
        public_dir=public_dir,
    )
