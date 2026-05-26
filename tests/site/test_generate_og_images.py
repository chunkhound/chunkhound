"""
Tests for the site/scripts/generate-og-images.mjs build script.

The script reads SVGs from the public directory and renders them
to 1200px-wide PNGs using @resvg/resvg-js.
"""

import json
import pathlib
import subprocess
import tempfile

from tests.site.tsx_runner import ROOT, NPM, sanitized_subprocess_env

GENERATE_SCRIPT = ROOT / "site" / "scripts" / "generate-og-images.mjs"

# Minimal valid SVG with the correct viewBox (1200x630 OG aspect ratio)
VALID_SVG = """<svg viewBox="0 0 1200 630" width="1200" height="630" xmlns="http://www.w3.org/2000/svg">
  <rect width="1200" height="630" fill="#21221e"/>
</svg>"""

INVALID_SVG = "this is not valid svg content"


def _run_generate(public_dir: pathlib.Path) -> subprocess.CompletedProcess:
    """Run generate-og-images.mjs against a fake public directory."""
    env = sanitized_subprocess_env(CHUNKHOUND_PUBLIC_DIR=str(public_dir))
    return subprocess.run(
        [NPM, "exec", "--prefix", "site", "--", "node", str(GENERATE_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )


def _png_dimensions(png_path: pathlib.Path) -> tuple[int, int]:
    """Read PNG IHDR chunk to extract width and height."""
    with open(png_path, "rb") as f:
        # PNG signature is 8 bytes, IHDR chunk starts at byte 16
        # IHDR structure: 4 bytes length, 4 bytes "IHDR", 4 bytes width, 4 bytes height
        f.seek(16)
        data = f.read(8)
        width = int.from_bytes(data[0:4], "big")
        height = int.from_bytes(data[4:8], "big")
    return (width, height)


def test_generates_both_pngs_from_svgs() -> None:
    """Both dark and light SVGs produce valid 1200px-wide PNGs."""
    with tempfile.TemporaryDirectory() as tmp:
        public_dir = pathlib.Path(tmp)
        for name in ("og-image-dark.svg", "og-image-light.svg"):
            (public_dir / name).write_text(VALID_SVG, encoding="utf-8")

        result = _run_generate(public_dir)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        for name in ("og-image-dark.png", "og-image-light.png"):
            png_path = public_dir / name
            assert png_path.exists(), f"{name} was not generated"
            assert png_path.stat().st_size > 50, f"{name} is too small to be a valid PNG"

            w, h = _png_dimensions(png_path)
            assert w == 1200, f"{name} width is {w}, expected 1200"
            assert h == 630, f"{name} height is {h}, expected 630"


def test_missing_svg_errors() -> None:
    """Missing SVG file produces a descriptive error."""
    with tempfile.TemporaryDirectory() as tmp:
        public_dir = pathlib.Path(tmp)
        # Only create one SVG so the other is missing
        (public_dir / "og-image-dark.svg").write_text(VALID_SVG, encoding="utf-8")

        result = _run_generate(public_dir)
        assert result.returncode != 0
        assert "OG SVG not found" in result.stderr


def test_invalid_svg_errors() -> None:
    """Invalid SVG content produces a descriptive error."""
    with tempfile.TemporaryDirectory() as tmp:
        public_dir = pathlib.Path(tmp)
        for name in ("og-image-dark.svg", "og-image-light.svg"):
            (public_dir / name).write_text(INVALID_SVG, encoding="utf-8")

        result = _run_generate(public_dir)
        assert result.returncode != 0
        assert "Failed to render" in result.stderr


def test_package_scripts_keep_prepare_site_only_for_dev_and_build() -> None:
    """Dev/build keep the shared prepare step while preview remains opt-in."""
    package_json = json.loads((ROOT / "site" / "package.json").read_text(encoding="utf-8"))
    scripts = package_json["scripts"]

    assert scripts["generate:og-images"] == "node scripts/generate-og-images.mjs"
    assert "prepare:site" in scripts
    assert "generate:og-images" in scripts["prepare:site"]
    assert "sync:changelog" in scripts["prepare:site"]
    assert scripts["predev"] == "npm run prepare:site"
    assert scripts["prebuild"] == "npm run prepare:site"
    assert scripts["preview"] == "astro preview"
    assert "prepreview" not in scripts
