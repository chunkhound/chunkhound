"""Convert text runs to outlined SVG paths using Inter (variable font).

WHY: Print shops / SVG consumers that lack the font rasterize <text> wrong;
outlined paths are font-independent and pixel-exact. Baselines include the
+11 bleed inset of the print canvas (holo display baselines 390/458,
light 357/400/458).
WHAT: Pins Inter at wght=700 (Bold), draws glyph outlines into SVG path `d`
data with px font-size, CSS letter-spacing, and x-centering. Paths carry no
fill - the consuming group decides (mask black / preview holo gradient).

Two variants: default serves the holo print file (dotless ChunkHound);
`--vinyl` serves the light print file - ChunkHound keeps its terminal
dot, emitted as a separate group so the print SVG fills it cyan.

USAGE: uv run --with fonttools python outline_text.py [--vinyl] [--out PATH]
(writes .outlined-text.svg next to this script by default; its <g>s go into
the print SVG)
"""

import argparse
import hashlib
import io
import urllib.request
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from numfmt import fnum

# Inter build pin: google/fonts main HEAD when the font was fetched. The commit
# URL freezes the bytes; the sha256 makes an upstream rewrite explicit.
FONT_COMMIT = "8e44913e4ff26fc997e6856c1ec40ff4791c98c5"
FONT_URL = (
    f"https://raw.githubusercontent.com/google/fonts/{FONT_COMMIT}/ofl/inter/"
    "Inter%5Bopsz,wght%5D.ttf"
)
FONT_SHA256 = "29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031"
FONT_CACHE = Path.home() / ".cache" / "chunkhound" / "inter-variable.ttf"
# Vendored copy of the pinned bytes above — checked first so print art builds
# offline and reproducibly without hitting the network. Lives at brand level
# (brand/fonts/) because it is shared: the sticker generator loads it for print
# outlines and the site's hero-note test loads it as a text-measurement fixture.
VENDORED_FONT = Path(__file__).parent.parent / "fonts" / "inter-variable.ttf"

# Text run specs: (text, size_px, letter_spacing_px, center_x, baseline_y)
# fill is applied per-run / per-segment by the caller.
RUNS = [
    ("LGTM", 64, 6.0, 256, 401),
    ("ChunkHound", 31, -1.5, 256, 469),
]

# Vinyl (light print): identical baselines to RUNS, but ChunkHound keeps its
# terminal dot (dot loss is scoped to the holo variant only).
VINYL_RUNS = [
    ("CERTIFIED", 35, 3.0, 256, 368),
    ("CODE HOUND", 35, 3.0, 256, 411),
    ("ChunkHound.", 31, -1.5, 256, 469),
]


def _verified(data: bytes) -> bytes:
    """Reject bytes that don't match the pinned upstream Inter build."""
    digest = hashlib.sha256(data).hexdigest()
    if digest != FONT_SHA256:
        raise ValueError(f"Inter sha256 mismatch: got {digest}, expected {FONT_SHA256}")
    return data


def _download_font(retries: int = 2) -> bytes:
    """Fetch pinned Inter bytes with retries; fail loud with offline hint."""
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            return urllib.request.urlopen(FONT_URL, timeout=30).read()
        except Exception as e:  # network down / DNS / TLS — retry, then explain
            last = e
    raise RuntimeError(
        "Inter font unavailable: vendored copy and cache both missing and "
        f"network download failed ({last}). Pre-seed the cache by running "
        "this script once with network access, or restore "
        "brand/fonts/inter-variable.ttf."
    ) from last


def load_inter_bold() -> TTFont:
    """Load the sha256-pinned Inter variable font (vendored → cached → net).

    WHY instancer: the raw variable font renders at default wght=400;
    pinning bakes Bold outlines/metrics into a static instance. WHY sha256:
    verified on every load so a stale cache or upstream rewrite fails loudly
    instead of silently changing the print art.
    """
    if VENDORED_FONT.exists():
        data = _verified(VENDORED_FONT.read_bytes())
    elif FONT_CACHE.exists():
        data = _verified(FONT_CACHE.read_bytes())
    else:
        data = _verified(_download_font())
        FONT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        FONT_CACHE.write_bytes(data)
    font = TTFont(io.BytesIO(data))
    return instantiateVariableFont(font, {"wght": 700})


def glyph_path(glyph_set, glyph_name: str) -> str:
    """Return SVG path `d` for one glyph in raw font units (y-up).

    Flip (y-up font → y-down SVG) and px scaling happen in the wrapping
    transform, keeping path data compact and per-point math unnecessary.
    """
    # 1 decimal of a font unit ≈ 0.008px at 35px size — invisible, and keeps
    # the file small (instancer emits long floats otherwise).
    pen = SVGPathPen(glyph_set, ntos=lambda v: fnum(v))
    glyph_set[glyph_name].draw(pen)
    return pen.getCommands()


def outline_run(
    font: TTFont,
    text: str,
    size: float,
    letter_spacing: float,
    center_x: float,
    baseline_y: float,
) -> tuple[list[tuple[str, float]], float, float, float]:
    """Outline one text run.

    Returns (segments, x0, baseline_y, scale) where segments is
    [(d, x_offset_px), ...] — each glyph path plus its pen x (advances + CSS
    letter-spacing after every glyph). x0 places the run centered at
    center_x; scale lets the caller build the flip+scale transform. Width
    includes the trailing letter-spacing so centering matches CSS
    text-anchor=middle semantics (consumed internally for x0).
    """
    cmap = font.getBestCmap()
    hmtx = font["hmtx"]
    upm = font["head"].unitsPerEm  # Inter: 2048 — never assume 1000
    glyph_set = font.getGlyphSet()
    scale = size / upm

    pen_x = 0.0
    segments: list[tuple[str, float]] = []
    for ch in text:
        gname = cmap[ord(ch)]
        advance = hmtx[gname][0]
        d = glyph_path(glyph_set, gname)
        if d:  # space glyphs have empty outlines
            segments.append((d, pen_x * scale))
        pen_x += advance + letter_spacing / scale  # spacing in font units

    # Shift so the run is centered at center_x, then y-baseline: font units
    # are flipped into y-down px by wrapping in a scaled transform below.
    x0 = center_x - (pen_x * scale) / 2
    return segments, x0, baseline_y, scale


def fmt(v: float) -> str:
    """Compact float formatting to keep SVG small (3-decimal geometry)."""
    return fnum(v, decimals=3)


def segment_paths(
    segments: list[tuple[str, float]], x0: float, baseline_y: float, scale: float
) -> list[str]:
    """One SVG path per glyph; transform encodes flip+scale+placement so the
    path data stays in raw font units (compact, no per-point math)."""
    return [
        '<path transform="translate('
        f"{fmt(x0 + xoff)},{fmt(baseline_y)}) scale("
        f'{fmt(scale)},{fmt(-scale)})" d="{d}"/>'
        for d, xoff in segments
    ]


def build_svg(font: TTFont | None = None) -> str:
    """Outline the holo runs into one <g> of fill-less paths."""
    if font is None:
        font = load_inter_bold()
    parts = ["<g>"]
    for text, size, ls, cx, by in RUNS:
        segments, x0, baseline_y, scale = outline_run(font, text, size, ls, cx, by)
        parts.extend(segment_paths(segments, x0, baseline_y, scale))
    parts.append("</g>")
    return "\n".join(parts)


def build_vinyl_svg(font: TTFont | None = None) -> str:
    """Outline the light print runs: main text and terminal dot as separate
    groups so the print SVG fills the dot cyan.

    font is injectable (defaults to the cached Inter download) so callers can
    pass a preloaded/pinned font instead of hitting the network.
    """
    if font is None:
        font = load_inter_bold()
    main_paths: list[str] = []
    dot_paths: list[str] = []
    for text, size, ls, cx, by in VINYL_RUNS:
        segments, x0, baseline_y, scale = outline_run(font, text, size, ls, cx, by)
        if text.endswith("."):  # terminal dot = last glyph, filled cyan in print
            main_paths.extend(segment_paths(segments[:-1], x0, baseline_y, scale))
            dot_paths = segment_paths(segments[-1:], x0, baseline_y, scale)
        else:
            main_paths.extend(segment_paths(segments, x0, baseline_y, scale))

    def group(gid: str, paths: list[str]) -> str:
        return f'<g id="{gid}">\n' + "\n".join(paths) + "\n</g>"

    return "\n".join([group("text-main", main_paths), group("text-dot", dot_paths)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vinyl", action="store_true", help="Light print runs")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / ".outlined-text.svg",
        help="Output SVG path (default: .outlined-text.svg next to this script)",
    )
    args = ap.parse_args()

    svg = build_vinyl_svg() if args.vinyl else build_svg()
    args.out.write_text(svg + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
