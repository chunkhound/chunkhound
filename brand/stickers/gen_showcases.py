#!/usr/bin/env python3
"""Generate true-scale laptop-lid showcase SVGs, one per sticker size.

Why a generator: showcases differ only in lid model / sticker size / placement —
one template keeps them in lockstep (same rule as outline_text.py for print art).

Physical models:
  - 14" MacBook Pro (M1-M4 share the chassis): lid 12.31 x 8.71 in, space black
    anodized aluminum, gloss black Apple logo centered (~1.75 in tall).
  - ThinkPad X1 Carbon Gen 12 (the canonical corporate ThinkPad): lid = chassis
    footprint 312.8 x 214.75 mm = 12.31 x 8.45 in (Lenovo PSREF), Eclipse Black
    matte carbon fiber. ThinkPad wordmark sits top-LEFT near the hinge (reads
    upright with the lid closed, hinge away); small Lenovo badge bottom-right.
    Exact logo coordinates are not published (±0.5 mm matters physically) —
    margins here are close approximations from product imagery.

Sticker Mule rounded-corner stock: 0.25 in corner radius at ALL sizes, full
bleed, no cutline/white border (that is die-cut language only).

Placement: straight (no tilt). MBP: top corners (engineer-default). X1 Carbon:
diagonal — holo top-right, light bottom-left (top-left belongs to the ThinkPad
wordmark, bottom-right to the Lenovo badge).

Scale: 128 px/in so a 2560x1440 canvas fills a 2x-retina agent-browser
viewport (set viewport 2560 1440) and the raw screenshot is the PNG export.

Stickers embed the display SVGs with the corner radius patched per size,
base64 data-URI'd into the showcase (self-contained; re-run this script to
re-sync with source art). Display tile = inset 16/480 on a 512 canvas, one
rect with rx=40 (= 0.25 in at 3 in); the holo band is that rect's stroke, so
patching rx re-curves tile and band together. SM's stock radius is a constant
0.25 in at ALL sizes (SM FAQ), so rx = 40*3/size.
"""

# Generated SVG markup must remain on single lines to preserve artifact byte matches.
# ruff: noqa: E501

import base64
from pathlib import Path

from numfmt import fnum

SRC = Path(__file__).parent

PX_PER_IN = 128
CANVAS_W, CANVAS_H = 2560, 1440
LID_Y = 80.0
CORNER_RADIUS_IN = 0.25  # SM stock rounded-corner radius, constant at all sizes
CORNER_MARGIN_IN = 0.35  # sticker inset from lid edges at the top corners

# 14" MacBook Pro
MBP_LID_W_IN, MBP_LID_H_IN = 12.31, 8.71
MBP_LOGO_H_IN = 1.75
MBP_SIZES_IN = (1, 2, 3)

# ThinkPad X1 Carbon Gen 12 — lid = chassis footprint (Lenovo PSREF)
X1C_LID_W_IN, X1C_LID_H_IN = 12.31, 8.45
X1C_TP_LOGO_W_IN = 1.46  # ~37 mm OEM badge
X1C_TP_LOGO_X_IN, X1C_TP_LOGO_Y_IN = 0.55, 0.42  # top-left margins (approx.)
X1C_LENOVO_BADGE_W_IN = 0.62
X1C_SIZES_IN = (2,)

# Apple logo (simple-icons, 24x24 viewBox) — the only logo path we need.
APPLE_PATH = (
    "M12.152 6.896c-.948 0-2.415-1.078-3.96-1.04-2.04.027-3.91 1.183-4.961 3.014"
    "-2.117 3.675-.546 9.103 1.519 12.09 1.013 1.454 2.208 3.09 3.792 3.039"
    "1.52-.065 2.09-.987 3.935-.987 1.831 0 2.35.987 3.96.948"
    "1.637-.026 2.676-1.48 3.676-2.948 1.156-1.688 1.636-3.325 1.662-3.415"
    "-.039-.013-3.182-1.221-3.22-4.857-.026-3.04 2.48-4.494 2.597-4.559"
    "-1.429-2.09-3.623-2.324-4.39-2.376-2-.156-3.675 1.09-4.61 1.09z"
    "M15.53 3.83c.843-1.012 1.4-2.427 1.245-3.83-1.207.052-2.662.805-3.532 1.818"
    "-.78.896-1.454 2.338-1.273 3.714 1.338.104 2.715-.688 3.559-1.701"
)

# ThinkPad wordmark (Wikimedia Commons File:ThinkPad Logo.svg, 72x26 viewBox,
# wordmark only; the dot over the i is the separate red circle below).
THINKPAD_PATH = (
    "M9.457.75v3.022h-3v20.12H3.022V3.773H0V.75h9.457zm4.16 5.624h.065"
    "c.455-.877 1.3-1.3 2.145-1.3 2.373 0 2.405 1.397 2.405 3.348v15.47h-3.12V8.876"
    "c0-.52-.032-1.138-.683-1.138-.78 0-.813.618-.813 1.138v15.016h-3.12V.75h3.12"
    "v5.623zm6.11 17.518V5.3h3.12v18.6h-3.12zM24.34 5.3h3.12v1.073h.065"
    "c.455-.877 1.3-1.3 2.146-1.3 2.373 0 2.405 1.397 2.405 3.348v15.47h-3.12V8.876"
    "c0-.52-.032-1.138-.683-1.138-.78 0-.813.618-.813 1.138v15.016h-3.12V5.3z"
    "m9.102 18.592V.75h3.12v13.1h.13l1.7-8.548h3.12l-1.788 8.386L41.6 23.892h-3.186"
    "L36.7 14.1h-.13v9.783h-3.12zm9.23 0V.75h3.607c3.803 0 5.493 1.43 5.396 4.518"
    "v5.005c.032 3.77-1.886 4.94-5.558 4.745v8.873h-3.445zm3.445-11.636"
    "c1.007 0 2.015.033 2.112-1.04V5.724c0-2.146-.747-2.308-2.112-2.2v8.742z"
    "M60.97 23.892h-2.926c-.194-.4-.194-.845-.194-1.268h-.065"
    "c-.65 1.072-1.138 1.495-2.438 1.495-1.365 0-2.112-.748-2.112-2.146v-6.24"
    "c0-3.77 4.518-1.787 4.518-6.532 0-.553.13-1.853-.618-1.853-.9 0-.747 1.495"
    "-.747 2.145v2.438h-3.152V8.193c0-1.917 1.495-3.12 3.867-3.12s3.868 1.203"
    " 3.868 3.12v15.7zm-3.12-9.62c-.878.52-1.495.9-1.495 1.885v4.68"
    "c.064.4.324.65.747.617.423-.065.748-.325.748-.748V14.27zm12.155 9.62h-3.12"
    "v-1.105h-.065c-.487.683-1.3 1.333-2.242 1.333-1.203 0-2.308-.65"
    "-2.308-2.178V7.25c0-1.527 1.104-2.178 2.308-2.178 1.007 0 1.787.683 2.242"
    " 1.397h.065V.75h3.12v23.142zM65.4 20.316c0 .975.357 1.138.747 1.138"
    "s.748-.163.748-1.138V8.876c0-.975-.358-1.138-.748-1.138S65.4 7.9 65.4"
    " 8.876v11.44z"
)
THINKPAD_DOT_CX, THINKPAD_DOT_CY, THINKPAD_DOT_R = 21.25, 2, 2  # red i-dot

# Lenovo badge (simple-icons, 24x24 viewBox; the badge rect is y 8..16 with the
# wordmark as subpaths). Filled with evenodd over a silver underlay -> silver
# letters knock through the black plaque, matching the OEM lid badge.
LENOVO_PATH = (
    "M21.044 12.288c0 .5-.343.867-.815.867-.464 0-.827-.38-.827-.867"
    " 0-.51.343-.868.815-.868.464 0 .827.381.827.868zm-14.305-.92"
    "a.787.787 0 0 0-.651.307.991.991 0 0 0-.172.738l1.479-.614"
    "a.708.708 0 0 0-.656-.43zm6.963.052c-.472 0-.816.358-.816.868"
    " 0 .486.364.867.828.867.472 0 .815-.368.815-.867 0-.487-.363-.868"
    "-.827-.868zM24 7.997v8.006H0V7.997h24zM5.01 13.05H3.088V9.825H2.23"
    "v4.003h2.78v-.777zm1.137-.094l2.163-.897a1.667 1.667 0 0 0-.37-.86"
    "c-.284-.33-.704-.505-1.216-.505-.931 0-1.633.686-1.633 1.593"
    " 0 .93.704 1.593 1.726 1.593.572 0 1.158-.272 1.432-.589l-.535-.411"
    "c-.357.264-.56.326-.885.326-.292 0-.52-.09-.682-.25zm5.57-1.039"
    "c0-.709-.507-1.223-1.252-1.223a1.28 1.28 0 0 0-1.005.494v-.442h-.846"
    "v3.081h.846v-1.753c0-.316.245-.651.698-.651.35 0 .712.243.712.651"
    "v1.753h.847v-1.91zm3.647.37c0-.904-.725-1.593-1.65-1.593"
    "-.933 0-1.663.7-1.663 1.593 0 .903.726 1.592 1.651 1.592"
    ".932 0 1.662-.7 1.662-1.592zm2.066 1.54l1.268-3.081h-.967l-.765"
    " 2.099-.765-2.1h-.966l1.268 3.081h.927zm4.449-1.54c0-.904-.725"
    "-1.593-1.65-1.593-.932 0-1.662.7-1.662 1.593 0 .903.725 1.592 1.65"
    " 1.592.932 0 1.662-.7 1.662-1.592z"
)

DEFS = f"""<defs>
    <linearGradient id="anodized" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#3a3733"/>
      <stop offset=".45" stop-color="#2e2c28"/>
      <stop offset="1" stop-color="#242220"/>
    </linearGradient>
    <linearGradient id="sheen" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#ffffff" stop-opacity="0"/>
      <stop offset=".42" stop-color="#ffffff" stop-opacity="0"/>
      <stop offset=".5" stop-color="#ffffff" stop-opacity=".07"/>
      <stop offset=".58" stop-color="#ffffff" stop-opacity="0"/>
      <stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="glossblack" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#26231f"/>
      <stop offset=".5" stop-color="#121110"/>
      <stop offset="1" stop-color="#1b1916"/>
    </linearGradient>
    <linearGradient id="eclipse" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#242423"/>
      <stop offset=".45" stop-color="#1b1b1a"/>
      <stop offset="1" stop-color="#131312"/>
    </linearGradient>
    <linearGradient id="tpgraphite" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#555552"/>
      <stop offset="1" stop-color="#383836"/>
    </linearGradient>
    <linearGradient id="vinylgloss" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ffffff" stop-opacity=".14"/>
      <stop offset=".35" stop-color="#ffffff" stop-opacity=".03"/>
      <stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
    <filter id="vinyl" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="{0.08 * PX_PER_IN:.0f}" stdDeviation="{0.11 * PX_PER_IN:.0f}" flood-color="#000000" flood-opacity=".35"/>
    </filter>
  </defs>"""


def sticker_data_uri(name: str, size_in: int) -> str:
    """Display art with the tile corner radius corrected for size.

    Why: SM's stock 0.25 in radius does not scale with sticker size, so the
    3 in display art (rx=40 on a 480-unit tile) needs rx=40*3/size. Patching
    the source rect is safe because each display file has exactly one rect and
    the holo band is that rect's stroke (follows the path automatically).
    """
    art = (SRC / f"sticker-{name}.svg").read_text()
    if art.count('rx="40"') != 1:
        raise ValueError(
            f'sticker-{name}.svg tile rect changed: expected exactly one rx="40"'
        )
    art = art.replace('rx="40"', f'rx="{120 / size_in:g}"')
    return "data:image/svg+xml;base64," + base64.b64encode(art.encode()).decode()


def sticker_group(name: str, cx: float, cy: float, size_in: int, tilt: float) -> str:
    size_px = size_in * PX_PER_IN
    radius_px = CORNER_RADIUS_IN * PX_PER_IN
    embed_scale = size_px / 480  # display tile = inset 16/480 on 512 canvas
    img_size = 512 * embed_scale
    img_x = cx - size_px / 2 - 16 * embed_scale
    img_y = cy - size_px / 2 - 16 * embed_scale
    tile_x, tile_y = cx - size_px / 2, cy - size_px / 2
    return f"""<g transform="rotate({tilt:g} {fnum(cx)} {fnum(cy)})">
    <image href="{sticker_data_uri(name, size_in)}" x="{fnum(img_x)}" y="{fnum(img_y)}" width="{fnum(img_size)}" height="{fnum(img_size)}" filter="url(#vinyl)"/>
    <rect x="{fnum(tile_x)}" y="{fnum(tile_y)}" width="{size_px}" height="{size_px}" rx="{fnum(radius_px)}" fill="url(#vinylgloss)"/>
  </g>"""


def canvas_open(label: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" role="img" aria-label="{label}">'
    )


def caption_strip(text: str, y: float) -> str:
    return (
        f'  <text x="{CANVAS_W // 2}" y="{fnum(y)}" text-anchor="middle" fill="#21221e" '
        f'font-family="Inter, system-ui, sans-serif" font-size="28" font-weight="600" '
        f'letter-spacing="5">{text}</text>'
    )


def lid_chrome(lid_w: float, lid_h: float, lid_x: float, rx: int, fill: str) -> list[str]:
    """Lid body + sheen + edge stroke, shared by both laptops."""
    return [
        f'  <rect x="{fnum(lid_x)}" y="{fnum(LID_Y)}" width="{fnum(lid_w)}" height="{fnum(lid_h)}" rx="{rx}" fill="url(#{fill})"/>',
        f'  <rect x="{fnum(lid_x)}" y="{fnum(LID_Y)}" width="{fnum(lid_w)}" height="{fnum(lid_h)}" rx="{rx}" fill="url(#sheen)"/>',
        f'  <rect x="{fnum(lid_x + 0.5)}" y="{fnum(LID_Y + 0.5)}" width="{fnum(lid_w - 1)}" height="{fnum(lid_h - 1)}" rx="{rx - 0.5}" fill="none" stroke="#000000" stroke-opacity=".4"/>',
    ]


def build_mbp(size_in: int) -> str:
    lid_w, lid_h = MBP_LID_W_IN * PX_PER_IN, MBP_LID_H_IN * PX_PER_IN
    lid_x = (CANVAS_W - lid_w) / 2
    cx_lid, cy_lid = CANVAS_W / 2, LID_Y + lid_h / 2
    logo_scale = MBP_LOGO_H_IN * PX_PER_IN / 24
    margin = CORNER_MARGIN_IN * PX_PER_IN
    half = size_in * PX_PER_IN / 2
    top_y = LID_Y + margin + half
    caption_y = LID_Y + lid_h + 0.55 * PX_PER_IN
    caption = (
        f"STICKER MULE {size_in}&#8243; &#215; {size_in}&#8243; ROUNDED CORNER &#183; "
        f"TRUE TO SCALE ON 14&#8243; MACBOOK PRO LID (12.31&#8243; &#215; 8.71&#8243;)"
    )
    parts = [
        canvas_open(f"Certified Code Hound {size_in} inch stickers at true scale on a 14 inch MacBook Pro lid"),
        f"  <!-- True scale: {PX_PER_IN} px/in. Lid {MBP_LID_W_IN}x{MBP_LID_H_IN} in; sticker {size_in}x{size_in} in = {size_in * PX_PER_IN} px; stock 0.25 in radius. See gen_showcases.py. -->",
        DEFS,
        f'  <rect width="{CANVAS_W}" height="{CANVAS_H}" fill="#f4f6f1"/>',
        "  <!-- Lid: 14in MacBook Pro, space black anodized aluminum. -->",
        *lid_chrome(lid_w, lid_h, lid_x, 19, "anodized"),
        "  <!-- Gloss black Apple logo, centered; faint top rim light = specular edge. -->",
        f'  <path transform="translate({fnum(cx_lid)} {fnum(cy_lid - 2)}) scale({logo_scale:.4f}) translate(-12 -12.4)" fill="#8a8580" opacity=".35" d="{APPLE_PATH}"/>',
        f'  <path transform="translate({fnum(cx_lid)} {fnum(cy_lid)}) scale({logo_scale:.4f}) translate(-12 -12.4)" fill="url(#glossblack)" d="{APPLE_PATH}"/>',
        "  <!-- Stickers: display art embedded so showcases track the source files. -->",
        sticker_group("light", lid_x + margin + half, top_y, size_in, 0),
        sticker_group("holo", lid_x + lid_w - margin - half, top_y, size_in, 0),
        caption_strip(caption, caption_y),
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def thinkpad_badge(lid_x: float) -> str:
    """ThinkPad wordmark + red i-dot, top-left near the hinge (closed-lid readable).

    Exact lid coordinates are unpublished; margins approximate product imagery.
    """
    s = X1C_TP_LOGO_W_IN * PX_PER_IN / 72
    x = lid_x + X1C_TP_LOGO_X_IN * PX_PER_IN
    y = LID_Y + X1C_TP_LOGO_Y_IN * PX_PER_IN
    dot = (
        f'<circle cx="{THINKPAD_DOT_CX}" cy="{THINKPAD_DOT_CY}" r="{THINKPAD_DOT_R}" '
        f'fill="#e32726"/>'
    )
    return (
        f'  <g transform="translate({fnum(x)} {fnum(y)}) scale({s:.4f})">\n'
        f'    <path transform="translate(0 -1.2)" fill="#8a8a86" opacity=".3" d="{THINKPAD_PATH}"/>\n'
        f'    <path fill="url(#tpgraphite)" d="{THINKPAD_PATH}"/>\n'
        f"    {dot}\n  </g>"
    )


def lenovo_badge(lid_x: float, lid_w: float, lid_h: float) -> str:
    """Small black plaque, silver letters (evenodd knockouts over silver underlay)."""
    s = X1C_LENOVO_BADGE_W_IN * PX_PER_IN / 24
    x = lid_x + lid_w - 0.5 * PX_PER_IN - 24 * s
    y = LID_Y + lid_h - 0.45 * PX_PER_IN - 8 * s
    return (
        f'  <g transform="translate({fnum(x)} {fnum(y)}) scale({s:.4f})">\n'
        f'    <rect x="0" y="8" width="24" height="8" fill="#9a9a96"/>\n'
        f'    <path fill="#0a0a0a" fill-rule="evenodd" d="{LENOVO_PATH}"/>\n'
        f"  </g>"
    )


def build_x1c(size_in: int) -> str:
    lid_w, lid_h = X1C_LID_W_IN * PX_PER_IN, X1C_LID_H_IN * PX_PER_IN
    lid_x = (CANVAS_W - lid_w) / 2
    margin = CORNER_MARGIN_IN * PX_PER_IN
    half = size_in * PX_PER_IN / 2
    top_y = LID_Y + margin + half
    # Diagonal: holo top-right, light bottom-left (top-left = ThinkPad
    # wordmark, bottom-right = Lenovo badge).
    bottom_y = LID_Y + lid_h - margin - half
    caption_y = LID_Y + lid_h + 0.55 * PX_PER_IN
    caption = (
        f"STICKER MULE {size_in}&#8243; &#215; {size_in}&#8243; ROUNDED CORNER &#183; "
        f"TRUE TO SCALE ON THINKPAD X1 CARBON GEN 12 LID (12.31&#8243; &#215; 8.45&#8243;)"
    )
    parts = [
        canvas_open(f"Certified Code Hound {size_in} inch stickers at true scale on a ThinkPad X1 Carbon Gen 12 lid"),
        f"  <!-- True scale: {PX_PER_IN} px/in. Lid {X1C_LID_W_IN}x{X1C_LID_H_IN} in (312.8x214.75 mm, Lenovo PSREF); sticker {size_in}x{size_in} in = {size_in * PX_PER_IN} px; stock 0.25 in radius. See gen_showcases.py. -->",
        DEFS,
        f'  <rect width="{CANVAS_W}" height="{CANVAS_H}" fill="#f4f6f1"/>',
        "  <!-- Lid: ThinkPad X1 Carbon Gen 12, Eclipse Black matte carbon fiber. -->",
        *lid_chrome(lid_w, lid_h, lid_x, 12, "eclipse"),
        "  <!-- Lid branding: ThinkPad wordmark top-left, Lenovo badge bottom-right. -->",
        thinkpad_badge(lid_x),
        lenovo_badge(lid_x, lid_w, lid_h),
        "  <!-- Stickers: display art embedded so showcases track the source files. -->",
        sticker_group("light", lid_x + margin + half, bottom_y, size_in, 0),
        sticker_group("holo", lid_x + lid_w - margin - half, top_y, size_in, 0),
        caption_strip(caption, caption_y),
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    for size in MBP_SIZES_IN:
        path = SRC / f"mbp-lid-showcase-{size}in.svg"
        path.write_text(build_mbp(size), encoding="utf-8")
        print(f"wrote {path}")
    for size in X1C_SIZES_IN:
        path = SRC / f"thinkpad-lid-showcase-{size}in.svg"
        path.write_text(build_x1c(size), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
