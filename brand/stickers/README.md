# brand/stickers

"Certified Code Hound" sticker assets: display art, print files for Sticker
Mule, true-scale laptop-lid showcases, and the generator scripts. SVG is the
source of truth; PNGs are rendered artifacts.

> **Status: POC — not final.** This art is exploratory and has no test
> coverage; expect it to change or move before it ships anywhere public.
> `make lint` checks these scripts with ruff; they are outside mypy's
> `chunkhound` scope and have no formatting/type gate.

## Files

| File | What it is |
|---|---|
| `outline_text.py` | **Source script.** Outlines text runs into SVG glyph paths using Inter Bold (variable font pinned to `wght=700` via fontTools instancer). Writes `.outlined-text.svg`; its `<g>` output gets pasted into the print SVGs. Default mode = holo runs (`LGTM`, dotless `ChunkHound`); `--vinyl` = light-print runs (`CERTIFIED`, `CODE HOUND`, `ChunkHound.` with the dot as a separate group filled cyan). |
| `gen_showcases.py` | **Source script.** Generates the four lid-showcase SVGs (2560x1440 canvas, 128 px/in true scale). Embeds `sticker-light.svg` / `sticker-holo.svg` as base64 data URIs with the tile corner radius patched per sticker size (rx = 40·3/size). Stdlib only. |
| `numfmt.py` | **Shared utility.** Compact, precision-limited float formatting used by both SVG generators to keep generated paths small without visible loss at sticker scale. |
| `../fonts/inter-variable.ttf` | **Vendored font** (876 KB), shared at brand level. sha256-pinned Inter variable font that `outline_text.py` loads first (offline-safe); see Licensing and provenance. |
| `../fonts/inter-OFL.txt` | **License text** for the vendored Inter font (SIL OFL 1.1), kept beside the font. |
| `.outlined-text.svg` | **Generated intermediate** from `outline_text.py` (holo mode default; `--out` overrides the path). Never hand-edit; content is copied into the print SVGs. |
| `sticker-holo.svg` | Display holo sticker (512 canvas, foil-gradient frame on charcoal tile, live `<text>` in Inter). Source art — edit this. |
| `sticker-light.svg` | Display light sticker (512 canvas, bone tile, live `<text>`). Source art — edit this. |
| `sticker-holo-print.svg` | Holo **print file**: full-bleed 512 canvas (3in stock), no cutline. Text is outlined paths from `outline_text.py`; holo shows through transparent areas via a luminance mask. Terminal dot intentionally dropped in this variant. |
| `sticker-light-print.svg` | Light vinyl **print file**: full-bleed 512 canvas, outlined text from `outline_text.py --vinyl`, terminal dot kept and filled cyan. |
| `sticker-board.svg` | Hand-maintained overview board (1200x1200). References `sticker-light.svg`, `sticker-holo.svg`, and `../logo.svg` via relative `<image href>` — keep those files next to it. |
| `mbp-lid-showcase-{1,2,3}in.svg` | **Generated** showcases: stickers at true scale on a 14" MacBook Pro lid (12.31x8.71 in). Never hand-edit — run `gen_showcases.py`. |
| `thinkpad-lid-showcase-2in.svg` | **Generated** showcase: stickers on a ThinkPad X1 Carbon Gen 12 lid (12.31x8.45 in), diagonal placement (ThinkPad wordmark top-left, Lenovo badge bottom-right). Never hand-edit. |
| `mbp-lid-showcase-{1,2,3}in.png` | **Rendered artifacts** (2560x1440 PNG): agent-browser screenshots of the showcase SVGs (viewport `2560 1440`; the raw screenshot is the PNG export). |
| `sticker-light.png`, `sticker-light-print.png` | **Rendered artifacts** (700x700 PNG) of the corresponding display/print SVGs. |
| `sticker-holo-preview.svg` | **Preview** (not a print file): simulates the Sticker Mule holographic print with an animated rainbow fill knocking out where the holo shows through. For README/social previews. |

## Marks and assets

The showcase SVG files reference the Apple, ThinkPad (Lenovo), and Sticker Mule
marks, and use [simple-icons](https://simpleicons.org/) (CC0) assets. Those
marks belong to their respective owners; these files are fan-made previews and
imply no endorsement from or affiliation with Apple, Lenovo, Sticker Mule, or the
simple-icons trademark holders.

## Licensing and provenance

The generators edit geometry only; the bundled type is [SIL Open Font License
1.1](https://scripts.sil.org/OFL) (OFL-1.1).

| Asset | Origin | License |
|---|---|---|
| `../fonts/inter-variable.ttf` | Inter variable font, vendored at the `outline_text.py` pin (`google/fonts` commit, `Inter[opsz,wght].ttf`; sha256 verified on every load) | OFL-1.1 — copyright 2020 The Inter Project Authors (`https://github.com/rsms/inter`); full text in `../fonts/inter-OFL.txt` |
| simple-icons SVGs | [simple-icons](https://simpleicons.org/) | CC0 |

The vendored bytes are unmodified, so the OFL requires the copyright notice and
license text to travel with them (`../fonts/inter-OFL.txt`). The license permits
bundling, embedding, and redistributing the font and its derivatives provided it
is not sold by itself and stays under OFL-1.1.

## Regeneration

Prerequisites: `uv` (repo standard). `outline_text.py` needs fonttools.
Font loads vendored `../fonts/inter-variable.ttf` first (offline-safe), then
`~/.cache/chunkhound/inter-variable.ttf`, then network (2 retries).

```bash
# Scripts import their sibling numfmt.py, so run from this directory.
cd brand/stickers

# 1. Outline text for print files (font: Inter variable, wght=700, pinned to a
#    google/fonts commit + sha256 and cached at ~/.cache/chunkhound/inter-variable.ttf).
#    --out keeps the two modes' intermediates apart: the vinyl run would
#    otherwise overwrite the holo .outlined-text.svg.
uv run --with fonttools python outline_text.py  # holo runs -> .outlined-text.svg
uv run --with fonttools python outline_text.py --vinyl --out .outlined-text-vinyl.svg
# Copy each output's <g> groups into the matching print SVG (manual step).

# 2. Regenerate showcase SVGs (re-run after editing sticker-light.svg or
#    sticker-holo.svg — showcases embed them as base64 data URIs).
uv run python gen_showcases.py
# -> mbp-lid-showcase-{1,2,3}in.svg, thinkpad-lid-showcase-2in.svg

# 3. Render PNGs: open each showcase SVG in agent-browser with viewport
#    2560 1440 and screenshot. The 700x700 PNGs are screenshots of their
#    SVGs at 700px (exact capture command not scripted — recreate ad hoc).
```

## Rules

- **Sources (hand-editable):** `outline_text.py`, `gen_showcases.py`, `numfmt.py`,
  `sticker-holo.svg`, `sticker-light.svg`, `sticker-holo-print.svg`,
  `sticker-light-print.svg`, `sticker-board.svg`.
- **Never hand-edit (generated):** the four lid-showcase SVGs and
  `.outlined-text*.svg` — always regenerate with the scripts.
- **Never hand-edit (rendered):** all PNGs — re-screenshot from the SVGs.
- PNGs and `.outlined-text*.svg` are gitignored (see root `.gitignore`): they
  are rendered locally from SVGs. Only SVGs + generator scripts are committed.
  Re-render PNGs locally after editing source SVGs — do not commit them.

Known approximations (documented in `gen_showcases.py`): ThinkPad wordmark /
Lenovo badge lid coordinates are not published and are close approximations
from product imagery; lid dimensions come from Lenovo PSREF.
