"""Shared compact float formatting for the sticker SVG generators.

WHY: font instancer/geometry floats are long; trimming keeps SVGs small
without visible precision loss at sticker scale. Single implementation per
the DRY rule (previously duplicated as ntos in outline_text.py and fnum in
gen_showcases.py).
"""


def fnum(v: float, decimals: int = 1) -> str:
    """Compact float: fixed decimals, trailing zeros trimmed, "0" fallback."""
    s = f"{v:.{decimals}f}".rstrip("0").rstrip(".")
    return s if s else "0"
