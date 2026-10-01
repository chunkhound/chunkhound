"""Shared compact float formatting for the sticker SVG generators.

WHY: font instancer/geometry floats are long; trimming keeps SVGs small
without visible precision loss at sticker scale. Single implementation per
the DRY rule (previously duplicated as ntos in outline_text.py and fnum in
gen_showcases.py).
"""


def fnum(v: float, decimals: int = 1) -> str:
    """Compact float: fixed decimals, trailing zeros trimmed, "0" fallback."""
    s = f"{v:.{decimals}f}"
    # Trim only the fractional part: at decimals=0 there is no ".", and rstrip
    # would otherwise eat the integer's trailing zeros (10 -> "1").
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    # ""/"-"/"-0" are all sign-preserving renderings of zero.
    return "0" if s in ("", "-", "-0") else s
