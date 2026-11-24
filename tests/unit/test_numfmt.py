"""Contract tests for brand/stickers/numfmt.py's compact float formatting.

numfmt is a leaf utility shared by the sticker SVG generators (outline_text.py,
gen_showcases.py): its return value is written verbatim into SVG geometry. It
lives outside any package, so it is loaded by path — same pattern as
tests/contracts/test_install_native.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

NUMFMT_PATH = Path(__file__).resolve().parents[2] / "brand" / "stickers" / "numfmt.py"


def _load_fnum():
    spec = importlib.util.spec_from_file_location("numfmt", NUMFMT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fnum


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        # Negative zero must not reach SVG geometry as "-0" or "-".
        (-0.04, 1, "0"),
        (-0.0, 1, "0"),
        (-0.4, 0, "0"),
        # decimals=0 must trim the fraction, not the integer's trailing zeros.
        (10, 0, "10"),
        (100, 0, "100"),
        (-10, 0, "-10"),
        # Ordinary values keep their sign and drop only the fraction's zeros.
        (-0.04, 3, "-0.04"),
        (-0.5, 1, "-0.5"),
        (-12.30, 1, "-12.3"),
        (2.0, 1, "2"),
        (1.5, 1, "1.5"),
        (0.0, 1, "0"),
    ],
)
def test_fnum_formats_compactly(value: float, decimals: int, expected: str) -> None:
    assert _load_fnum()(value, decimals) == expected
