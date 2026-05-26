"""Shared utilities for PNG dimension assertions in site tests."""

import pathlib


def png_dimensions(png_path: pathlib.Path) -> tuple[int, int]:
    """Read PNG IHDR chunk to extract width and height."""
    with open(png_path, "rb") as f:
        sig = f.read(8)
        assert sig == b"\x89PNG\r\n\x1a\n", f"Not a valid PNG file: {png_path}"
        # IHDR starts at byte 16: 8-byte signature + 4-byte length + 4-byte chunk type
        f.seek(16)
        data = f.read(8)
        width = int.from_bytes(data[0:4], "big")
        height = int.from_bytes(data[4:8], "big")
    return (width, height)
