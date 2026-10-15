"""Contract tests for format_bytes() user-visible output.

Binary units, one decimal, GB is the terminal unit — values above GiB keep
rendering as GB, never TB.
"""

import pytest

from chunkhound.services.progress_utils import format_bytes


@pytest.mark.parametrize(
    ("byte_count", "expected"),
    [
        (0, "0.0B"),
        (1, "1.0B"),
        (1023, "1023.0B"),
        (1024, "1.0KB"),
        (1024**2, "1.0MB"),
        (1024**3, "1.0GB"),
        # Docstring example.
        (9.8 * 1024**3, "9.8GB"),
        # GB is terminal: even 5TiB must render as GB, never TB.
        (5 * 1024**4, "5120.0GB"),
    ],
)
def test_format_bytes_exact_output(byte_count: float, expected: str) -> None:
    assert format_bytes(byte_count) == expected
