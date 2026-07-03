"""Tests for PowerShell language mapping and parsing.

These are contract tests through the real parser: they fail on a build that
routes .ps1 through the UNKNOWN fallback (which only emits generic chunks) and
pass once PowerShell is wired with function/class/method-aware chunking.
"""

from pathlib import Path

from chunkhound.core.detection import detect_language
from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import get_parser_factory


def parse_powershell(content: str):
    """Parse PowerShell content through the real parser factory."""
    factory = get_parser_factory()
    parser = factory.create_parser(Language.POWERSHELL)
    return parser.parse_content(content, "test.ps1", FileId(1))


def test_ps1_routes_to_powershell():
    """.ps1 / .psm1 must resolve to Language.POWERSHELL (not UNKNOWN)."""
    assert detect_language(Path("x.ps1")) == Language.POWERSHELL
    assert detect_language(Path("x.psm1")) == Language.POWERSHELL


class TestPowerShellTypedChunks:
    """The fallback parser cannot emit these typed symbols — only the real
    PowerShell mapping can."""

    def test_function_class_method_chunks(self):
        content = """function Get-Foo {
    param([string]$Name)
    Write-Output "Hello $Name"
    Write-Output "again $Name"
}

class Widget {
    [int]$Size

    [int] Spin() {
        $result = $this.Size * 2
        Write-Output "spinning to $result"
        return $result
    }
}
"""
        chunks = parse_powershell(content)

        def has(chunk_type, symbol):
            return any(
                c.chunk_type == chunk_type and c.symbol == symbol for c in chunks
            )

        symbols = sorted(c.symbol for c in chunks)
        assert has(ChunkType.FUNCTION, "Get-Foo"), symbols
        assert has(ChunkType.CLASS, "Widget"), symbols
        assert has(ChunkType.METHOD, "Spin"), symbols
        assert has(ChunkType.PROPERTY, "Size"), symbols

    def test_filter_is_a_function(self):
        """PowerShell `filter` uses function_statement and chunks as FUNCTION."""
        content = """filter Convert-Upper {
    $_.ToString().ToUpper()
    Write-Output "converted"
}
"""
        chunks = parse_powershell(content)
        assert any(
            c.chunk_type == ChunkType.FUNCTION and c.symbol == "Convert-Upper"
            for c in chunks
        ), sorted(c.symbol for c in chunks)

    def test_enum_chunk(self):
        content = """enum Color {
    Red
    Green
    Blue
}
"""
        chunks = parse_powershell(content)
        assert any(
            c.chunk_type == ChunkType.ENUM and c.symbol == "Color" for c in chunks
        ), sorted(c.symbol for c in chunks)
