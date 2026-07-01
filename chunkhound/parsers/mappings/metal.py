"""Metal Shading Language mapping for the unified parser architecture.

MSL is a C++14 dialect, so the C++ tree-sitter grammar parses its real
structure (functions, structs, templates). This mapping reuses CppMapping
wholesale and only re-tags emitted chunks as Metal so a dedicated
tree-sitter-metal grammar can be swapped in later by changing one line
(cheap path).
"""

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.cpp import CppMapping


class MetalMapping(CppMapping):
    """C++-backed mapping tagged as Metal (.metal). See module docstring."""

    def __init__(self) -> None:
        super().__init__()
        # CppMapping.__init__ hardcodes Language.CPP; re-tag so chunks are METAL.
        self.language = Language.METAL
