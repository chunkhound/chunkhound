"""Contract test for Metal (.metal) language support.

Metal Shading Language is C++14, so ChunkHound reuses the C++ tree-sitter
grammar via MetalMapping (cheap path). These tests pin the
user-visible contract: a real MLX kernel snippet, routed as Language.METAL,
yields cpp-*typed* CLASS/FUNCTION chunks by name — symbols the UNKNOWN
fallback parser cannot emit. They fail on main (no Language.METAL).

ERROR nodes around Metal-only qualifiers (thread/threadgroup address spaces,
[[kernel]] attributes) are expected; cAST chunks *around* them.
"""

from pathlib import Path

from chunkhound.core.detection.language_detector import detect_language
from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import get_parser_factory

# Real snippet from mlx/backend/metal/kernels/arg_reduce.metal — a templated
# struct plus a templated reducer method, both verified to be captured by the
# cpp grammar.
METAL_SNIPPET = """
#include <metal_simdgroup>

using namespace metal;

template <typename U>
struct IndexValPair {
  uint32_t index;
  U val;
};

template <typename U>
struct ArgMax {
  IndexValPair<U> reduce(IndexValPair<U> best, IndexValPair<U> current) {
    if (best.val < current.val) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  IndexValPair<U>
  reduce_many(IndexValPair<U> best, thread U* vals, uint32_t offset) {
    for (int i = 0; i < N; i++) {
      if (vals[i] > best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};
"""


def parse_metal(content: str):
    parser = get_parser_factory().create_parser(Language.METAL)
    return parser.parse_content(content, "test.metal", FileId(1))


def test_metal_extension_routes_to_metal():
    """.metal files route to Language.METAL, not the UNKNOWN fallback."""
    assert detect_language(Path("kernel.metal")) == Language.METAL


def test_metal_captures_struct_as_class_chunk():
    """The templated struct is captured as a typed CLASS chunk by name."""
    chunks = parse_metal(METAL_SNIPPET)
    class_symbols = {c.symbol for c in chunks if c.chunk_type == ChunkType.CLASS}
    assert any("IndexValPair" in s for s in class_symbols), (
        f"Expected an IndexValPair CLASS chunk, got: {sorted(class_symbols)}"
    )


def test_metal_captures_method_as_function_chunk():
    """The templated reducer method is captured as a typed FUNCTION chunk."""
    chunks = parse_metal(METAL_SNIPPET)
    func_symbols = {c.symbol for c in chunks if c.chunk_type == ChunkType.FUNCTION}
    assert "reduce_many" in func_symbols, (
        f"Expected a 'reduce_many' FUNCTION chunk, got: {sorted(func_symbols)}"
    )


def test_metal_chunks_stay_tagged_metal():
    """Emitted chunks keep Language.METAL — not the cpp grammar's identity."""
    chunks = parse_metal(METAL_SNIPPET)
    assert chunks, "expected at least one chunk"
    assert all(c.language == Language.METAL for c in chunks), (
        f"chunks leaked non-METAL languages: "
        f"{sorted({c.language for c in chunks}, key=str)}"
    )


def test_metal_parse_result_tagged_metal(tmp_path):
    """ParseResult.language is METAL, matching file detection end-to-end."""
    f = tmp_path / "kernel.metal"
    f.write_text(METAL_SNIPPET)
    result = get_parser_factory().create_parser(Language.METAL).parse_with_result(
        f, FileId(1)
    )
    assert result.language == Language.METAL
