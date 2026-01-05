#!/usr/bin/env python3
"""Simple test for Vue cross-reference analysis."""

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkType, FileId, Language, LineNumber
from chunkhound.parsers.vue_cross_ref import (
    build_symbol_table,
    match_template_references,
)


def main():
    """Test cross-reference analysis directly."""
    print("Testing Vue cross-reference analysis...")

    # Create script chunks
    script_chunks = [
        Chunk(
            symbol="test",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="const message = ref('Hello')",
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
        ),
        Chunk(
            symbol="test",
            start_line=LineNumber(2),
            end_line=LineNumber(2),
            code="const count = ref(0)",
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
        ),
        Chunk(
            symbol="increment",
            start_line=LineNumber(5),
            end_line=LineNumber(7),
            code="function increment() {\n  count.value++\n}",
            chunk_type=ChunkType.FUNCTION,
            file_id=FileId(1),
            language=Language.VUE,
        ),
    ]

    print(f"\nBuilding symbol table from {len(script_chunks)} chunks...")
    symbol_table = build_symbol_table(script_chunks)

    print(f"Variables: {list(symbol_table.variables.keys())}")
    print(f"Functions: {list(symbol_table.functions.keys())}")

    # Create template chunks
    template_chunks = [
        Chunk(
            symbol="{{ message }}",
            start_line=LineNumber(10),
            end_line=LineNumber(10),
            code="{{ message }}",
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={"interpolation_expression": "message"},
        ),
        Chunk(
            symbol="@click",
            start_line=LineNumber(11),
            end_line=LineNumber(11),
            code='<button @click="increment">{{ count }}</button>',
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={
                "handler_expression": "increment",
                "interpolation_expression": "count",
            },
        ),
    ]

    print(f"\nMatching references in {len(template_chunks)} template chunks...")
    updated_chunks = match_template_references(template_chunks, symbol_table)

    for chunk in updated_chunks:
        print(f"\nChunk: {chunk.symbol}")
        if "script_references" in chunk.metadata:
            print(f"  References: {chunk.metadata['script_references']}")
        if "undefined_references" in chunk.metadata:
            print(f"  Undefined: {chunk.metadata['undefined_references']}")

    print("\nTest complete!")


if __name__ == "__main__":
    main()
