#!/usr/bin/env python3
"""Test script to compare TypeScript and Python parsing."""

import tempfile
import shutil
from pathlib import Path
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.core.types.common import Language

def test_file_parsing(file_path: Path, language: Language):
    """Parse a file and print extracted chunks."""
    print(f"\n{'='*80}")
    print(f"Parsing: {file_path.name} (Language: {language.value})")
    print(f"{'='*80}\n")

    parser = create_parser_for_language(language)

    with open(file_path, 'rb') as f:
        content = f.read()

    chunks = parser.parse_file(file_path, content)

    print(f"Total chunks extracted: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk #{i}:")
        print(f"  Symbol: {chunk.symbol}")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"  Metadata: {chunk.metadata}")
        if chunk.code:
            preview = chunk.code[:100].replace('\n', ' ')
            if len(chunk.code) > 100:
                preview += '...'
            print(f"  Code preview: {preview}")
        print()

    return chunks

if __name__ == '__main__':
    # Test TypeScript parsing
    ts_file = Path(__file__).parent / 'test_ts_parsing.ts'
    ts_chunks = test_file_parsing(ts_file, Language.TYPESCRIPT)

    # Test Python parsing
    py_file = Path(__file__).parent / 'test_py_parsing.py'
    py_chunks = test_file_parsing(py_file, Language.PYTHON)

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"TypeScript chunks: {len(ts_chunks)}")
    print(f"Python chunks: {len(py_chunks)}")
    print()

    # Show what was extracted
    print("TypeScript symbols:")
    for chunk in ts_chunks:
        print(f"  - {chunk.symbol} ({chunk.chunk_type})")

    print("\nPython symbols:")
    for chunk in py_chunks:
        print(f"  - {chunk.symbol} ({chunk.chunk_type})")
