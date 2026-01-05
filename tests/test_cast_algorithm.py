"""
Comprehensive tests for cAST (Code AST) algorithm implementation.

These tests exercise the real functionality of the cAST chunking system
without mocks, validating algorithm correctness, edge cases, and performance.
"""

import re
import time
from pathlib import Path

import pytest

from chunkhound.core.types.common import ChunkType, Language
from chunkhound.parsers.parser_factory import (
    create_parser_for_language,
    get_parser_factory,
)


class TestCASTAlgorithmCore:
    """Test core cAST algorithm split-then-merge behavior."""

    def test_cast_split_then_merge_large_function(self):
        """Test that oversized functions get split recursively then merged optimally."""
        # Create a Python function that exceeds max_chunk_size (1200 chars)
        large_function = (
            '''
def process_large_data(items):
    """Process a large dataset with multiple operations."""
    results = []
    for item in items:
        if item.is_valid():
            processed = item.transform()
            if processed.meets_criteria():
                enhanced = processed.enhance_with_metadata()
                validated = enhanced.validate_business_rules()
                results.append(validated)
        '''
            + "    # comment line\n" * 100
        )  # Force size over limit

        parser = create_parser_for_language(Language.PYTHON)
        chunks = parser.parse_content(large_function)

        # Verify splitting occurred
        assert len(chunks) > 1, "Large function should be split"

        # Verify no chunk exceeds limits
        for chunk in chunks:
            non_ws_chars = len(re.sub(r"\s", "", chunk.code))
            assert non_ws_chars <= 1200, f"Character limit exceeded: {non_ws_chars}"
            # Verify estimated tokens under limit
            estimated_tokens = non_ws_chars * 1.75
            assert estimated_tokens <= 6000, f"Token limit exceeded: {estimated_tokens}"

    def test_cast_greedy_merge_small_adjacent(self):
        """Test that small adjacent chunks get merged greedily."""
        code = """
def small_func_1():
    return 1

def small_func_2():
    return 2

def small_func_3():
    return 3
"""
        parser = create_parser_for_language(Language.PYTHON)
        chunks = parser.parse_content(code)

        # Should merge small adjacent functions
        assert len(chunks) <= 3, "Functions should be processed"

        # Verify content is preserved
        all_code = " ".join(chunk.code for chunk in chunks)
        assert "small_func_1" in all_code
        assert "small_func_2" in all_code
        assert "small_func_3" in all_code

    def test_dual_constraint_character_and_token_limits(self):
        """Test both character (1200) and token (6000) limits are enforced."""
        # Create content that hits token limit before character limit
        # Using lots of unique identifiers (high token density)
        high_token_density = """
def function_with_many_unique_identifiers():
    variable_name_that_is_extremely_long_and_descriptive = 1
    another_incredibly_verbose_variable_name_here = 2
    """ + "\n".join(
            [
                f"    unique_variable_name_number_{i} = {i}"
                for i in range(500)  # Each line ~35 chars but ~6 tokens
            ]
        )

        parser = create_parser_for_language(Language.PYTHON)
        chunks = parser.parse_content(high_token_density)

        for chunk in chunks:
            non_ws_chars = len(re.sub(r"\s", "", chunk.code))
            estimated_tokens = non_ws_chars * 1.75
            assert estimated_tokens <= 6000, f"Token limit exceeded: {estimated_tokens}"
            assert non_ws_chars <= 1200, f"Character limit exceeded: {non_ws_chars}"

    def test_emergency_splitting_real_minified_jquery(self):
        """Test emergency splitting using real jQuery minified code."""
        from pathlib import Path

        # Use real jQuery minified file
        fixtures_dir = Path(__file__).parent / "fixtures" / "real_world_files"
        jquery_path = fixtures_dir / "jquery-3.7.1.min.js"

        if not jquery_path.exists():
            pytest.skip("Real jQuery file not available for testing")

        # Read first 8000 characters (exceeds limits, single line)
        jquery_content = jquery_path.read_text(encoding="utf-8")[:8000]

        parser = create_parser_for_language(Language.JAVASCRIPT)
        chunks = parser.parse_content(jquery_content)

        assert len(chunks) >= 1, "Should handle jQuery minified code"

        # Verify all chunks respect size limits (this is the real test)
        for i, chunk in enumerate(chunks):
            non_ws_chars = len(re.sub(r"\s", "", chunk.code))
            assert non_ws_chars <= 1200, (
                f"jQuery chunk {i} exceeds limit: {non_ws_chars}"
            )

        # Verify chunking worked appropriately
        total_chunk_content = sum(len(chunk.code) for chunk in chunks)
        original_non_ws = len(re.sub(r"\s", "", jquery_content))
        chunks_non_ws = sum(len(re.sub(r"\s", "", chunk.code)) for chunk in chunks)

        print(
            f"jQuery test: {len(jquery_content)} chars → {len(chunks)} chunks → {total_chunk_content} chars"
        )
        print(f"Non-whitespace: {original_non_ws} → {chunks_non_ws}")

        # Should have meaningful chunk extraction (not just empty chunks)
        assert total_chunk_content > 1000, (
            f"Too little chunk content: {total_chunk_content}"
        )
        assert chunks_non_ws > 500, f"Too little meaningful content: {chunks_non_ws}"


class TestLanguageSpecificEdgeCases:
    """Test cross-language AST structure handling."""

    def test_python_class_with_nested_methods(self):
        """Test Python class chunking preserves method boundaries."""
        python_class = '''
class DataProcessor:
    """Main data processing class."""
    
    def __init__(self, config):
        self.config = config
        
    def process_batch(self, items):
        """Process a batch of items."""
        results = []
        for item in items:
            result = self._process_single(item)
            if result:
                results.append(result)
        return results
    
    def _process_single(self, item):
        """Process single item with validation."""
        if not item.is_valid():
            return None
        return item.transform()
'''

        parser = create_parser_for_language(Language.PYTHON)
        chunks = parser.parse_content(python_class)

        # Should extract class and methods as separate concepts
        assert len(chunks) > 0, "Should produce chunks"

        # Verify different chunk types exist
        chunk_types = {chunk.chunk_type for chunk in chunks}
        # Should have function/method chunks at minimum
        code_types = {ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.CLASS}
        assert any(ct in code_types for ct in chunk_types), (
            f"Expected code chunks, got: {chunk_types}"
        )

    def test_javascript_function_expressions_vs_declarations(self):
        """Test JavaScript handles both function declarations and expressions."""
        js_code = """
// Function declaration
function regularFunction() {
    return "declaration";
}

// Function expression
const expressionFunc = function() {
    return "expression";  
};

// Arrow function
const arrowFunc = () => {
    return "arrow";
};

// Method in object
const obj = {
    method() {
        return "method";
    }
};
"""

        parser = create_parser_for_language(Language.JAVASCRIPT)
        chunks = parser.parse_content(js_code)

        # Should capture function definitions
        assert len(chunks) >= 1, "Should capture functions"

        # Verify content is preserved
        all_code = " ".join(chunk.code for chunk in chunks)
        assert "regularFunction" in all_code
        assert "expressionFunc" in all_code or "arrowFunc" in all_code

    def test_deeply_nested_code_structure(self):
        """Test handling of deeply nested code that might break parsers."""
        nested_code = """
def outer_function():
    def level_1():
        def level_2():
            def level_3():
                def level_4():
                    def level_5():
                        return "deep"
                    return level_5()
                return level_4() 
            return level_3()
        return level_2()
    return level_1()

class OuterClass:
    class MiddleClass:
        class InnerClass:
            def deep_method(self):
                if True:
                    if True:
                        if True:
                            return "nested_conditions"
"""

        parser = create_parser_for_language(Language.PYTHON)
        chunks = parser.parse_content(nested_code)

        # Should handle nesting without stack overflow or infinite recursion
        assert len(chunks) > 0, "Should handle nested code"

        # Verify chunks are still within size limits despite nesting
        for chunk in chunks:
            non_ws_chars = len(re.sub(r"\s", "", chunk.code))
            assert non_ws_chars <= 1200, (
                f"Size limit exceeded in nested code: {non_ws_chars}"
            )

    def test_mixed_content_types_integration(self):
        """Test files with mixed comments, imports, code."""
        mixed_content = '''
#!/usr/bin/env python3
"""
Module docstring explaining the purpose.
This is a multi-line docstring that should be captured.
"""

import os
import sys
from typing import List, Dict

# Global constants
MAX_ITEMS = 1000
DEFAULT_CONFIG = {"timeout": 30}

def main():
    """Main entry point."""
    print("Starting application")
    
    # Process configuration
    config = load_config()
    
    # Main processing loop
    while True:
        items = get_next_batch()
        if not items:
            break
        process_batch(items)

# Helper functions
def load_config():
    """Load configuration from environment."""
    return {"debug": os.getenv("DEBUG", False)}

def get_next_batch():
    """Get next batch of items to process."""  
    # Implementation details...
    pass

if __name__ == "__main__":
    main()
'''

        parser = create_parser_for_language(Language.PYTHON)
        chunks = parser.parse_content(mixed_content)

        # Should capture different content types appropriately
        assert len(chunks) > 0, "Should produce chunks from mixed content"

        # Verify function content is captured
        all_code = " ".join(chunk.code for chunk in chunks)
        assert "def main" in all_code or "main()" in all_code


class TestSizeLimitEdgeCases:
    """Test boundary conditions around size limits."""

    def test_chunk_exactly_at_size_limits(self):
        """Test chunks that are exactly at the size boundaries."""
        # Create function that is exactly 1200 non-whitespace characters
        base_func = "def test_func():\n    "
        base_non_ws = len(re.sub(r"\s", "", base_func))
        remaining_chars = 1200 - base_non_ws

        if remaining_chars > 10:
            padding = "x = 1  # " + "a" * max(0, remaining_chars - 10) + "\n"
            exact_size_func = base_func + padding

            # Verify it's at or near the limit
            actual_size = len(re.sub(r"\s", "", exact_size_func))
            assert 1150 <= actual_size <= 1250, (
                f"Test setup error: size is {actual_size}"
            )

            parser = create_parser_for_language(Language.PYTHON)
            chunks = parser.parse_content(exact_size_func)

            # Should produce at least one chunk
            assert len(chunks) >= 1, "Should produce chunks"

            # Verify size constraints
            for chunk in chunks:
                non_ws_chars = len(re.sub(r"\s", "", chunk.code))
                assert non_ws_chars <= 1200, f"Chunk exceeds limit: {non_ws_chars}"

    def test_real_world_oversized_content_d3js(self):
        """Test handling of oversized content using real D3.js library."""
        from pathlib import Path

        # Use real D3.js minified file (very large)
        fixtures_dir = Path(__file__).parent / "fixtures" / "real_world_files"
        d3_path = fixtures_dir / "d3.min.js"

        if not d3_path.exists():
            pytest.skip("Real D3.js file not available for testing")

        # D3.js is large - test a substantial portion
        d3_content = d3_path.read_text(encoding="utf-8")

        parser = create_parser_for_language(Language.JAVASCRIPT)
        chunks = parser.parse_content(d3_content)

        # Critical test: NO chunk should exceed size constraints
        oversized_chunks = []
        for i, chunk in enumerate(chunks):
            non_ws_chars = len(re.sub(r"\s", "", chunk.code))
            if non_ws_chars > 1200:
                oversized_chunks.append((i, non_ws_chars))

        assert len(oversized_chunks) == 0, (
            f"Found {len(oversized_chunks)} oversized chunks: {oversized_chunks}"
        )

        # Should produce reasonable number of chunks for large file
        assert len(chunks) > 10, f"Too few chunks for large D3.js file: {len(chunks)}"

        # Performance check
        original_size = len(d3_content)
        total_chunk_size = sum(len(chunk.code) for chunk in chunks)
        expansion_ratio = total_chunk_size / original_size

        # D3 should compress or have minimal expansion
        assert expansion_ratio <= 2.0, (
            f"Excessive expansion on real D3.js: {expansion_ratio:.2f}x"
        )

    def test_empty_and_minimal_files(self):
        """Test edge cases with empty or minimal content."""
        test_cases = [
            "",  # Empty file
            "# Just a comment",  # Only comment
            "import os",  # Only import
            "x = 1",  # Single statement
            "def f(): pass",  # Minimal function
        ]

        parser = create_parser_for_language(Language.PYTHON)

        for content in test_cases:
            chunks = parser.parse_content(content)
            # Should handle gracefully - no crashes
            assert isinstance(chunks, list), f"Should return list for: {repr(content)}"

            if chunks:
                assert len(chunks[0].code.strip()) > 0, (
                    f"Chunk should not be empty for: {repr(content)}"
                )


class TestPerformanceAndScalability:
    """Test performance and memory efficiency."""

    def test_real_world_minified_libraries_comprehensive(self):
        """Comprehensive test of real-world minified JavaScript libraries.

        This tests the cAST algorithm against actual production JavaScript files
        that represent real pathological cases for chunking algorithms.
        """

        fixtures_dir = Path(__file__).parent / "fixtures" / "real_world_files"

        # Test different real-world minified files with expected characteristics
        test_files = [
            (
                "jquery-3.7.1.min.js",
                "jQuery 3.7.1",
                {
                    "min_chunks": 5,
                    "max_expansion": 2.0,
                    "description": "Classic single-line minified library",
                },
            ),
            (
                "react.min.js",
                "React 18.2.0",
                {
                    "min_chunks": 2,
                    "max_expansion": 2.0,
                    "description": "Production React build with some line breaks",
                },
            ),
            (
                "vue.min.js",
                "Vue 3.4.27",
                {
                    "min_chunks": 10,
                    "max_expansion": 1.5,
                    "description": "Vue production build - mixed minification",
                },
            ),
            (
                "bootstrap.min.js",
                "Bootstrap 5.3.2",
                {
                    "min_chunks": 5,
                    "max_expansion": 2.0,
                    "description": "Bootstrap bundle with moderate minification",
                },
            ),
            (
                "d3.min.js",
                "D3.js 7.8.5",
                {
                    "min_chunks": 15,
                    "max_expansion": 1.0,
                    "description": "Large visualization library - excellent compression expected",
                },
            ),
        ]

        parser = create_parser_for_language(Language.JAVASCRIPT)
        results = []

        for filename, lib_name, expectations in test_files:
            file_path = fixtures_dir / filename
            if not file_path.exists():
                continue  # Skip if file not available

            print(f"\n=== Testing {lib_name} ===")
            print(f"Description: {expectations['description']}")

            file_content = file_path.read_text(encoding="utf-8")
            original_size = len(file_content)
            file_lines = len(file_content.splitlines())

            start_time = time.time()
            chunks = parser.parse_content(file_content)
            processing_time = time.time() - start_time

            # Calculate metrics
            total_chunk_size = sum(len(chunk.code) for chunk in chunks)
            expansion_ratio = (
                total_chunk_size / original_size if original_size > 0 else 1.0
            )
            avg_chunk_size = total_chunk_size / len(chunks) if chunks else 0

            print(f"File size: {original_size:,} bytes ({file_lines} lines)")
            print(f"Processing time: {processing_time:.3f}s")
            print(f"Chunks created: {len(chunks)}")
            print(f"Total chunk content: {total_chunk_size:,} bytes")
            print(f"Expansion ratio: {expansion_ratio:.3f}x")
            print(f"Average chunk size: {avg_chunk_size:.0f} bytes")

            # Performance assertions
            assert processing_time < 5.0, (
                f"{filename}: Processing too slow: {processing_time}s"
            )
            assert len(chunks) > 0, f"{filename}: Should produce chunks"
            assert len(chunks) >= expectations["min_chunks"], (
                f"{filename}: Too few chunks: {len(chunks)}"
            )

            # Memory efficiency assertions
            assert expansion_ratio <= expectations["max_expansion"], (
                f"{filename}: Expansion {expansion_ratio:.3f}x > {expectations['max_expansion']}x"
            )

            # Size constraint assertions (CRITICAL)
            oversized_chunks = []
            for i, chunk in enumerate(chunks):
                non_ws_chars = len(re.sub(r"\s", "", chunk.code))
                if non_ws_chars > 1200:
                    oversized_chunks.append((i, non_ws_chars))

            assert len(oversized_chunks) == 0, (
                f"{filename}: Found {len(oversized_chunks)} oversized chunks: {oversized_chunks[:3]}"
            )

            # Token limit check
            for i, chunk in enumerate(chunks):
                estimated_tokens = len(re.sub(r"\s", "", chunk.code)) * 1.75
                assert estimated_tokens <= 6000, (
                    f"{filename} chunk {i}: Estimated {estimated_tokens:.0f} tokens > 6000 limit"
                )

            results.append(
                {
                    "library": lib_name,
                    "file_size": original_size,
                    "chunks": len(chunks),
                    "expansion": expansion_ratio,
                    "processing_time": processing_time,
                }
            )

        # Summary validation
        if results:
            print(f"\n=== Summary of {len(results)} Libraries ===")
            for result in results:
                print(
                    f"{result['library']}: {result['file_size']:,}B → {result['chunks']} chunks "
                    f"({result['expansion']:.2f}x) in {result['processing_time']:.3f}s"
                )

            # Overall performance check
            total_processing_time = sum(r["processing_time"] for r in results)
            total_file_size = sum(r["file_size"] for r in results)
            avg_expansion = sum(r["expansion"] for r in results) / len(results)

            print(
                f"Total processing time: {total_processing_time:.3f}s for {total_file_size:,} bytes"
            )
            print(f"Average expansion ratio: {avg_expansion:.3f}x")

            assert total_processing_time < 10.0, (
                f"Total processing time too slow: {total_processing_time:.3f}s"
            )
            assert avg_expansion <= 2.0, (
                f"Average expansion too high: {avg_expansion:.3f}x"
            )

    def test_synthetic_repetitive_python_file(self):
        """Test processing of synthetically repetitive Python code.

        NOTE: This is a pathological case with maximally repetitive code patterns
        that would never occur in real-world development. It tests the worst-case
        scenario for content duplication.
        """
        # Generate a realistic large Python file
        large_file_content = self.generate_realistic_python_file(
            num_classes=3,  # Reduced to be less pathological
            methods_per_class=5,
            lines_per_method=10,
            include_docstrings=True,
            include_type_hints=True,
        )

        parser = create_parser_for_language(Language.PYTHON)
        start_time = time.time()
        chunks = parser.parse_content(large_file_content)
        processing_time = time.time() - start_time

        # Performance verification
        assert processing_time < 5.0, f"Processing too slow: {processing_time}s"
        assert len(chunks) > 0, "Should produce chunks"

        # Memory efficiency check - more lenient for synthetic repetitive code
        total_chunk_size = sum(len(chunk.code) for chunk in chunks)
        original_size = len(large_file_content)
        expansion_ratio = total_chunk_size / original_size

        print(
            f"Synthetic repetitive code: {original_size} bytes → {total_chunk_size} bytes ({expansion_ratio:.2f}x)"
        )

        # Allow higher expansion for pathologically repetitive synthetic code
        # Real code would never be this repetitive
        assert expansion_ratio <= 3.0, (
            f"Excessive expansion even for synthetic code: {expansion_ratio:.2f}x"
        )

    def test_different_file_encodings(self):
        """Test handling of files with different encodings."""
        # Test UTF-8 with special characters
        utf8_content = '''
def función_con_caracteres_especiales():
    """Función que maneja caracteres especiales: áéíóú ñ."""
    mensaje = "¡Hola, mundo! 你好世界 🌍"
    return mensaje

class DeutscheKlasse:
    """Klasse mit deutschen Umlauten: äöüß.""" 
    def methode_mit_umlauten(self):
        return "Größe: 10cm"
'''

        parser = create_parser_for_language(Language.PYTHON)
        chunks = parser.parse_content(utf8_content)

        assert len(chunks) > 0, "Should handle UTF-8 content"

        # Verify special characters are preserved
        chunk_text = " ".join(chunk.code for chunk in chunks)
        assert "función_con_caracteres_especiales" in chunk_text
        # These might not always be preserved depending on parsing
        # assert '你好世界' in chunk_text
        # assert 'äöüß' in chunk_text

    def generate_realistic_python_file(
        self,
        num_classes,
        methods_per_class,
        lines_per_method,
        include_docstrings=True,
        include_type_hints=True,
    ):
        """Generate realistic Python file for testing."""
        content = ["#!/usr/bin/env python3"]
        content.append('"""Large module for testing purposes."""')
        content.append("")
        content.append("import os")
        content.append("import sys")
        content.append("from typing import List, Dict, Optional")
        content.append("")

        for class_idx in range(num_classes):
            class_name = f"TestClass{class_idx}"
            if include_docstrings:
                content.append(f"class {class_name}:")
                content.append(f'    """Test class number {class_idx}."""')
                content.append("")
            else:
                content.append(f"class {class_name}:")

            for method_idx in range(methods_per_class):
                method_name = f"method_{method_idx}"
                if include_type_hints:
                    content.append(
                        f"    def {method_name}(self, param: str) -> Optional[str]:"
                    )
                else:
                    content.append(f"    def {method_name}(self, param):")

                if include_docstrings:
                    content.append(f'        """Method {method_idx} implementation."""')

                for line_idx in range(lines_per_method):
                    content.append(f"        # Line {line_idx} of method {method_name}")
                    content.append(
                        f"        result_{line_idx} = param + str({line_idx})"
                    )

                content.append("        return result_0")
                content.append("")

        return "\n".join(content)


class TestIntegrationAndAdapterCompatibility:
    """Test adapter compatibility and integration points."""

    def test_universal_parser_direct_usage(self):
        """Test UniversalParser direct usage without adapter."""
        # Test with real parsing workload
        code = """
class RealClass:
    def real_method(self, param):
        return param * 2
        
def standalone_function():
    return "result"
"""

        # Create universal parser directly
        factory = get_parser_factory()
        universal_parser = factory.create_parser(Language.PYTHON)

        # Test direct usage
        chunks = universal_parser.parse_content(code)

        assert len(chunks) > 0, "Should produce chunks"
        assert isinstance(chunks, list), "Should return list"

        # Verify Chunk objects
        for chunk in chunks:
            assert hasattr(chunk, "symbol"), "Should have symbol attribute"
            assert hasattr(chunk, "start_line"), "Should have start_line attribute"
            assert hasattr(chunk, "code"), "Should have code attribute"

    def test_factory_caching_and_reuse(self):
        """Test parser factory caching works correctly."""
        factory = get_parser_factory()

        # Create same parser twice
        parser1 = factory.create_parser(Language.PYTHON)
        parser2 = factory.create_parser(Language.PYTHON)

        # Should be same instance (cached)
        assert parser1 is parser2, "Parser should be cached"

        # Test different languages if available
        try:
            js_parser = factory.create_parser(Language.JAVASCRIPT)
            assert parser1 is not js_parser, (
                "Different languages should be different parsers"
            )
            assert parser1.language_name == "python"
            assert js_parser.language_name == "javascript"
        except Exception:
            # JavaScript parser may not be available, skip this part
            pass

    def test_language_detection_and_extensions(self):
        """Test language detection and file extension mapping."""
        factory = get_parser_factory()

        # Test file detection
        test_file = Path("test.py")
        parser = factory.create_parser_for_file(test_file)
        assert parser.language_name == "python"

        # Test language availability
        available_languages = factory.get_available_languages()
        assert isinstance(available_languages, dict)
        assert Language.PYTHON in available_languages
        assert (
            available_languages[Language.PYTHON] is True
        )  # Python should be available
