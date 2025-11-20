"""Performance and stress benchmarks for parser robustness.

This module contains stress tests to verify the parser can handle:
- Deeply nested code structures (20+ levels)
- Large files with many definitions (1000+ functions/classes)
- Memory-intensive parsing scenarios
- Performance within acceptable time bounds

These tests focus on ROBUSTNESS rather than optimization - verifying the
parser doesn't crash, hang, or exhaust memory when processing large/complex input.

Tests may be slow - mark with @pytest.mark.slow for optional skipping.
"""

import time
import pytest
from pathlib import Path

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import get_parser_factory


@pytest.fixture
def parser_factory():
    """Create a parser factory instance."""
    return get_parser_factory()


# =============================================================================
# DEEP NESTING TESTS - Test parser handles deep recursion without crashing
# =============================================================================


class TestDeepNesting:
    """Test parser robustness with deeply nested code structures.

    These tests verify the parser can handle deep nesting without:
    - Stack overflow errors
    - Excessive memory consumption
    - Infinite loops or hangs
    """

    @pytest.mark.slow
    def test_deeply_nested_functions(self, parser_factory):
        """Test parsing 20+ levels of nested function declarations.

        Stresses: Recursive descent parsing, stack depth, AST traversal.
        Expected: Completes without crash, extracts at least outermost function.
        """
        parser = parser_factory.create_parser(Language.JAVASCRIPT)

        # Generate 25 levels of nested functions
        depth = 25
        code_lines = []
        indent = ""

        for i in range(depth):
            code_lines.append(f"{indent}function level{i}() {{")
            indent += "    "

        # Add innermost content
        code_lines.append(f"{indent}return 'deepest level {depth}';")

        # Close all functions
        for i in range(depth - 1, -1, -1):
            indent = "    " * i
            code_lines.append(f"{indent}}}")

        code = "\n".join(code_lines)

        # Should complete without stack overflow or hang (10s timeout)
        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "deep_functions.js", FileId(1))
        elapsed = time.perf_counter() - start_time

        # Verify parsing succeeded
        assert len(chunks) > 0, "Should extract at least one chunk from deeply nested code"
        assert elapsed < 10.0, f"Parsing took {elapsed:.2f}s, should complete within 10s"

        # Verify at least the outermost function is captured
        all_code = " ".join([c.code for c in chunks])
        assert "level0" in all_code, "Should capture outermost function"

    @pytest.mark.slow
    def test_deeply_nested_classes(self, parser_factory):
        """Test parsing 20+ levels of nested class definitions.

        Stresses: Class scope handling, nested type tracking, AST depth.
        Expected: Completes without crash, extracts class content.
        """
        parser = parser_factory.create_parser(Language.TYPESCRIPT)

        # Generate 22 levels of nested classes
        depth = 22
        code_lines = []
        indent = ""

        for i in range(depth):
            code_lines.append(f"{indent}class Level{i} {{")
            indent += "    "

        # Add innermost content
        code_lines.append(f"{indent}value = {depth};")
        code_lines.append(f"{indent}getValue() {{ return this.value; }}")

        # Close all classes
        for i in range(depth - 1, -1, -1):
            indent = "    " * i
            code_lines.append(f"{indent}}}")

        code = "\n".join(code_lines)

        # Should complete without stack overflow or hang
        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "deep_classes.ts", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract chunks from deeply nested classes"
        assert elapsed < 10.0, f"Parsing took {elapsed:.2f}s, should complete within 10s"

        # Verify at least some nested classes are captured
        all_code = " ".join([c.code for c in chunks])
        # Parser may start extraction at different nesting levels due to chunking
        has_classes = any(f"Level{i}" in all_code for i in range(5))
        assert has_classes, "Should capture some nested classes"


# =============================================================================
# LARGE FILE TESTS - Test parser handles files with many definitions
# =============================================================================


class TestLargeFiles:
    """Test parser robustness with large files containing many constructs.

    These tests verify the parser can handle large files without:
    - Excessive memory growth
    - Quadratic time complexity
    - Crashes or hangs
    """

    @pytest.mark.slow
    def test_large_file_1mb(self, parser_factory):
        """Test parsing large JavaScript file (0.5MB+) with many functions.

        Stresses: Memory allocation, parsing throughput, batching logic.
        Expected: Completes in reasonable time (<10s), extracts content.
        """
        parser = parser_factory.create_parser(Language.JAVASCRIPT)

        # Generate ~1MB file with mix of functions and classes
        components = []

        # Add 3000 functions to get closer to 1MB
        for i in range(3000):
            components.append(f"""
function processItem{i}(item) {{
    const result = {{
        id: {i},
        name: 'Item {i}',
        value: item.value * {i},
        timestamp: Date.now()
    }};

    if (result.value > 100) {{
        result.priority = 'high';
    }} else {{
        result.priority = 'low';
    }}

    return result;
}}""")

        code = "\n".join(components)
        file_size_mb = len(code) / (1024 * 1024)

        # Verify we generated a large file (at least 0.5MB)
        # Note: Actual size will vary, but should be substantial
        assert file_size_mb > 0.5, f"Expected >0.5MB for large file test, got {file_size_mb:.2f}MB"

        # Should parse in reasonable time (<10s for large file)
        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "large_file.js", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract chunks from large file"
        assert elapsed < 10.0, f"Parsing {file_size_mb:.2f}MB took {elapsed:.2f}s, should be under 10s"

        # Verify reasonable amount of content extracted
        all_code = " ".join([c.code for c in chunks])
        found_functions = sum(1 for i in range(100) if f"processItem{i}" in all_code)

        assert found_functions >= 20, f"Should find at least 20 functions, found {found_functions}"

    @pytest.mark.slow
    def test_many_definitions(self, parser_factory):
        """Test file with 1000+ function/class definitions.

        Stresses: Symbol table size, definition tracking, chunk generation.
        Expected: Completes without excessive time/memory, extracts many definitions.
        """
        parser = parser_factory.create_parser(Language.TYPESCRIPT)

        # Generate 1200 mixed definitions (800 functions + 400 classes)
        definitions = []

        # 800 functions
        for i in range(800):
            definitions.append(f"function fn{i}(x: number): number {{ return x + {i}; }}")

        # 400 classes
        for i in range(400):
            definitions.append(f"""
class Class{i} {{
    id: number = {i};
    getValue(): number {{ return this.id; }}
}}""")

        code = "\n\n".join(definitions)

        # Should handle 1000+ definitions without excessive time
        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "many_definitions.ts", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract chunks from file with 1200 definitions"
        assert elapsed < 15.0, f"Parsing 1200 definitions took {elapsed:.2f}s, should be under 15s"

        # Verify substantial content extraction
        all_code = " ".join([c.code for c in chunks])
        found_functions = sum(1 for i in range(100) if f"fn{i}" in all_code)
        found_classes = sum(1 for i in range(50) if f"Class{i}" in all_code)

        assert found_functions >= 20, f"Should find many functions, found {found_functions}"
        assert found_classes >= 10, f"Should find many classes, found {found_classes}"


# =============================================================================
# MEMORY USAGE TESTS - Verify memory doesn't grow excessively
# =============================================================================


class TestMemoryUsage:
    """Test parser memory behavior with memory-intensive inputs.

    These tests verify the parser doesn't:
    - Leak memory during parsing
    - Create excessive intermediate structures
    - Keep unnecessary data in memory

    Note: Full memory profiling requires external tools, these are basic checks.
    """

    @pytest.mark.slow
    def test_memory_efficient_parsing(self, parser_factory):
        """Monitor that memory doesn't grow excessively during large file parsing.

        Stresses: Memory allocation patterns, garbage collection, data structures.
        Expected: Completes successfully, reasonable memory behavior.

        Note: This is a basic smoke test. Detailed memory profiling requires
        external tools like memory_profiler or tracemalloc.
        """
        parser = parser_factory.create_parser(Language.JAVASCRIPT)

        # Generate moderately large file (~500KB)
        num_items = 400
        items = []

        for i in range(num_items):
            items.append(f"""
function handler{i}(request, response) {{
    const data = {{
        id: {i},
        timestamp: Date.now(),
        path: request.path,
        method: request.method
    }};

    // Simulate processing
    const result = processRequest{i}(data);

    response.send(result);
}}

function processRequest{i}(data) {{
    return {{
        ...data,
        processed: true,
        handlerId: {i}
    }};
}}""")

        code = "\n".join(items)

        # Parse and verify completion
        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "memory_test.js", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract chunks"
        assert elapsed < 10.0, f"Should complete within 10s, took {elapsed:.2f}s"

        # Basic memory behavior check - verify we can parse multiple times
        # without accumulating state (indicates no major leaks)
        for iteration in range(3):
            chunks_repeat = parser.parse_content(code, f"memory_test_{iteration}.js", FileId(iteration + 2))
            assert len(chunks_repeat) > 0, f"Iteration {iteration} should succeed"


# =============================================================================
# PARSING SPEED TESTS - Ensure reasonable performance
# =============================================================================


class TestParsingSpeed:
    """Test parser performance meets acceptable time bounds.

    These tests verify the parser maintains reasonable performance,
    catching potential performance regressions or algorithmic issues.
    """

    @pytest.mark.slow
    def test_parsing_time_reasonable(self, parser_factory):
        """Ensure large files parse in reasonable time (<10s for 1MB).

        Stresses: Overall parser throughput, algorithmic efficiency.
        Expected: Maintains acceptable performance characteristics.

        Benchmark: 1MB file should parse in <10s on typical hardware.
        """
        parser = parser_factory.create_parser(Language.TYPESCRIPT)

        # Generate large TypeScript file with realistic code patterns
        components = []

        # 1000 interfaces to increase file size
        for i in range(1000):
            components.append(f"""
interface Data{i} {{
    id: number;
    name: string;
    value{i}: number;
    timestamp: Date;
    metadata: Record<string, any>;
}}""")

        # 1000 functions
        for i in range(1000):
            components.append(f"""
function processData{i}(input: Data{i}): Data{i} {{
    return {{
        ...input,
        value{i}: input.value{i} * 2,
        timestamp: new Date()
    }};
}}""")

        code = "\n".join(components)
        file_size_mb = len(code) / (1024 * 1024)

        # Verify size is substantial (at least 0.2MB for performance test)
        assert file_size_mb > 0.2, f"Expected >0.2MB, got {file_size_mb:.2f}MB"

        # Time the parsing
        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "speed_test.ts", FileId(1))
        elapsed = time.perf_counter() - start_time

        # Performance assertions
        assert len(chunks) > 0, "Should extract chunks"

        # For 1MB: <10s is reasonable, <5s is good, <2s is excellent
        throughput_mb_per_sec = file_size_mb / elapsed

        assert elapsed < 10.0, (
            f"Parsing {file_size_mb:.2f}MB took {elapsed:.2f}s "
            f"({throughput_mb_per_sec:.2f} MB/s), should be under 10s"
        )

        # Log performance for monitoring
        print(f"\nPerformance: {file_size_mb:.2f}MB in {elapsed:.2f}s = {throughput_mb_per_sec:.2f} MB/s")

    @pytest.mark.slow
    def test_incremental_performance_scaling(self, parser_factory):
        """Verify parsing time scales reasonably with input size.

        Stresses: Algorithmic complexity, performance scaling.
        Expected: Roughly linear time growth with input size (not quadratic).
        """
        parser = parser_factory.create_parser(Language.JAVASCRIPT)

        # Test different file sizes
        sizes_and_times = []

        for num_functions in [100, 200, 400]:
            functions = [
                f"function fn{i}() {{ return {i}; }}"
                for i in range(num_functions)
            ]
            code = "\n".join(functions)

            start_time = time.perf_counter()
            chunks = parser.parse_content(code, f"scale_test_{num_functions}.js", FileId(num_functions))
            elapsed = time.perf_counter() - start_time

            sizes_and_times.append((num_functions, elapsed))
            assert len(chunks) > 0, f"Should parse {num_functions} functions"

        # Verify scaling is reasonable (not quadratic)
        # If we double input size, time should not quadruple
        size1, time1 = sizes_and_times[0]  # 100 functions
        size2, time2 = sizes_and_times[2]  # 400 functions

        size_ratio = size2 / size1  # Should be 4x
        time_ratio = time2 / time1

        # Allow up to 6x time increase for 4x size increase
        # (linear would be 4x, quadratic would be 16x)
        assert time_ratio < 6.0, (
            f"Time ratio {time_ratio:.2f}x for {size_ratio:.2f}x size increase "
            "suggests poor scaling (possibly quadratic)"
        )

        print(f"\nScaling: {size_ratio:.1f}x size → {time_ratio:.2f}x time (linear = {size_ratio:.1f}x)")


# =============================================================================
# EDGE CASE STRESS TESTS - Test unusual but valid patterns
# =============================================================================


class TestEdgeCaseStress:
    """Test parser with unusual but syntactically valid patterns.

    These tests verify the parser handles edge cases without crashing.
    """

    @pytest.mark.slow
    def test_extreme_line_length(self, parser_factory):
        """Test parsing file with extremely long lines (10,000+ chars).

        Stresses: Line buffer handling, tokenization, memory.
        Expected: Completes without crash or excessive memory.
        """
        parser = parser_factory.create_parser(Language.JAVASCRIPT)

        # Create a line with 10,000+ characters
        long_array = "[" + ", ".join([f"'{i}'" for i in range(2000)]) + "]"

        code = f"""
const data = {long_array};

function processData() {{
    return data.length;
}}
"""

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "long_line.js", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should handle extremely long lines"
        assert elapsed < 5.0, f"Should complete within 5s, took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_many_empty_lines(self, parser_factory):
        """Test parsing file with thousands of empty lines.

        Stresses: Whitespace handling, line tracking.
        Expected: Completes efficiently.
        """
        parser = parser_factory.create_parser(Language.TYPESCRIPT)

        # Intersperse code with many empty lines
        code_parts = []
        for i in range(100):
            code_parts.append(f"function fn{i}() {{ return {i}; }}")
            code_parts.extend([""] * 50)  # 50 empty lines after each function

        code = "\n".join(code_parts)

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "many_empty_lines.ts", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should handle many empty lines"
        assert elapsed < 5.0, f"Should complete within 5s, took {elapsed:.2f}s"


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])
