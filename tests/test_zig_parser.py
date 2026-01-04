"""Tests for Zig language parser."""

import pytest

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory


@pytest.fixture
def parser_factory():
    """Create a parser factory instance."""
    return ParserFactory()


def test_zig_parser_available(parser_factory):
    """Test that Zig parser is available."""
    assert parser_factory.is_language_available(Language.ZIG)


def test_zig_parser_creates(parser_factory):
    """Test that Zig parser can be created."""
    parser = parser_factory.create_parser(Language.ZIG)
    assert parser is not None


def test_zig_function_parsing(parser_factory):
    """Test parsing Zig function declarations."""
    parser = parser_factory.create_parser(Language.ZIG)

    sample_code = """
    pub fn add(a: i32, b: i32) i32 {
        return a + b;
    }

    fn multiply(x: i32, y: i32) i32 {
        return x * y;
    }
    """

    chunks = parser.parse_content(sample_code, "test.zig", FileId(1))

    # Should extract chunks containing function code
    assert len(chunks) > 0, "Should extract at least one chunk"
    function_chunks = [c for c in chunks if "add" in c.code or "multiply" in c.code]
    assert len(function_chunks) > 0, "Should find chunks with function code"


def test_zig_struct_parsing(parser_factory):
    """Test parsing Zig struct declarations."""
    parser = parser_factory.create_parser(Language.ZIG)

    sample_code = """
    const Point = struct {
        x: f64,
        y: f64,

        pub fn init(x: f64, y: f64) Point {
            return Point{ .x = x, .y = y };
        }
    };

    pub const Rectangle = struct {
        width: f64,
        height: f64,
    };
    """

    chunks = parser.parse_content(sample_code, "test.zig", FileId(1))

    # Should extract chunks containing struct code
    assert len(chunks) > 0, "Should extract at least one chunk"
    struct_chunks = [c for c in chunks if "Point" in c.code or "Rectangle" in c.code]
    assert len(struct_chunks) > 0, "Should find chunks with struct code"


def test_zig_enum_parsing(parser_factory):
    """Test parsing Zig enum declarations."""
    parser = parser_factory.create_parser(Language.ZIG)

    sample_code = """
    pub const Color = enum {
        Red,
        Green,
        Blue,
    };

    const Direction = enum {
        North,
        South,
        East,
        West,
    };
    """

    chunks = parser.parse_content(sample_code, "test.zig", FileId(1))

    # Should extract chunks containing enum code
    assert len(chunks) > 0, "Should extract at least one chunk"
    enum_chunks = [c for c in chunks if "Color" in c.code or "Direction" in c.code]
    assert len(enum_chunks) > 0, "Should find chunks with enum code"


def test_zig_import_parsing(parser_factory):
    """Test parsing Zig import statements."""
    parser = parser_factory.create_parser(Language.ZIG)

    sample_code = """
    const std = @import("std");
    const builtin = @import("builtin");
    const my_module = @import("my_module.zig");
    """

    chunks = parser.parse_content(sample_code, "test.zig", FileId(1))

    # Should extract chunks containing import code
    assert len(chunks) > 0, "Should extract at least one chunk"
    import_chunks = [c for c in chunks if "@import" in c.code]
    assert len(import_chunks) > 0, "Should find chunks with import statements"


def test_zig_comment_parsing(parser_factory):
    """Test parsing Zig comments."""
    parser = parser_factory.create_parser(Language.ZIG)

    sample_code = """
    // This is a line comment
    /// This is a doc comment
    //! This is a module doc comment

    pub fn example() void {
        // Function implementation
    }
    """

    chunks = parser.parse_content(sample_code, "test.zig", FileId(1))

    # Should find comments
    comment_chunks = [c for c in chunks if "comment" in c.symbol.lower()]
    assert len(comment_chunks) >= 1, (
        f"Expected at least 1 comment, found {len(comment_chunks)}"
    )


def test_zig_variable_parsing(parser_factory):
    """Test parsing Zig variable declarations."""
    parser = parser_factory.create_parser(Language.ZIG)

    sample_code = """
    const PI = 3.14159;
    var counter: i32 = 0;
    pub const VERSION = "1.0.0";
    """

    chunks = parser.parse_content(sample_code, "test.zig", FileId(1))

    # Should extract chunks containing variable code
    assert len(chunks) > 0, "Should extract at least one chunk"
    var_chunks = [
        c
        for c in chunks
        if any(name in c.code for name in ["PI", "counter", "VERSION"])
    ]
    assert len(var_chunks) > 0, "Should find chunks with variable declarations"


def test_zig_complex_code(parser_factory):
    """Test parsing a complex Zig code sample."""
    parser = parser_factory.create_parser(Language.ZIG)

    sample_code = """
    const std = @import("std");

    /// A simple calculator
    pub const Calculator = struct {
        result: f64,

        pub fn init() Calculator {
            return Calculator{ .result = 0.0 };
        }

        pub fn add(self: *Calculator, value: f64) void {
            self.result += value;
        }

        pub fn subtract(self: *Calculator, value: f64) void {
            self.result -= value;
        }
    };

    pub fn main() !void {
        var calc = Calculator.init();
        calc.add(10.0);
        calc.subtract(5.0);

        const stdout = std.io.getStdOut().writer();
        try stdout.print("Result: {d}\\n", .{calc.result});
    }
    """

    chunks = parser.parse_content(sample_code, "calculator.zig", FileId(1))

    # Should extract multiple types of chunks
    assert len(chunks) > 0, "Should extract some chunks from complex code"

    # Check we got imports, structs, and functions in the code
    has_import = any("@import" in c.code for c in chunks)
    has_struct = any("Calculator" in c.code and "struct" in c.code for c in chunks)
    has_function = any("fn main" in c.code or "fn add" in c.code for c in chunks)

    assert has_import, "Should find import statement in code"
    assert has_struct, "Should find Calculator struct in code"
    assert has_function, "Should find function definitions in code"


def test_zig_function_metadata(parser_factory):
    """Test that function metadata is correctly extracted."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    pub fn publicFunc(a: i32, b: i32) i32 {
        // cAST merges adjacent siblings when combined size < max_chunk_size (1200 chars).
        // This function has enough content to prevent merging.
        var result: i32 = 0;
        var counter: i32 = 0;
        while (counter < 10) : (counter += 1) {
            result += a + b;
            const temp1 = result * 2;
            const temp2 = temp1 + counter;
            const temp3 = temp2 - a;
            const temp4 = temp3 * b;
            result = temp4;
        }
        const multiplier1: i32 = 5;
        const multiplier2: i32 = 10;
        const multiplier3: i32 = 15;
        const offset1: i32 = 100;
        const offset2: i32 = 200;
        const offset3: i32 = 300;
        const intermediate1 = result * multiplier1 + offset1;
        const intermediate2 = result * multiplier2 + offset2;
        const intermediate3 = result * multiplier3 + offset3;
        const sum_intermediates = intermediate1 + intermediate2 + intermediate3;
        const product = a * b;
        const quotient = sum_intermediates / product;
        const final_result = quotient - offset1;
        return final_result;
    }

    fn privateFunc() void {
        // Second function with substantial implementation
        var counter: i32 = 0;
        var sum: i32 = 0;
        var product: i32 = 1;
        while (counter < 20) : (counter += 1) {
            sum += counter;
            product *= 2;
            const doubled = sum * 2;
            const tripled = sum * 3;
            const quadrupled = sum * 4;
            sum = (doubled + tripled + quadrupled) / 9;
        }
        const x1: i32 = 42;
        const x2: i32 = 43;
        const x3: i32 = 44;
        const y1: i32 = 13;
        const y2: i32 = 14;
        const y3: i32 = 15;
        const z1 = x1 * y1;
        const z2 = x2 * y2;
        const z3 = x3 * y3;
        const total = z1 + z2 + z3 + sum + product;
        _ = total;
    }
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))
    func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]

    assert len(func_chunks) >= 2, "Should find at least 2 function chunks"

    # Find public function
    pub_func = next((c for c in func_chunks if "publicFunc" in c.code), None)
    assert pub_func is not None, "Should find publicFunc"
    assert pub_func.metadata.get("kind") == "function"
    assert pub_func.metadata.get("visibility") == "pub"

    # Find private function
    priv_func = next((c for c in func_chunks if "privateFunc" in c.code), None)
    assert priv_func is not None, "Should find privateFunc"
    assert priv_func.metadata.get("kind") == "function"
    assert priv_func.metadata.get("visibility") is None, (
        "Private function should not have visibility"
    )


def test_zig_struct_metadata(parser_factory):
    """Test that struct metadata is correctly extracted."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    pub const PublicStruct = struct {
        value: i32,
    };











    const PrivateStruct = struct {
        data: []const u8,
    };
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))
    struct_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]

    assert len(struct_chunks) >= 2, "Should find at least 2 struct chunks"

    # Find public struct
    pub_struct = next((c for c in struct_chunks if "PublicStruct" in c.code), None)
    assert pub_struct is not None, "Should find PublicStruct"
    assert pub_struct.metadata.get("kind") == "struct"
    assert pub_struct.metadata.get("visibility") == "pub"

    # Find private struct
    priv_struct = next((c for c in struct_chunks if "PrivateStruct" in c.code), None)
    assert priv_struct is not None, "Should find PrivateStruct"
    assert priv_struct.metadata.get("kind") == "struct"
    assert priv_struct.metadata.get("visibility") is None, (
        "Private struct should not have visibility"
    )


def test_zig_variable_metadata(parser_factory):
    """Test that variable metadata is correctly extracted."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    const PI = 3.14159;











    var counter: i32 = 0;











    pub const VERSION = "1.0.0";
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))

    # Find const variable
    pi_chunk = next((c for c in chunks if "PI" in c.code and "3.14159" in c.code), None)
    assert pi_chunk is not None, "Should find PI constant"
    assert pi_chunk.metadata.get("kind") == "variable"
    assert pi_chunk.metadata.get("mutability") == "const"
    assert pi_chunk.metadata.get("visibility") is None

    # Find var variable
    counter_chunk = next((c for c in chunks if "counter" in c.code), None)
    assert counter_chunk is not None, "Should find counter variable"
    assert counter_chunk.metadata.get("kind") == "variable"
    assert counter_chunk.metadata.get("mutability") == "var"

    # Find pub const variable
    version_chunk = next((c for c in chunks if "VERSION" in c.code), None)
    assert version_chunk is not None, "Should find VERSION constant"
    assert version_chunk.metadata.get("kind") == "variable"
    assert version_chunk.metadata.get("mutability") == "const"
    assert version_chunk.metadata.get("visibility") == "pub"


def test_zig_comment_metadata(parser_factory):
    """Test that comment metadata is correctly extracted."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    // This is a line comment
    /// This is a doc comment
    //! This is a module doc comment

    pub fn example() void {}
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))
    comment_chunks = [c for c in chunks if c.chunk_type == ChunkType.COMMENT]

    assert len(comment_chunks) >= 1, "Should find at least one comment chunk"

    # Check for doc comment
    doc_comment = next(
        (c for c in comment_chunks if c.metadata.get("comment_type") == "doc"), None
    )
    if doc_comment:
        assert doc_comment.metadata.get("is_doc_comment") is True

    # Check for module doc comment
    module_doc = next(
        (c for c in comment_chunks if c.metadata.get("comment_type") == "module_doc"),
        None,
    )
    if module_doc:
        assert module_doc.metadata.get("is_doc_comment") is True


def test_zig_import_metadata(parser_factory):
    """Test that import metadata is correctly extracted."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    const std = @import("std");
    const builtin = @import("builtin");
    const my_module = @import("my_module.zig");
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))

    # Find std import
    std_import = next((c for c in chunks if '@import("std")' in c.code), None)
    assert std_import is not None, "Should find std import"
    if std_import.metadata.get("import_path"):
        assert std_import.metadata.get("import_path") == "std"

    # Find builtin import
    builtin_import = next((c for c in chunks if '@import("builtin")' in c.code), None)
    assert builtin_import is not None, "Should find builtin import"
    if builtin_import.metadata.get("builtin"):
        assert builtin_import.metadata.get("builtin") == "builtin"


def test_zig_chunk_types(parser_factory):
    """Test that chunk types are correctly assigned."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    // Comment
    const std = @import("std");











    pub fn myFunction() void {}











    pub const MyStruct = struct {
        value: i32,
    };
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))

    # Verify we have different chunk types
    chunk_types = {c.chunk_type for c in chunks}

    # Should have at least some of these chunk types
    has_function = any(c.chunk_type == ChunkType.FUNCTION for c in chunks)
    has_class = any(c.chunk_type == ChunkType.CLASS for c in chunks)

    assert has_function, "Should have at least one FUNCTION chunk"
    assert has_class, "Should have at least one CLASS chunk (struct)"


def test_zig_union_parsing(parser_factory):
    """Test parsing Zig union declarations."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    pub const Value = union(enum) {
        int: i32,
        float: f64,
        bool: bool,
    };

    const Result = union {
        ok: []const u8,
        err: []const u8,
    };
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))

    # Should extract union chunks
    union_chunks = [c for c in chunks if "Value" in c.code or "Result" in c.code]
    assert len(union_chunks) > 0, "Should find union declarations"

    # Verify public union metadata
    value_union = next(
        (c for c in chunks if "Value" in c.code and "union" in c.code), None
    )
    if value_union and value_union.metadata.get("kind") == "union":
        assert value_union.metadata.get("visibility") == "pub"


def test_zig_opaque_parsing(parser_factory):
    """Test parsing Zig opaque type declarations."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    pub const Handle = opaque {};

    const InternalHandle = opaque {};
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))

    # Should extract opaque chunks
    opaque_chunks = [c for c in chunks if "Handle" in c.code and "opaque" in c.code]
    assert len(opaque_chunks) > 0, "Should find opaque type declarations"

    # Verify public opaque metadata
    handle_opaque = next((c for c in chunks if "pub const Handle" in c.code), None)
    if handle_opaque and handle_opaque.metadata.get("kind") == "opaque":
        assert handle_opaque.metadata.get("visibility") == "pub"


def test_zig_error_handling(parser_factory):
    """Test parsing Zig error handling constructs."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    fn riskyOperation() !i32 {
        return error.SomethingWentWrong;
    }











    fn caller() !void {
        const result = try riskyOperation();
        defer cleanup();
        errdefer handleError();
    }
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))

    # Should extract functions with error unions
    func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
    assert len(func_chunks) >= 2, "Should find at least 2 function chunks"

    # Verify try/defer/errdefer appear in code
    caller_chunk = next((c for c in chunks if "caller" in c.code), None)
    assert caller_chunk is not None, "Should find caller function"
    assert "try" in caller_chunk.code or "defer" in caller_chunk.code, (
        "Should contain error handling keywords"
    )


def test_zig_comptime_features(parser_factory):
    """Test parsing Zig comptime features."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    fn generic(comptime T: type, value: T) T {
        return value;
    }

    inline fn inlineFunc() void {}

    test "basic test" {
        const x = 42;
    }
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))

    # Should extract comptime generic function
    assert any("generic" in c.code for c in chunks), "Should find generic function"

    # Should extract inline function
    assert any("inlineFunc" in c.code for c in chunks), "Should find inline function"

    # Should extract test blocks
    test_chunks = [
        c for c in chunks if "test" in c.code.lower() and "basic test" in c.code
    ]
    assert len(test_chunks) > 0, "Should find test block"


def test_zig_function_symbol_names(parser_factory):
    """Test that Zig parser generates correct function symbol names."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    pub fn add(a: i32, b: i32) i32 {
        // cAST merges small adjacent siblings. Expanding functions prevents merging.
        var result: i32 = 0;
        var temp_a: i32 = a;
        var temp_b: i32 = b;
        var counter: i32 = 0;
        while (counter < 5) : (counter += 1) {
            result += temp_a;
            temp_a = temp_a * 2;
        }
        result += temp_b;
        const multiplier: i32 = 3;
        const offset: i32 = 10;
        const intermediate = result * multiplier + offset;
        return intermediate;
    }

    fn subtract(x: i32, y: i32) i32 {
        // Second function with substantial implementation
        var result: i32 = x;
        var subtrahend: i32 = y;
        var iterations: i32 = 0;
        while (iterations < 5) : (iterations += 1) {
            result = result - 1;
            subtrahend = subtrahend + 1;
        }
        const adjustment: i32 = 5;
        const factor: i32 = 2;
        const temp1 = result - subtrahend;
        const temp2 = temp1 * factor;
        const final_result = temp2 - adjustment;
        return final_result;
    }

    inline fn multiply(x: i32, y: i32) i32 {
        // Third function with substantial implementation
        var product: i32 = 0;
        var multiplicand: i32 = x;
        var loop_count: i32 = 0;
        while (loop_count < y) : (loop_count += 1) {
            product += multiplicand;
            const temp = product * 2;
            product = temp / 2;
        }
        const bonus: i32 = 7;
        const divisor: i32 = 3;
        const adjusted = (product + bonus) / divisor;
        return adjusted * divisor;
    }
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))
    symbols = {c.symbol for c in chunks}

    assert "add" in symbols, f"Expected 'add' in symbols, got: {sorted(symbols)}"
    assert "subtract" in symbols, (
        f"Expected 'subtract' in symbols, got: {sorted(symbols)}"
    )
    assert "multiply" in symbols, (
        f"Expected 'multiply' in symbols, got: {sorted(symbols)}"
    )


def test_zig_struct_symbol_names(parser_factory):
    """Test that Zig parser generates correct struct/enum symbol names."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    const Point = struct { x: f64, y: f64 };











    pub const Color = enum { Red, Green, Blue };











    pub const Result = union(enum) { ok: i32, err: []const u8 };
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))
    symbols = {c.symbol for c in chunks}

    assert "Point" in symbols, f"Expected 'Point' in symbols, got: {sorted(symbols)}"
    assert "Color" in symbols, f"Expected 'Color' in symbols, got: {sorted(symbols)}"
    assert "Result" in symbols, f"Expected 'Result' in symbols, got: {sorted(symbols)}"


def test_zig_variable_symbol_names(parser_factory):
    """Test that Zig parser generates correct variable symbol names."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    const PI = 3.14159;











    var globalCounter: i32 = 0;











    pub const VERSION = "1.0.0";
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))
    symbols = {c.symbol for c in chunks}

    assert "PI" in symbols, f"Expected 'PI' in symbols, got: {sorted(symbols)}"
    assert "globalCounter" in symbols, (
        f"Expected 'globalCounter' in symbols, got: {sorted(symbols)}"
    )
    assert "VERSION" in symbols, (
        f"Expected 'VERSION' in symbols, got: {sorted(symbols)}"
    )


def test_zig_import_symbol_names(parser_factory):
    """Test that Zig parser generates correct import symbol names."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    const std = @import("std");











    const builtin = @import("builtin");











    const my_module = @import("my_module.zig");
    """

    chunks = parser.parse_content(code, "test.zig", FileId(1))
    symbols = {c.symbol for c in chunks}

    # Import symbols might be named differently, so check they exist
    import_chunks = [c for c in chunks if "@import" in c.code]
    assert len(import_chunks) >= 3, (
        f"Expected at least 3 import chunks, got {len(import_chunks)}"
    )


def test_zig_realistic_module(parser_factory):
    """Test parsing a realistic Zig module with multiple features."""
    parser = parser_factory.create_parser(Language.ZIG)

    code = """
    const std = @import("std");

    /// Custom error set for file operations
    pub const FileError = error{
        NotFound,
        PermissionDenied,
        AlreadyExists,
        Timeout,
        InvalidFormat,
        BufferOverflow,
    };

    /// Result type for operations that can fail
    pub const Result = union(enum) {
        success: []const u8,
        failure: FileError,

        pub fn isSuccess(self: Result) bool {
            return switch (self) {
                .success => true,
                .failure => false,
            };
        }

        pub fn getError(self: Result) ?FileError {
            return switch (self) {
                .success => null,
                .failure => |err| err,
            };
        }

        pub fn getValue(self: Result) ?[]const u8 {
            return switch (self) {
                .success => |val| val,
                .failure => null,
            };
        }
    };

    /// Configuration struct
    pub const Config = struct {
        max_retries: u32,
        timeout_ms: u64,
        buffer_size: usize,
        enable_cache: bool,
        log_level: u8,

        pub fn init() Config {
            return Config{
                .max_retries = 3,
                .timeout_ms = 5000,
                .buffer_size = 4096,
                .enable_cache = true,
                .log_level = 2,
            };
        }

        pub fn validate(self: *const Config) bool {
            return self.max_retries > 0 and self.timeout_ms > 0 and self.buffer_size > 0;
        }

        pub fn withRetries(self: Config, retries: u32) Config {
            var new_config = self;
            new_config.max_retries = retries;
            return new_config;
        }
    };

    var global_config: Config = undefined;

    /// Attempts to read a file with error handling
    pub fn readFile(path: []const u8) ![]const u8 {
        // cAST merges small adjacent siblings. Making functions large prevents merging.
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const size = try file.getEndPos();
        const buffer = try std.heap.page_allocator.alloc(u8, size);
        errdefer std.heap.page_allocator.free(buffer);

        var bytes_read: usize = 0;
        var total_read: usize = 0;
        var retry_count: u32 = 0;

        while (total_read < size and retry_count < global_config.max_retries) {
            bytes_read = try file.read(buffer[total_read..]);
            total_read += bytes_read;
            if (bytes_read == 0) {
                retry_count += 1;
            }
        }

        if (total_read < size) {
            return error.Incomplete;
        }

        const checksum: u32 = 0;
        var i: usize = 0;
        while (i < total_read) : (i += 1) {
            _ = checksum + buffer[i];
        }

        return buffer;
    }

    /// Generic function using comptime
    fn max(comptime T: type, a: T, b: T) T {
        // Second function with substantial implementation
        var result: T = a;
        const comparison_a = a > b;
        const comparison_b = b > a;

        if (comparison_a) {
            result = a;
            const temp_a = a * 1;
            _ = temp_a;
        } else if (comparison_b) {
            result = b;
            const temp_b = b * 1;
            _ = temp_b;
        } else {
            result = a;
            const temp_c = a + 0;
            _ = temp_c;
        }

        const value1: T = result;
        const value2: T = if (a > b) a else b;
        const value3: T = if (value1 > value2) value1 else value2;

        return value3;
    }

    test "config initialization" {
        // Test block with substantial implementation
        const config = Config.init();
        try std.testing.expect(config.max_retries == 3);
        try std.testing.expect(config.timeout_ms == 5000);
        try std.testing.expect(config.buffer_size == 4096);
        try std.testing.expect(config.enable_cache == true);
        try std.testing.expect(config.log_level == 2);

        const is_valid = config.validate();
        try std.testing.expect(is_valid == true);

        const modified = config.withRetries(5);
        try std.testing.expect(modified.max_retries == 5);
    }

    test "result type" {
        // Second test block with substantial implementation
        const success_result = Result{ .success = "data" };
        try std.testing.expect(success_result.isSuccess());

        const value = success_result.getValue();
        try std.testing.expect(value != null);

        const error_value = success_result.getError();
        try std.testing.expect(error_value == null);

        const failure_result = Result{ .failure = FileError.NotFound };
        try std.testing.expect(!failure_result.isSuccess());

        const fail_value = failure_result.getValue();
        try std.testing.expect(fail_value == null);
    }
    """

    chunks = parser.parse_content(code, "realistic.zig", FileId(1))

    # Verify we found multiple types of constructs
    assert len(chunks) > 0, "Should extract chunks from realistic code"

    # Check for imports
    has_import = any("@import" in c.code for c in chunks)
    assert has_import, "Should find import statements"

    # Check for union type
    has_union = any("Result" in c.code and "union" in c.code for c in chunks)
    assert has_union, "Should find Result union type"

    # Check for struct
    has_struct = any("Config" in c.code and "struct" in c.code for c in chunks)
    assert has_struct, "Should find Config struct"

    # Check for functions with error handling
    has_error_func = any("readFile" in c.code and "!" in c.code for c in chunks)
    assert has_error_func, "Should find function with error union return"

    # Check for comptime function
    has_comptime = any("max" in c.code and "comptime" in c.code for c in chunks)
    assert has_comptime, "Should find generic comptime function"

    # Check for test blocks
    has_test = any("test" in c.code.lower() for c in chunks)
    assert has_test, "Should find test blocks"

    # Verify metadata for some chunks
    func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
    assert len(func_chunks) >= 3, (
        f"Should find at least 3 functions, found {len(func_chunks)}"
    )

    # Verify public function has visibility metadata
    read_file_func = next((c for c in func_chunks if "readFile" in c.code), None)
    if read_file_func:
        assert read_file_func.metadata.get("visibility") == "pub", (
            "readFile should be public"
        )


def test_zig_file_extension_detection():
    """Test that .zig files are detected as Zig language."""
    from pathlib import Path

    assert Language.from_file_extension(Path("main.zig")) == Language.ZIG
    assert Language.from_file_extension("test.zig") == Language.ZIG
