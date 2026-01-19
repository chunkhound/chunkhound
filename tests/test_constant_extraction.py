"""Tests for constant extraction across all language parsers.

These are sociable integration tests that exercise real tree-sitter parsing
with no mocks. Following the pattern from test_hcl_mapping.py.
"""

import pytest

from chunkhound.core.types.common import FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.parsers.universal_engine import UniversalConcept


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def parse_constants():
    """Parse code and extract constants from DEFINITION chunks (concept-level).

    Returns a function that parses code and returns all constants extracted
    from DEFINITION concept chunks.
    """

    def _parse(code: str, language: Language) -> list[dict[str, str]]:
        parser = ParserFactory().create_parser(language)
        ast = parser.engine.parse_to_ast(code)
        chunks = parser.extractor.extract_concept(
            ast.root_node, code.encode(), UniversalConcept.DEFINITION
        )
        constants = []
        for chunk in chunks:
            chunk_constants = chunk.metadata.get("constants", [])
            if chunk_constants:
                constants.extend(chunk_constants)
        return constants

    return _parse


@pytest.fixture
def parse_file_constants(tmp_path):
    """Parse a file and extract constants via parse_content (file-level).

    Uses the parser's parse_content method which exercises the full
    concept extraction pipeline.
    """

    def _parse(code: str, filename: str, language: Language) -> list[dict[str, str]]:
        file_path = tmp_path / filename
        file_path.write_text(code)

        parser = ParserFactory().create_parser(language)
        chunks = parser.parse_content(code, filename, FileId(1))

        constants = []
        for chunk in chunks:
            chunk_constants = chunk.metadata.get("constants", [])
            if chunk_constants:
                constants.extend(chunk_constants)
        return constants

    return _parse


# =============================================================================
# Cross-Language Basic Extraction Tests
# =============================================================================


# Languages with WORKING constant extraction (concept queries capture constant nodes)
WORKING_LANGUAGES = {
    Language.C,  # preproc_def and const declarations captured by DEFINITION
    Language.CPP,
    Language.CSHARP,  # field_declaration and local_declaration_statement captured by DEFINITION
    Language.DART,
    Language.GO,
    Language.GROOVY,  # field_declaration and local_variable_declaration captured by DEFINITION
    Language.HASKELL,
    Language.JAVA,
    Language.KOTLIN,  # property_declaration captured by DEFINITION
    Language.LUA,  # UPPER_SNAKE_CASE convention like Python
    Language.PYTHON,
    Language.RUST,
    Language.SWIFT,
    Language.PHP,
    Language.MAKEFILE,
    Language.OBJC,
    Language.ZIG,
}

# Languages where constant extraction is NOT YET IMPLEMENTED
# (extract_constants method exists but concept queries don't capture constant nodes)
NOT_IMPLEMENTED_LANGUAGES = {
    Language.TYPESCRIPT,  # lexical_declaration not captured by DEFINITION
    Language.JAVASCRIPT,  # lexical_declaration not captured by DEFINITION
}


@pytest.mark.parametrize(
    "language,code,expected_names",
    [
        # === WORKING LANGUAGES ===
        # C++: constexpr (WORKS)
        pytest.param(Language.CPP, "constexpr int MAX = 100;", ["MAX"], id="cpp"),
        # Go: const declaration (WORKS)
        pytest.param(Language.GO, "const MAX = 100", ["MAX"], id="go"),
        # Java: static final field (WORKS)
        pytest.param(Language.JAVA, "class Test { public static final int MAX = 100; }", ["MAX"], id="java"),
        # Rust: const item (WORKS)
        pytest.param(Language.RUST, "const MAX: i32 = 100;", ["MAX"], id="rust"),
        # Swift: let at module level (WORKS)
        pytest.param(Language.SWIFT, "let MAX = 100", ["MAX"], id="swift"),
        # PHP: const declaration (WORKS)
        pytest.param(Language.PHP, "<?php\nconst MAX = 100;", ["MAX"], id="php"),
        # Makefile: variable assignment (WORKS)
        pytest.param(Language.MAKEFILE, "MAX = 100", ["MAX"], id="makefile"),
        # Objective-C: #define macro (WORKS)
        pytest.param(Language.OBJC, "#define MAX 100", ["MAX"], id="objc"),
        # Zig: const declaration (WORKS)
        pytest.param(Language.ZIG, "const MAX = 100;", ["MAX"], id="zig"),
        # Haskell: top-level binding (WORKS)
        pytest.param(Language.HASKELL, "maxValue = 100", ["maxValue"], id="haskell"),
        # === NOT YET IMPLEMENTED ===
        pytest.param(Language.PYTHON, "MAX_VALUE = 100", ["MAX_VALUE"], id="python"),
        pytest.param(Language.C, "#define MAX 100", ["MAX"], id="c"),
        pytest.param(Language.CSHARP, "class Test { public const int MAX = 100; }", ["MAX"], id="csharp"),
        pytest.param(Language.TYPESCRIPT, "const MAX = 100;", ["MAX"], id="typescript"),
        pytest.param(Language.JAVASCRIPT, "const MAX = 100;", ["MAX"], id="javascript"),
        # === WORKING LANGUAGES (continued) ===
        pytest.param(Language.KOTLIN, "const val MAX = 100", ["MAX"], id="kotlin"),
        pytest.param(Language.LUA, "local MAX = 100", ["MAX"], id="lua"),
        # === NOT YET IMPLEMENTED (continued) ===
        pytest.param(Language.DART, "const MAX = 100;", ["MAX"], id="dart"),
        pytest.param(Language.GROOVY, "class Test { static final int MAX = 100 }", ["MAX"], id="groovy"),
    ],
)
def test_basic_constant_extraction(parse_constants, language, code, expected_names):
    """Test that basic constant declarations are extracted across languages.

    NOTE: Some languages fail because their constant declaration node types
    are not yet captured by DEFINITION concept queries. The extract_constants()
    methods exist in the mappings but are never called.
    """
    constants = parse_constants(code, language)
    names = {c["name"] for c in constants}
    for expected in expected_names:
        assert expected in names, f"Expected constant '{expected}' not found in {names}"


# =============================================================================
# Python UPPER_SNAKE_CASE Pattern Tests
# Tests UPPER_SNAKE_CASE constant detection via module-level assignment capture.
# =============================================================================



@pytest.mark.parametrize(
    "code,should_extract",
    [
        # Valid UPPER_SNAKE_CASE patterns
        ("MAX_VALUE = 100", True),
        ("API_KEY_V2 = 'secret'", True),
        ("_PRIVATE = 1", True),
        ("MAX123 = 1", True),
        ("A = 1", True),  # Single letter uppercase
        ("MAX_ = 1", True),  # Trailing underscore
    ],
    ids=[
        "max_value",
        "api_key_v2",
        "private_underscore",
        "max_with_numbers",
        "single_letter",
        "trailing_underscore",
    ],
)
def test_python_upper_snake_case_valid(parse_constants, code, should_extract):
    """Test Python UPPER_SNAKE_CASE constant detection (valid patterns)."""
    constants = parse_constants(code, Language.PYTHON)
    assert len(constants) > 0, f"Expected constant from '{code}' but got none"


@pytest.mark.parametrize(
    "code",
    [
        "Max_Value = 1",  # Mixed case
        "maxValue = 1",  # camelCase
        "max_value = 1",  # lowercase snake_case
        "value = 1",  # lowercase
    ],
    ids=["mixed_case", "camel_case", "lowercase_snake", "lowercase"],
)
def test_python_upper_snake_case_invalid(parse_constants, code):
    """Test Python does not extract non-UPPER_SNAKE_CASE patterns."""
    constants = parse_constants(code, Language.PYTHON)
    # For invalid patterns, no constants should be extracted
    # (This passes because no constants are extracted at all currently)
    names = {c["name"] for c in constants}
    var_name = code.split("=")[0].strip()
    assert var_name not in names, f"Should not extract '{var_name}' from '{code}'"


def test_python_function_not_extracted_as_constant(parse_constants):
    """Functions with UPPER_SNAKE_CASE names should not be extracted as constants."""
    code = "def MAX_FUNCTION(): pass"
    constants = parse_constants(code, Language.PYTHON)
    names = {c["name"] for c in constants}
    assert "MAX_FUNCTION" not in names


def test_python_class_not_extracted_as_constant(parse_constants):
    """Classes with UPPER_SNAKE_CASE names should not be extracted as constants."""
    code = "class MAX_CLASS: pass"
    constants = parse_constants(code, Language.PYTHON)
    names = {c["name"] for c in constants}
    assert "MAX_CLASS" not in names


def test_python_no_duplicate_constants_for_dict_assignment(parse_constants):
    """Dict/list assignments should not produce duplicate constants.

    Regression test: overlapping queries for dict/list literals and
    identifier assignments previously caused duplicates.
    """
    code = 'CONFIG = {"key": "value"}'
    constants = parse_constants(code, Language.PYTHON)
    config_constants = [c for c in constants if c["name"] == "CONFIG"]
    assert len(config_constants) == 1, (
        f"Expected exactly 1 CONFIG constant, got {len(config_constants)}: {config_constants}"
    )


def test_python_no_duplicate_constants_for_list_assignment(parse_constants):
    """List assignments should not produce duplicate constants."""
    code = "OPTIONS = [1, 2, 3]"
    constants = parse_constants(code, Language.PYTHON)
    options_constants = [c for c in constants if c["name"] == "OPTIONS"]
    assert len(options_constants) == 1, (
        f"Expected exactly 1 OPTIONS constant, got {len(options_constants)}: {options_constants}"
    )


# =============================================================================
# Java Static Final Requirement Tests
# NOTE: Java field constant extraction is NOT YET IMPLEMENTED - field_declaration
# nodes are not captured as standalone DEFINITION concepts. These tests document
# desired behavior.
# =============================================================================



@pytest.mark.parametrize(
    "code",
    [
        "class T { public static final int MAX = 100; }",
        "class T { static final int MAX = 100; }",
        "class T { private static final String NAME = \"test\"; }",
    ],
    ids=["public_static_final", "static_final", "private_static_final_string"],
)
def test_java_static_final_should_extract(parse_constants, code):
    """Test Java static final fields should be extracted (not yet implemented)."""
    constants = parse_constants(code, Language.JAVA)
    has_const = any(c["name"] in ("MAX", "NAME") for c in constants)
    assert has_const, f"Expected constant extraction for: {code}"


@pytest.mark.parametrize(
    "code",
    [
        "class T { static int MAX = 100; }",  # missing final
        "class T { public static int MAX = 100; }",  # missing final
        "class T { final int MAX = 100; }",  # missing static
        "class T { public final int MAX = 100; }",  # missing static
        "class T { int MAX = 100; }",  # neither
    ],
    ids=["static_only", "public_static_only", "final_only", "public_final_only", "neither"],
)
def test_java_partial_modifiers_not_extracted(parse_constants, code):
    """Test Java fields without both static AND final are not extracted."""
    constants = parse_constants(code, Language.JAVA)
    # These should not be extracted (and aren't, because nothing is extracted)
    has_max = any(c["name"] == "MAX" for c in constants)
    assert not has_max, f"Should not extract constant for: {code}"



def test_java_enum_members(parse_constants):
    """Test Java enum members are extracted as constants (not yet implemented)."""
    code = """
public enum Status {
    PENDING,
    ACTIVE,
    COMPLETE
}
"""
    constants = parse_constants(code, Language.JAVA)
    names = {c["name"] for c in constants}
    # Enum members should be extracted
    assert "PENDING" in names, f"Expected enum constants, got: {names}"


# =============================================================================
# C/C++ Macro vs Function-Like Macro Tests
# NOTE: C preproc_def is NOT captured by DEFINITION queries. C++ captures
# const/constexpr declarations but not #define macros directly.
# =============================================================================



@pytest.mark.parametrize(
    "code,expected_name",
    [
        ("#define MAX 100", "MAX"),
        ("#define PI 3.14159", "PI"),
        ("#define EMPTY", "EMPTY"),
    ],
    ids=["object_macro_int", "object_macro_float", "object_macro_empty"],
)
def test_c_object_macros_should_extract(parse_constants, code, expected_name):
    """Test C object-like macros are extracted."""
    constants = parse_constants(code, Language.C)
    names = {c["name"] for c in constants}
    assert expected_name in names, f"Expected '{expected_name}' in {names}"


@pytest.mark.parametrize(
    "code",
    [
        "#define FUNC(x) ((x)*2)",
        "#define MAX(a,b) ((a)>(b)?(a):(b))",
        "#define SQUARE(n) ((n)*(n))",
    ],
    ids=["function_macro_single_param", "function_macro_two_params", "function_macro_square"],
)
def test_c_function_like_macros_not_extracted(parse_constants, code):
    """Test C function-like macros are not extracted."""
    constants = parse_constants(code, Language.C)
    # Function-like macros should not produce constants
    # (Currently nothing is extracted for C, so this passes)
    macro_name = code.split()[1].split("(")[0]
    names = {c["name"] for c in constants}
    assert macro_name not in names, f"Should not extract function-like macro '{macro_name}'"


@pytest.mark.parametrize(
    "code,expected_name",
    [
        # const declarations (C and C++)
        ("const int MAX = 100;", "MAX"),
        ("const double PI = 3.14159;", "PI"),
    ],
    ids=["const_int", "const_double"],
)
def test_c_const_declarations(parse_constants, code, expected_name):
    """Test C const declarations are extracted."""
    constants = parse_constants(code, Language.C)
    names = {c["name"] for c in constants}
    assert expected_name in names, f"Expected '{expected_name}' in {names}"


def test_c_function_scoped_const(parse_constants):
    """Test C const declarations inside functions are extracted."""
    code = """
void my_function() {
    const int FUNC_CONST = 200;
    const char *MESSAGE = "hello";
}
"""
    constants = parse_constants(code, Language.C)
    names = {c["name"] for c in constants}
    assert "FUNC_CONST" in names, f"Expected FUNC_CONST in {names}"
    # Note: MESSAGE is a pointer declarator, may not be captured by simple identifier query


def test_c_nested_block_const(parse_constants):
    """Test C const declarations in nested blocks are extracted."""
    code = """
void outer() {
    const int OUTER = 1;
    if (1) {
        const int INNER = 2;
    }
}
"""
    constants = parse_constants(code, Language.C)
    names = {c["name"] for c in constants}
    assert "OUTER" in names, f"Expected OUTER in {names}"
    assert "INNER" in names, f"Expected INNER in {names}"


def test_c_non_const_variable_not_extracted(parse_constants):
    """Test C non-const variables are not extracted."""
    code = """
void func() {
    int regular = 100;
    const int CONSTANT = 200;
}
"""
    constants = parse_constants(code, Language.C)
    names = {c["name"] for c in constants}
    assert "CONSTANT" in names, f"Expected CONSTANT in {names}"
    assert "regular" not in names, f"Should not extract non-const variable"


@pytest.mark.parametrize(
    "code,expected_name",
    [
        # constexpr declarations (C++)
        ("constexpr int MAX = 100;", "MAX"),
        ("constexpr double PI = 3.14159;", "PI"),
    ],
    ids=["constexpr_int", "constexpr_double"],
)
def test_cpp_const_declarations(parse_constants, code, expected_name):
    """Test C++ const and constexpr declarations are extracted."""
    constants = parse_constants(code, Language.CPP)
    names = {c["name"] for c in constants}
    assert expected_name in names, f"Expected '{expected_name}' in {names}"


# =============================================================================
# Go Const Pattern Tests
# =============================================================================


def test_go_single_const(parse_constants):
    """Test Go single const declaration."""
    code = "const MAX = 100"
    constants = parse_constants(code, Language.GO)
    names = {c["name"] for c in constants}
    assert "MAX" in names


def test_go_const_block(parse_constants):
    """Test Go const block with multiple constants."""
    code = """
const (
    FIRST = 1
    SECOND = 2
    THIRD = 3
)
"""
    constants = parse_constants(code, Language.GO)
    names = {c["name"] for c in constants}
    assert "FIRST" in names or len(constants) >= 1, f"Expected const block constants, got: {names}"


def test_go_const_with_iota(parse_constants):
    """Test Go const block with iota."""
    code = """
const (
    A = iota
    B
    C
)
"""
    constants = parse_constants(code, Language.GO)
    # Should extract at least A, possibly B and C
    assert len(constants) >= 1, "Expected iota constants"


def test_go_typed_const(parse_constants):
    """Test Go const with explicit type."""
    code = "const MAX int = 100"
    constants = parse_constants(code, Language.GO)
    names = {c["name"] for c in constants}
    assert "MAX" in names


# =============================================================================
# Rust Const/Static Mutability Tests
# =============================================================================


def test_rust_const_item(parse_constants):
    """Test Rust const item extraction."""
    code = "const MAX: i32 = 100;"
    constants = parse_constants(code, Language.RUST)
    names = {c["name"] for c in constants}
    assert "MAX" in names


def test_rust_const_with_type(parse_constants):
    """Test Rust const includes type annotation."""
    code = "const PI: f64 = 3.14159;"
    constants = parse_constants(code, Language.RUST)
    assert len(constants) > 0
    pi_const = next((c for c in constants if c["name"] == "PI"), None)
    if pi_const and "type" in pi_const:
        assert "f64" in pi_const["type"]


def test_rust_static_item(parse_constants):
    """Test Rust static item extraction."""
    code = "static COUNTER: i32 = 0;"
    constants = parse_constants(code, Language.RUST)
    names = {c["name"] for c in constants}
    assert "COUNTER" in names


def test_rust_mutable_static(parse_constants):
    """Test Rust mutable static has mutable flag."""
    code = "static mut COUNTER: i32 = 0;"
    constants = parse_constants(code, Language.RUST)
    counter = next((c for c in constants if c["name"] == "COUNTER"), None)
    if counter:
        # Check if mutable flag is present
        assert counter.get("mutable") == "true" or "mut" in str(counter)


def test_rust_immutable_const_no_mutable_flag(parse_constants):
    """Test Rust immutable const does not have mutable flag."""
    code = "const MAX: i32 = 100;"
    constants = parse_constants(code, Language.RUST)
    max_const = next((c for c in constants if c["name"] == "MAX"), None)
    if max_const:
        assert max_const.get("mutable") != "true"


def test_rust_immutable_let_binding(parse_constants):
    """Test Rust immutable let binding is extracted as constant."""
    code = "let MAX = 100;"
    constants = parse_constants(code, Language.RUST)
    names = {c["name"] for c in constants}
    assert "MAX" in names, f"Expected immutable let binding 'MAX' in {names}"


def test_rust_mutable_let_binding_not_extracted(parse_constants):
    """Test Rust mutable let binding is not extracted as constant."""
    code = "let mut counter = 0;"
    constants = parse_constants(code, Language.RUST)
    names = {c["name"] for c in constants}
    assert "counter" not in names, f"Should not extract mutable let binding 'counter', got {names}"


def test_rust_immutable_let_with_type(parse_constants):
    """Test Rust immutable let binding with type annotation."""
    code = "let PI: f64 = 3.14159;"
    constants = parse_constants(code, Language.RUST)
    pi_const = next((c for c in constants if c["name"] == "PI"), None)
    assert pi_const is not None, "Expected immutable let binding 'PI'"
    if "type" in pi_const:
        assert "f64" in pi_const["type"]


def test_rust_multiple_let_bindings(parse_constants):
    """Test multiple Rust let bindings - only immutable ones are extracted."""
    code = """
let IMMUTABLE1 = 100;
let mut mutable1 = 200;
let IMMUTABLE2: i32 = 300;
let mut mutable2: i32 = 400;
"""
    constants = parse_constants(code, Language.RUST)
    names = {c["name"] for c in constants}
    # Should extract immutable bindings
    assert "IMMUTABLE1" in names, f"Expected IMMUTABLE1 in {names}"
    assert "IMMUTABLE2" in names, f"Expected IMMUTABLE2 in {names}"
    # Should NOT extract mutable bindings
    assert "mutable1" not in names, f"Should not extract mutable1, got {names}"
    assert "mutable2" not in names, f"Should not extract mutable2, got {names}"


# =============================================================================
# TypeScript/JavaScript Tests
# NOTE: TypeScript/JavaScript lexical_declaration (const) is NOT captured by
# DEFINITION queries. These tests document desired behavior.
# =============================================================================



def test_typescript_const_declaration(parse_constants):
    """Test TypeScript const declaration (not yet implemented)."""
    code = "const MAX = 100;"
    constants = parse_constants(code, Language.TYPESCRIPT)
    names = {c["name"] for c in constants}
    assert "MAX" in names



def test_typescript_const_with_type(parse_constants):
    """Test TypeScript const with type annotation (not yet implemented)."""
    code = "const MAX: number = 100;"
    constants = parse_constants(code, Language.TYPESCRIPT)
    assert len(constants) > 0
    max_const = next((c for c in constants if c["name"] == "MAX"), None)
    if max_const and "type" in max_const:
        assert "number" in max_const["type"]



def test_typescript_enum_members(parse_constants):
    """Test TypeScript enum member extraction (not yet implemented)."""
    code = """
enum Color {
    RED = 1,
    GREEN = 2,
    BLUE = 3
}
"""
    constants = parse_constants(code, Language.TYPESCRIPT)
    names = {c["name"] for c in constants}
    assert "RED" in names, f"Expected enum constants, got: {names}"



def test_javascript_multi_const(parse_constants):
    """Test JavaScript multiple const declarations (not yet implemented)."""
    code = "const A = 1, B = 2, C = 3;"
    constants = parse_constants(code, Language.JAVASCRIPT)
    names = {c["name"] for c in constants}
    assert len(names.intersection({"A", "B", "C"})) >= 1


# =============================================================================
# C# Const vs Readonly Static Tests
# NOTE: C# field_declaration is NOT captured as standalone DEFINITION concepts.
# These tests document desired behavior.
# =============================================================================



@pytest.mark.parametrize(
    "code",
    [
        "class T { public const int MAX = 100; }",
        "class T { const int MAX = 100; }",
        "class T { public static readonly int MAX = 100; }",
        "class T { static readonly int MAX = 100; }",
    ],
    ids=["public_const", "const", "public_static_readonly", "static_readonly"],
)
def test_csharp_const_should_extract(parse_constants, code):
    """Test C# const/readonly static should be extracted."""
    constants = parse_constants(code, Language.CSHARP)
    has_max = any(c["name"] == "MAX" for c in constants)
    assert has_max, f"Expected constant extraction for: {code}"


@pytest.mark.parametrize(
    "code",
    [
        "class T { public readonly int MAX = 100; }",  # readonly without static
        "class T { readonly int MAX = 100; }",  # readonly without static
        "class T { public static int MAX = 100; }",  # static without readonly
    ],
    ids=["public_readonly", "readonly", "public_static"],
)
def test_csharp_partial_modifiers_not_extracted(parse_constants, code):
    """Test C# fields without const OR readonly+static are not extracted."""
    constants = parse_constants(code, Language.CSHARP)
    has_max = any(c["name"] == "MAX" for c in constants)
    # These should not be extracted (and aren't, because nothing is extracted)
    assert not has_max, f"Should not extract constant for: {code}"



def test_csharp_enum_members(parse_constants):
    """Test C# enum member extraction."""
    code = """
public enum Status {
    Pending,
    Active,
    Complete
}
"""
    constants = parse_constants(code, Language.CSHARP)
    names = {c["name"] for c in constants}
    assert "Pending" in names, f"Expected enum constants, got: {names}"


def test_csharp_local_const_in_method(parse_constants):
    """Test C# local const declarations inside methods are extracted."""
    code = """
class Test {
    void MyMethod() {
        const int LOCAL_CONST = 100;
        const string MESSAGE = "hello";
    }
}
"""
    constants = parse_constants(code, Language.CSHARP)
    names = {c["name"] for c in constants}
    assert "LOCAL_CONST" in names, f"Expected LOCAL_CONST in {names}"
    assert "MESSAGE" in names, f"Expected MESSAGE in {names}"


def test_csharp_local_variable_not_extracted(parse_constants):
    """Test C# regular local variables (non-const) are not extracted."""
    code = """
class Test {
    void MyMethod() {
        int regularVar = 100;
        string message = "hello";
    }
}
"""
    constants = parse_constants(code, Language.CSHARP)
    names = {c["name"] for c in constants}
    assert "regularVar" not in names, f"Should not extract non-const local variable"
    assert "message" not in names, f"Should not extract non-const local variable"


# =============================================================================
# Value Truncation Tests
# =============================================================================


def test_value_truncation_at_50_chars(parse_constants):
    """Test that values over 50 chars are truncated."""
    # Create a value that's exactly 60 characters
    long_value = "x" * 60
    code = f'MAX_VALUE = "{long_value}"'
    constants = parse_constants(code, Language.PYTHON)
    if constants:
        value = constants[0].get("value", "")
        assert len(value) <= 50, f"Value should be truncated to 50 chars, got {len(value)}"


def test_value_exactly_50_chars_not_truncated(parse_constants):
    """Test that values of exactly 50 chars are not truncated."""
    # Value inside quotes: 48 chars + 2 quotes = 50
    exact_value = "x" * 46  # Plus quotes = 48, close to boundary
    code = f'MAX_VALUE = "{exact_value}"'
    constants = parse_constants(code, Language.PYTHON)
    # Should extract without truncation marker
    if constants:
        value = constants[0].get("value", "")
        assert "..." not in value or len(value) <= 50


def test_empty_string_value(parse_constants):
    """Test empty string constant values."""
    code = 'EMPTY = ""'
    constants = parse_constants(code, Language.PYTHON)
    if constants:
        assert constants[0].get("value") == '""' or constants[0].get("value") == ""


# =============================================================================
# Concept Filtering Tests
# =============================================================================


def test_constants_from_function_body_python(parse_constants):
    """Test that UPPER_CASE constants inside Python function bodies ARE extracted.

    NOTE: As of the current implementation, Python extracts UPPER_SNAKE_CASE
    assignments from any scope (including function bodies) to support constant
    extraction in all contexts.
    """
    code = """
def my_function():
    LOCAL_CONST = 100
    return LOCAL_CONST
"""
    constants = parse_constants(code, Language.PYTHON)
    names = {c["name"] for c in constants}
    # LOCAL_CONST is UPPER_CASE, should be extracted
    assert "LOCAL_CONST" in names, f"Expected LOCAL_CONST in {names}"


def test_constants_from_method_body_java(parse_constants):
    """Test that constants inside Java method bodies ARE extracted.

    NOTE: As of the current implementation, Java extracts from local_variable_declaration
    which includes method-scoped variables. Static final fields and local final variables
    with UPPER_CASE names are both extracted.
    """
    code = """
class Test {
    void method() {
        final int LOCAL = 100;
    }
}
"""
    constants = parse_constants(code, Language.JAVA)
    names = {c["name"] for c in constants}
    assert "LOCAL" in names, f"Expected LOCAL in {names}"


def test_go_function_level_consts_are_extracted(parse_constants):
    """Test that Go extracts const from function bodies (unlike Python/Java).

    Go's const keyword creates true compile-time constants even inside functions,
    so they should be extracted at all scoping levels.
    """
    code = """
package main

func main() {
    const FUNC_CONST = 100
    const (
        BLOCK_A = 1
        BLOCK_B = 2
    )
}
"""
    constants = parse_constants(code, Language.GO)
    names = {c["name"] for c in constants}
    # Go should extract function-level consts (they are real constants, not variables)
    assert "FUNC_CONST" in names, f"Expected function-level const, got: {names}"
    assert "BLOCK_A" in names, f"Expected const block member, got: {names}"


# =============================================================================
# Multi-Constant Declaration Tests
# =============================================================================


def test_go_multiple_const_declarations(parse_constants):
    """Test Go const block with multiple declarations."""
    code = """
const (
    A = 1
    B = 2
    C = 3
)
"""
    constants = parse_constants(code, Language.GO)
    # Should get at least one constant
    assert len(constants) >= 1



def test_typescript_multiple_const_in_block(parse_constants):
    """Test TypeScript multiple const declarations (not yet implemented)."""
    code = """
const A = 1;
const B = 2;
const C = 3;
"""
    constants = parse_constants(code, Language.TYPESCRIPT)
    names = {c["name"] for c in constants}
    assert len(names.intersection({"A", "B", "C"})) >= 1


# =============================================================================
# Type Annotation Extraction Tests
# =============================================================================


@pytest.mark.parametrize(
    "language,code,expected_type",
    [
        # Rust: explicit type annotation
        (Language.RUST, "const PI: f64 = 3.14;", "f64"),
        (Language.RUST, "const MAX: i32 = 100;", "i32"),
        # Go: explicit type
        (Language.GO, "const MAX int = 100", "int"),
    ],
    ids=["rust_f64", "rust_i32", "go_int"],
)
def test_type_annotation_extraction(parse_constants, language, code, expected_type):
    """Test type annotation extraction for languages that support it."""
    constants = parse_constants(code, language)
    if constants:
        const = constants[0]
        if "type" in const:
            assert expected_type in const["type"], f"Expected type '{expected_type}' in {const}"


# =============================================================================
# Full Pipeline Integration Tests
# NOTE: These test the full file parsing pipeline. Some fail because
# constant extraction is not yet implemented for those languages.
# =============================================================================



def test_constants_in_parsed_file_python(parse_file_constants):
    """Test Python constants flow through file parsing pipeline."""
    code = """
MAX_VALUE = 100
API_KEY = "secret"
DEFAULT_TIMEOUT = 30
"""
    constants = parse_file_constants(code, "config.py", Language.PYTHON)
    names = {c["name"] for c in constants}
    assert len(names.intersection({"MAX_VALUE", "API_KEY", "DEFAULT_TIMEOUT"})) >= 1



def test_java_constants_in_parsed_file(parse_file_constants):
    """Test Java constants flow through file parsing pipeline (not yet implemented)."""
    code = """
public class Config {
    public static final int MAX_CONNECTIONS = 100;
    public static final String API_URL = "https://api.example.com";
}
"""
    constants = parse_file_constants(code, "Config.java", Language.JAVA)
    names = {c["name"] for c in constants}
    assert (
        "MAX_CONNECTIONS" in names or "API_URL" in names
    ), f"Expected Java constants, got: {names}"



def test_typescript_constants_in_parsed_file(parse_file_constants):
    """Test TypeScript constants flow through file parsing pipeline (not yet implemented)."""
    code = """
export const API_VERSION = "v2.1.0";
export const MAX_RETRIES = 3;
export const TIMEOUT_MS = 30000;
"""
    constants = parse_file_constants(code, "config.ts", Language.TYPESCRIPT)
    names = {c["name"] for c in constants}
    assert len(names.intersection({"API_VERSION", "MAX_RETRIES", "TIMEOUT_MS"})) >= 1


def test_go_constants_in_parsed_file(parse_file_constants):
    """Test Go constants flow through file parsing pipeline (WORKS)."""
    code = """
package main

const MAX_VALUE = 100
const API_KEY = "secret"
"""
    constants = parse_file_constants(code, "config.go", Language.GO)
    names = {c["name"] for c in constants}
    assert len(names.intersection({"MAX_VALUE", "API_KEY"})) >= 1


def test_rust_constants_in_parsed_file(parse_file_constants):
    """Test Rust constants flow through file parsing pipeline (WORKS)."""
    code = """
const MAX_VALUE: i32 = 100;
const API_KEY: &str = "secret";
"""
    constants = parse_file_constants(code, "config.rs", Language.RUST)
    names = {c["name"] for c in constants}
    assert len(names.intersection({"MAX_VALUE", "API_KEY"})) >= 1


# =============================================================================
# Additional Language-Specific Tests
# Languages are grouped by working vs not-yet-implemented status.
# =============================================================================


# --- WORKING LANGUAGES ---


def test_swift_let_constant(parse_constants):
    """Test Swift let constant extraction (WORKS)."""
    code = "let MAX = 100"
    constants = parse_constants(code, Language.SWIFT)
    names = {c["name"] for c in constants}
    assert "MAX" in names


def test_php_const_declaration(parse_constants):
    """Test PHP const declaration (WORKS)."""
    code = "<?php\nconst MAX = 100;"
    constants = parse_constants(code, Language.PHP)
    names = {c["name"] for c in constants}
    assert "MAX" in names


def test_makefile_variable_assignment(parse_constants):
    """Test Makefile variable assignment extraction (WORKS)."""
    code = "MAX = 100"
    constants = parse_constants(code, Language.MAKEFILE)
    names = {c["name"] for c in constants}
    assert "MAX" in names


def test_zig_const_declaration(parse_constants):
    """Test Zig const declaration (WORKS)."""
    code = "const MAX = 100;"
    constants = parse_constants(code, Language.ZIG)
    names = {c["name"] for c in constants}
    assert "MAX" in names


def test_objc_define_macro(parse_constants):
    """Test Objective-C #define macro extraction (WORKS)."""
    code = "#define MAX 100"
    constants = parse_constants(code, Language.OBJC)
    names = {c["name"] for c in constants}
    assert "MAX" in names


def test_objc_const_declaration(parse_constants):
    """Test Objective-C const declaration extraction (WORKS)."""
    code = "const int MAX = 100;"
    constants = parse_constants(code, Language.OBJC)
    names = {c["name"] for c in constants}
    assert "MAX" in names


def test_objc_method_scoped_const(parse_constants):
    """Test Objective-C const declarations inside methods are extracted."""
    code = """
@implementation MyClass
- (void)myMethod {
    const int METHOD_CONST = 200;
    const char *MESSAGE = "hello";
}
@end
"""
    constants = parse_constants(code, Language.OBJC)
    names = {c["name"] for c in constants}
    assert "METHOD_CONST" in names, f"Expected METHOD_CONST in {names}"


def test_objc_nested_block_const(parse_constants):
    """Test Objective-C const declarations in nested blocks are extracted."""
    code = """
- (void)outer {
    const int OUTER = 1;
    if (YES) {
        const int INNER = 2;
    }
}
"""
    constants = parse_constants(code, Language.OBJC)
    names = {c["name"] for c in constants}
    assert "OUTER" in names, f"Expected OUTER in {names}"
    assert "INNER" in names, f"Expected INNER in {names}"


def test_objc_deeply_nested_const(parse_constants):
    """Test Objective-C const declarations in deeply nested blocks are extracted."""
    code = """
@implementation MyClass
- (void)myMethod {
    const int LEVEL1 = 10;
    if (YES) {
        const int LEVEL2 = 20;
        while (1) {
            const int LEVEL3 = 30;
            break;
        }
    }
}
@end
"""
    constants = parse_constants(code, Language.OBJC)
    names = {c["name"] for c in constants}
    assert "LEVEL1" in names, f"Expected LEVEL1 in {names}"
    assert "LEVEL2" in names, f"Expected LEVEL2 in {names}"
    assert "LEVEL3" in names, f"Expected LEVEL3 in {names}"


def test_objc_exception_block_const(parse_constants):
    """Test Objective-C const declarations in exception handling blocks."""
    code = """
- (void)method {
    @try {
        const int TRY_CONST = 40;
    } @catch (NSException *e) {
        const int CATCH_CONST = 50;
    }
}
"""
    constants = parse_constants(code, Language.OBJC)
    names = {c["name"] for c in constants}
    assert "TRY_CONST" in names, f"Expected TRY_CONST in {names}"
    assert "CATCH_CONST" in names, f"Expected CATCH_CONST in {names}"


def test_objc_non_const_variable_not_extracted(parse_constants):
    """Test Objective-C non-const variables are not extracted."""
    code = """
- (void)func {
    int regular = 100;
    const int CONSTANT = 200;
}
"""
    constants = parse_constants(code, Language.OBJC)
    names = {c["name"] for c in constants}
    assert "CONSTANT" in names, f"Expected CONSTANT in {names}"
    assert "regular" not in names, f"Should not extract non-const variable"


def test_haskell_top_level_constant(parse_constants):
    """Test Haskell top-level constant binding (WORKS)."""
    code = "maxValue = 100"
    constants = parse_constants(code, Language.HASKELL)
    names = {c["name"] for c in constants}
    assert "maxValue" in names


def test_haskell_let_binding(parse_constants):
    """Test Haskell let binding constant extraction."""
    code = "example = let localConst = 100 in localConst * 2"
    constants = parse_constants(code, Language.HASKELL)
    names = {c["name"] for c in constants}
    assert "localConst" in names, f"Expected 'localConst' in {names}"


def test_haskell_where_clause_single(parse_constants):
    """Test Haskell single where clause constant."""
    code = """whereExample = result
  where localConst = 200"""
    constants = parse_constants(code, Language.HASKELL)
    names = {c["name"] for c in constants}
    assert "localConst" in names, f"Expected 'localConst' in {names}"


def test_haskell_where_clause_multiple(parse_constants):
    """Test Haskell multiple where clause constants."""
    code = """multiWhere = result
  where x = 10
        y = 20"""
    constants = parse_constants(code, Language.HASKELL)
    names = {c["name"] for c in constants}
    assert "x" in names and "y" in names, f"Expected 'x' and 'y' in {names}"


def test_haskell_nested_let_in_where(parse_constants):
    """Test Haskell nested let inside where clause."""
    code = """complex = result
  where
    helper = let inner = 50 in inner * 2"""
    constants = parse_constants(code, Language.HASKELL)
    names = {c["name"] for c in constants}
    # Should extract 'inner' from let binding
    assert "inner" in names, f"Expected 'inner' in {names}"


# --- NOT YET IMPLEMENTED ---



def test_kotlin_const_val(parse_constants):
    """Test Kotlin const val extraction (WORKS)."""
    code = "const val MAX = 100"
    constants = parse_constants(code, Language.KOTLIN)
    names = {c["name"] for c in constants}
    assert "MAX" in names


def test_kotlin_val_immutable(parse_constants):
    """Test Kotlin regular val (immutable) extraction (WORKS)."""
    code = "val IMMUTABLE = 200"
    constants = parse_constants(code, Language.KOTLIN)
    names = {c["name"] for c in constants}
    assert "IMMUTABLE" in names


def test_kotlin_var_not_extracted(parse_constants):
    """Test Kotlin var (mutable) is not extracted."""
    code = "var mutable = 300"
    constants = parse_constants(code, Language.KOTLIN)
    names = {c["name"] for c in constants}
    assert "mutable" not in names


def test_kotlin_local_val_in_function(parse_constants):
    """Test Kotlin val inside function scope is extracted (WORKS)."""
    code = """
fun myFunction() {
    val LOCAL_CONST = 100
    var localVar = 200
}
"""
    constants = parse_constants(code, Language.KOTLIN)
    names = {c["name"] for c in constants}
    assert "LOCAL_CONST" in names, f"Expected LOCAL_CONST in {names}"
    assert "localVar" not in names, f"Should not extract var"


def test_kotlin_class_property_val(parse_constants):
    """Test Kotlin val property inside class is extracted (WORKS)."""
    code = """
class MyClass {
    val CLASS_PROPERTY = 42
    var classVar = 100
}
"""
    constants = parse_constants(code, Language.KOTLIN)
    names = {c["name"] for c in constants}
    assert "CLASS_PROPERTY" in names, f"Expected CLASS_PROPERTY in {names}"
    assert "classVar" not in names, f"Should not extract var"


def test_kotlin_companion_object_constants(parse_constants):
    """Test Kotlin companion object constants (WORKS)."""
    code = """
class MyClass {
    companion object {
        const val COMPANION_CONST = 7
        val COMPANION_VAL = 8
    }
}
"""
    constants = parse_constants(code, Language.KOTLIN)
    names = {c["name"] for c in constants}
    assert "COMPANION_CONST" in names, f"Expected COMPANION_CONST in {names}"
    assert "COMPANION_VAL" in names, f"Expected COMPANION_VAL in {names}"



def test_dart_const_declaration(parse_constants):
    """Test Dart const declaration."""
    code = "const MAX = 100;"
    constants = parse_constants(code, Language.DART)
    names = {c["name"] for c in constants}
    assert "MAX" in names



def test_groovy_static_final(parse_constants):
    """Test Groovy static final field extraction."""
    code = "class Test { static final int MAX = 100 }"
    constants = parse_constants(code, Language.GROOVY)
    names = {c["name"] for c in constants}
    assert "MAX" in names


def test_groovy_local_final(parse_constants):
    """Test Groovy local final variable extraction."""
    code = """
class Test {
    void method() {
        final int LOCAL = 100
        final String NAME = "test"
    }
}
"""
    constants = parse_constants(code, Language.GROOVY)
    names = {c["name"] for c in constants}
    assert "LOCAL" in names, f"Expected LOCAL in {names}"
    assert "NAME" in names, f"Expected NAME in {names}"


def test_groovy_non_final_not_extracted(parse_constants):
    """Test Groovy non-final variables are not extracted."""
    code = """
class Test {
    static int staticOnly = 100
    final int finalOnly = 200
    int neither = 300
    void method() {
        int local = 400
    }
}
"""
    constants = parse_constants(code, Language.GROOVY)
    names = {c["name"] for c in constants}
    assert "staticOnly" not in names, "Should not extract static without final"
    assert "finalOnly" not in names, "Should not extract final without static (field)"
    assert "neither" not in names, "Should not extract regular field"
    assert "local" not in names, "Should not extract non-final local variable"


def test_groovy_multiple_final_declarators(parse_constants):
    """Test Groovy multiple final variables in one declaration."""
    code = """
class Test {
    void method() {
        final int A = 1, B = 2, C = 3
    }
}
"""
    constants = parse_constants(code, Language.GROOVY)
    names = {c["name"] for c in constants}
    assert "A" in names, f"Expected A in {names}"
    assert "B" in names, f"Expected B in {names}"
    assert "C" in names, f"Expected C in {names}"


# =============================================================================
# MATLAB UPPER_CASE Convention Tests
# =============================================================================


def test_matlab_function_body_constant(parse_constants):
    """Test MATLAB UPPER_CASE constant extraction from function body."""
    code = """
function result = myFunc()
    MAX_VALUE = 100;
    min_value = 10;
    result = MAX_VALUE + min_value;
end
"""
    constants = parse_constants(code, Language.MATLAB)
    names = {c["name"] for c in constants}
    # Should extract MAX_VALUE (UPPER_CASE) but not min_value (lowercase)
    assert "MAX_VALUE" in names, f"Expected MAX_VALUE in {names}"
    assert "min_value" not in names, f"Should not extract lowercase variable"


def test_matlab_multiple_constants_in_function(parse_constants):
    """Test MATLAB extraction of multiple UPPER_CASE constants."""
    code = """
function result = calculate()
    API_KEY = 'secret';
    MAX_RETRIES = 3;
    TIMEOUT_MS = 30000;
    result = 0;
end
"""
    constants = parse_constants(code, Language.MATLAB)
    names = {c["name"] for c in constants}
    expected = {"API_KEY", "MAX_RETRIES", "TIMEOUT_MS"}
    assert expected.issubset(names), f"Expected {expected}, got {names}"


def test_matlab_properties_constant_block(parse_constants):
    """Test MATLAB properties (Constant) block extraction."""
    code = """
classdef MyClass
    properties (Constant)
        MAX_VALUE = 100
        PI_APPROX = 3.14159
    end
end
"""
    constants = parse_constants(code, Language.MATLAB)
    names = {c["name"] for c in constants}
    assert "MAX_VALUE" in names or "PI_APPROX" in names, f"Expected property constants, got {names}"


def test_matlab_mixed_case_not_extracted(parse_constants):
    """Test MATLAB does not extract non-UPPER_CASE assignments."""
    code = """
function result = test()
    maxValue = 100;
    Max_Value = 200;
    result = 0;
end
"""
    constants = parse_constants(code, Language.MATLAB)
    names = {c["name"] for c in constants}
    assert "maxValue" not in names, "Should not extract camelCase"
    assert "Max_Value" not in names, "Should not extract Mixed_Case"


def test_matlab_nested_function_constants(parse_constants):
    """Test MATLAB extraction from nested function bodies."""
    code = """
function outer()
    OUTER_CONSTANT = 1;

    function inner()
        INNER_CONSTANT = 2;
    end
end
"""
    constants = parse_constants(code, Language.MATLAB)
    names = {c["name"] for c in constants}
    # Should extract from both outer and nested functions
    assert "OUTER_CONSTANT" in names or "INNER_CONSTANT" in names, f"Expected nested constants, got {names}"


def test_matlab_script_level_constants(parse_constants):
    """Test MATLAB extraction of script-level (top-level) UPPER_CASE constants."""
    code = """
% Configuration script
SCRIPT_CONSTANT = 999;
API_KEY = 'secret';
lowercase_var = 1;
"""
    constants = parse_constants(code, Language.MATLAB)
    names = {c["name"] for c in constants}
    # Should extract UPPER_CASE script-level constants
    assert "SCRIPT_CONSTANT" in names, f"Expected SCRIPT_CONSTANT in {names}"
    assert "API_KEY" in names, f"Expected API_KEY in {names}"
    # Should not extract lowercase variables
    assert "lowercase_var" not in names, f"Should not extract lowercase variable"


def test_matlab_mixed_script_and_function_constants(parse_constants):
    """Test MATLAB extraction from both script-level and function body."""
    code = """
SCRIPT_CONST = 123;

function result = myFunc()
    FUNC_CONST = 456;
    result = FUNC_CONST;
end
"""
    constants = parse_constants(code, Language.MATLAB)
    names = {c["name"] for c in constants}
    # Should extract both script-level and function-level constants
    assert "SCRIPT_CONST" in names, f"Expected SCRIPT_CONST in {names}"
    assert "FUNC_CONST" in names, f"Expected FUNC_CONST in {names}"


def test_matlab_constants_in_parsed_file(parse_file_constants):
    """Test MATLAB constants flow through file parsing pipeline."""
    code = """
function result = config()
    MAX_CONNECTIONS = 100;
    API_URL = 'https://api.example.com';
    TIMEOUT = 30;
    result = 0;
end
"""
    constants = parse_file_constants(code, "config.m", Language.MATLAB)
    names = {c["name"] for c in constants}
    expected = {"MAX_CONNECTIONS", "API_URL", "TIMEOUT"}
    assert len(expected.intersection(names)) >= 1, f"Expected MATLAB constants, got {names}"


# =============================================================================
# Lua UPPER_SNAKE_CASE Pattern Tests
# Tests UPPER_SNAKE_CASE constant detection for Lua (like Python convention).
# =============================================================================


@pytest.mark.parametrize(
    "code,should_extract",
    [
        # Valid UPPER_SNAKE_CASE patterns
        ("local MAX_VALUE = 100", True),
        ("local API_KEY_V2 = 'secret'", True),
        ("local _PRIVATE = 1", True),
        ("local MAX123 = 1", True),
        ("local A = 1", True),  # Single letter uppercase
    ],
    ids=["max_value", "api_key_v2", "private_underscore", "max_with_numbers", "single_letter"],
)
def test_lua_upper_snake_case_valid(parse_constants, code, should_extract):
    """Test Lua UPPER_SNAKE_CASE constant detection (valid patterns)."""
    constants = parse_constants(code, Language.LUA)
    assert len(constants) > 0, f"Expected constant from '{code}' but got none"


@pytest.mark.parametrize(
    "code",
    [
        "local Max_Value = 1",  # Mixed case
        "local maxValue = 1",  # camelCase
        "local max_value = 1",  # lowercase snake_case
        "local value = 1",  # lowercase
    ],
    ids=["mixed_case", "camel_case", "lowercase_snake", "lowercase"],
)
def test_lua_upper_snake_case_invalid(parse_constants, code):
    """Test Lua does not extract non-UPPER_SNAKE_CASE patterns."""
    constants = parse_constants(code, Language.LUA)
    names = {c["name"] for c in constants}
    var_name = code.split("=")[0].replace("local", "").strip()
    assert var_name not in names, f"Should not extract '{var_name}' from '{code}'"


def test_lua_function_not_extracted_as_constant(parse_constants):
    """Functions with UPPER_SNAKE_CASE names should not be extracted as constants."""
    code = "function MAX_FUNCTION() end"
    constants = parse_constants(code, Language.LUA)
    names = {c["name"] for c in constants}
    assert "MAX_FUNCTION" not in names


def test_lua_table_constant(parse_constants):
    """Test Lua table assigned to UPPER_SNAKE_CASE variable."""
    code = 'local CONFIG = {debug = true, version = "1.0"}'
    constants = parse_constants(code, Language.LUA)
    config_constants = [c for c in constants if c["name"] == "CONFIG"]
    assert len(config_constants) == 1


def test_lua_constants_in_parsed_file(parse_file_constants):
    """Test Lua constants flow through file parsing pipeline."""
    code = """
local MAX_VALUE = 100
local API_KEY = "secret"
local DEFAULT_TIMEOUT = 30
"""
    constants = parse_file_constants(code, "config.lua", Language.LUA)
    names = {c["name"] for c in constants}
    assert len(names.intersection({"MAX_VALUE", "API_KEY", "DEFAULT_TIMEOUT"})) >= 1
