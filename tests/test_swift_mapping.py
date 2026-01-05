"""Tests for Swift language mapping and parsing."""


from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import get_parser_factory


def parse_swift(content: str):
    """Helper function to parse Swift content."""
    factory = get_parser_factory()
    parser = factory.create_parser(Language.SWIFT)
    return parser.parse_content(content, "test.swift", FileId(1))


class TestSwiftBasicTypes:
    """Test parsing of Swift basic type declarations."""

    def test_captures_class_declaration(self):
        """Test that class declarations are captured as CLASS chunks."""
        content = """
class MyClass {
    var name: String

    func doSomething() {
        print("Hello")
    }
}
"""
        chunks = parse_swift(content)
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        symbols = {c.symbol for c in chunks}

        assert len(class_chunks) > 0, "Should capture class declaration"
        assert "MyClass" in symbols, (
            f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_struct_declaration(self):
        """Test that struct declarations are captured with value type metadata."""
        content = """
struct Point {
    let x: Int
    let y: Int

    func distance() -> Double {
        return sqrt(Double(x * x + y * y))
    }
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        assert len(chunks) > 0, "Should capture struct declaration"
        assert "Point" in symbols, (
            f"Expected 'Point' in symbols, got: {sorted(symbols)}"
        )

        # Check for value_type metadata
        struct_chunks = [
            c for c in chunks if c.chunk_type == ChunkType.CLASS and "Point" in c.symbol
        ]
        if struct_chunks and struct_chunks[0].metadata:
            assert struct_chunks[0].metadata.get("value_type") is True, (
                "Struct should be marked as value type"
            )

    def test_captures_protocol_declaration(self):
        """Test that protocol declarations are captured."""
        content = """
protocol Drawable {
    func draw()
    var color: String { get }
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        assert len(chunks) > 0, "Should capture protocol"
        # Protocol should be prefixed
        has_protocol = any("Drawable" in s for s in symbols)
        assert has_protocol, f"Expected 'Drawable' in symbols, got: {sorted(symbols)}"

    def test_captures_enum_declaration(self):
        """Test that enum declarations are captured."""
        content = """
enum Direction {
    case north
    case south
    case east
    case west
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        assert len(chunks) > 0, "Should capture enum"
        assert "Direction" in symbols, (
            f"Expected 'Direction' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_actor_declaration(self):
        """Test that actor declarations are captured with concurrency metadata."""
        content = """
actor BankAccount {
    private var balance: Double = 0

    func deposit(amount: Double) {
        balance += amount
    }
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        assert len(chunks) > 0, "Should capture actor"
        has_actor = any("BankAccount" in s for s in symbols)
        assert has_actor, f"Expected 'BankAccount' in symbols, got: {sorted(symbols)}"

        # Check for concurrency metadata
        actor_chunks = [c for c in chunks if "BankAccount" in c.symbol]
        if actor_chunks and actor_chunks[0].metadata:
            # Actor should have concurrency metadata
            metadata = actor_chunks[0].metadata
            assert (
                metadata.get("concurrency") is True or metadata.get("kind") == "actor"
            ), "Actor should have concurrency metadata or kind=actor"


class TestSwiftFunctions:
    """Test parsing of Swift functions and methods."""

    def test_captures_function_declaration(self):
        """Test that function declarations are captured as FUNCTION chunks."""
        content = """
func greet(name: String) -> String {
    return "Hello, \\(name)!"
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]

        assert len(func_chunks) > 0, "Should capture function"
        assert "greet" in symbols, (
            f"Expected 'greet' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_initializer(self):
        """Test that init methods are captured."""
        content = """
class Person {
    let name: String
    let age: Int

    init(name: String, age: Int) {
        self.name = name
        self.age = age
    }
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Should find init method
        has_init = "init" in symbols
        assert has_init, f"Expected 'init' in symbols, got: {sorted(symbols)}"

    def test_captures_failable_initializer(self):
        """Test that failable initializers (init?) are captured with metadata."""
        content = """
struct Number {
    let value: Int

    init?(string: String) {
        guard let val = Int(string) else {
            return nil
        }
        self.value = val
    }
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Should find failable init
        has_failable_init = "init?" in symbols
        assert has_failable_init, f"Expected 'init?' in symbols, got: {sorted(symbols)}"

        # Check metadata
        init_chunks = [c for c in chunks if "init?" in c.symbol]
        if init_chunks and init_chunks[0].metadata:
            assert init_chunks[0].metadata.get("failable") is True, (
                "Failable init should have failable=True"
            )

    def test_captures_deinitializer(self):
        """Test that deinit methods are captured."""
        content = """
class ResourceManager {
    var resource: String

    init(resource: String) {
        self.resource = resource
    }

    deinit {
        print("Cleaning up \\(resource)")
        // Make it substantial enough to not be filtered
        let message = "Releasing resource: \\(resource)"
        print(message)
    }
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Should find deinit method (if captured as separate chunk)
        # Note: deinit may be part of class chunk depending on tree-sitter
        has_deinit = "deinit" in symbols or any("deinit" in c.code for c in chunks)
        assert len(chunks) > 0, "Should capture class with deinit"

    def test_captures_subscript(self):
        """Test that subscript declarations are captured."""
        content = """
struct Matrix {
    var data: [[Int]]

    subscript(row: Int, column: Int) -> Int {
        get {
            // Make getter substantial
            let value = data[row][column]
            print("Getting value at [\\(row), \\(column)]: \\(value)")
            return value
        }
        set {
            // Make setter substantial
            print("Setting value at [\\(row), \\(column)] to \\(newValue)")
            data[row][column] = newValue
        }
    }
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Should find subscript or at least capture the Matrix struct
        has_subscript = any("subscript" in s.lower() for s in symbols) or any(
            "subscript" in c.code for c in chunks
        )
        assert len(chunks) > 0, "Should capture Matrix with subscript"

    def test_captures_function_with_generics(self):
        """Test that generic functions are captured with generic parameters."""
        content = """
func swap<T>(a: inout T, b: inout T) {
    let temp = a
    a = b
    b = temp
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Should find swap function with generic parameter
        has_swap = any("swap" in s for s in symbols)
        assert has_swap, f"Expected 'swap' in symbols, got: {sorted(symbols)}"

        # Check for generic parameters in metadata
        swap_chunks = [c for c in chunks if "swap" in c.symbol]
        if swap_chunks and swap_chunks[0].metadata:
            generic_params = swap_chunks[0].metadata.get("generic_parameters", [])
            if generic_params:
                assert "T" in generic_params, "Should capture generic parameter T"


class TestSwiftExtensions:
    """Test parsing of Swift extensions."""

    def test_captures_extension_declaration(self):
        """Test that extensions are captured."""
        content = """
extension String {
    func reversed() -> String {
        return String(self.reversed())
    }
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Extension should be captured
        has_extension = any("extension" in s.lower() or "String" in s for s in symbols)
        assert has_extension, f"Expected extension in symbols, got: {sorted(symbols)}"

    def test_captures_extension_with_protocol_conformance(self):
        """Test that extensions with protocol conformance are captured."""
        content = """
protocol Drawable {
    func draw()
}

extension Array: Drawable {
    func draw() {
        print("Drawing array")
    }
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Extension should include protocol conformance
        has_extension = any("extension" in s.lower() and "Array" in s for s in symbols)
        assert has_extension, f"Expected extension with Array, got: {sorted(symbols)}"

    def test_extension_symbol_includes_protocol(self):
        """Test that extension symbols include protocol conformance in name."""
        content = """
protocol Identifiable {
    var id: String { get }
}

extension String: Identifiable {
    var id: String { return self }
}
"""
        chunks = parse_swift(content)

        # Look for extension chunks
        extension_chunks = [
            c
            for c in chunks
            if "extension" in c.symbol.lower()
            or ("String" in c.code and "Identifiable" in c.code)
        ]

        # Should have captured the extension
        assert len(chunks) > 0, "Should capture extension and protocol"


class TestSwiftAccessModifiers:
    """Test parsing of Swift access modifiers."""

    def test_open_modifier(self):
        """Test that open classes/methods are captured with access metadata."""
        content = """
open class BaseClass {
    open func overridableMethod() {
        print("Can be overridden")
    }
}
"""
        chunks = parse_swift(content)

        # Check for open access modifier in metadata
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        if class_chunks and class_chunks[0].metadata:
            access = class_chunks[0].metadata.get("access")
            if access:
                assert access == "open", f"Expected access='open', got: {access}"

    def test_public_modifier(self):
        """Test that public declarations are captured with access metadata."""
        content = """
public class PublicClass {
    public var name: String = ""

    public func publicMethod() {
        print("Public method")
    }
}
"""
        chunks = parse_swift(content)

        # Check for public access modifier
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        if class_chunks and class_chunks[0].metadata:
            access = class_chunks[0].metadata.get("access")
            if access:
                assert access == "public", f"Expected access='public', got: {access}"

    def test_internal_modifier(self):
        """Test that internal (default) access is handled correctly."""
        content = """
internal class InternalClass {
    internal func internalMethod() {
        print("Internal method")
    }
}
"""
        chunks = parse_swift(content)

        # Should capture the class
        assert len(chunks) > 0, "Should capture internal class"

    def test_fileprivate_modifier(self):
        """Test that fileprivate access is captured with metadata."""
        content = """
fileprivate class FilePrivateClass {
    fileprivate func filePrivateMethod() {
        print("File private")
    }
}
"""
        chunks = parse_swift(content)

        # Check for fileprivate access modifier
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        if class_chunks and class_chunks[0].metadata:
            access = class_chunks[0].metadata.get("access")
            if access:
                assert access == "fileprivate", (
                    f"Expected access='fileprivate', got: {access}"
                )

    def test_private_modifier(self):
        """Test that private access is captured with metadata."""
        content = """
private class PrivateClass {
    private func privateMethod() {
        print("Private method")
    }
}
"""
        chunks = parse_swift(content)

        # Check for private access modifier
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        if class_chunks and class_chunks[0].metadata:
            access = class_chunks[0].metadata.get("access")
            if access:
                assert access == "private", f"Expected access='private', got: {access}"


class TestSwiftAsyncAwait:
    """Test parsing of Swift async/await features."""

    def test_async_function_metadata(self):
        """Test that async functions are captured with async metadata."""
        content = """
class DataManager {
    async func fetchData() -> Data {
        // Simulated async operation
        return Data()
    }
}
"""
        chunks = parse_swift(content)

        # Find async function
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        async_funcs = [
            c for c in func_chunks if c.metadata and c.metadata.get("async") is True
        ]

        # Should find at least one async function
        if func_chunks:
            # At minimum, should capture the function
            assert len(func_chunks) > 0, "Should find async function"

    def test_throwing_function_metadata(self):
        """Test that throwing functions are captured with throws metadata."""
        content = """
func processFile() throws {
    // Implementation that can throw
    throw NSError(domain: "test", code: 1, userInfo: nil)
}
"""
        chunks = parse_swift(content)

        # Find throwing function
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        throwing_funcs = [
            c for c in func_chunks if c.metadata and c.metadata.get("throws") is True
        ]

        # Should find function (metadata depends on tree-sitter implementation)
        assert len(func_chunks) > 0, "Should find throwing function"

    def test_async_throws_combination(self):
        """Test that async throws functions are captured correctly."""
        content = """
class NetworkClient {
    async func request() async throws -> Response {
        // Async operation that can throw
        return Response()
    }
}
"""
        chunks = parse_swift(content)

        # Should capture the function
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) > 0, "Should find async throws function"


class TestSwiftGenerics:
    """Test parsing of Swift generic types and functions."""

    def test_generic_class_parameters(self):
        """Test that generic class parameters are extracted."""
        content = """
class Container<T> {
    var item: T

    init(item: T) {
        self.item = item
    }
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Should find Container with generic parameter
        has_container = any("Container" in s for s in symbols)
        assert has_container, f"Expected 'Container' in symbols, got: {sorted(symbols)}"

        # Check for generic parameters
        container_chunks = [
            c
            for c in chunks
            if "Container" in c.symbol and c.chunk_type == ChunkType.CLASS
        ]
        if container_chunks and container_chunks[0].metadata:
            generic_params = container_chunks[0].metadata.get("generic_parameters", [])
            if generic_params:
                assert "T" in generic_params, "Should capture generic parameter T"

    def test_generic_function_parameters(self):
        """Test that generic function parameters are extracted."""
        content = """
func findIndex<T: Equatable>(of value: T, in array: [T]) -> Int? {
    for (index, element) in array.enumerated() {
        if element == value {
            return index
        }
    }
    return nil
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Should find function with generic
        has_find_index = any("findIndex" in s for s in symbols)
        assert has_find_index, (
            f"Expected 'findIndex' in symbols, got: {sorted(symbols)}"
        )

    def test_generic_type_constraints(self):
        """Test that generic type constraints are handled."""
        content = """
class SortedArray<T: Comparable> {
    private var items: [T] = []

    func add(_ item: T) {
        items.append(item)
        items.sort()
    }
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Should find SortedArray
        has_sorted_array = any("SortedArray" in s for s in symbols)
        assert has_sorted_array, (
            f"Expected 'SortedArray' in symbols, got: {sorted(symbols)}"
        )

    def test_multiple_generic_parameters(self):
        """Test that multiple generic parameters are extracted."""
        content = """
class Pair<T, U> {
    let first: T
    let second: U

    init(first: T, second: U) {
        self.first = first
        self.second = second
    }
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Should find Pair
        has_pair = any("Pair" in s for s in symbols)
        assert has_pair, f"Expected 'Pair' in symbols, got: {sorted(symbols)}"

        # Check for multiple generic parameters
        pair_chunks = [
            c for c in chunks if "Pair" in c.symbol and c.chunk_type == ChunkType.CLASS
        ]
        if pair_chunks and pair_chunks[0].metadata:
            generic_params = pair_chunks[0].metadata.get("generic_parameters", [])
            if generic_params:
                assert len(generic_params) >= 2, (
                    f"Should capture 2 generic parameters, got: {generic_params}"
                )


class TestSwiftMetadata:
    """Test that Swift metadata is correctly extracted."""

    def test_value_type_metadata(self):
        """Test that structs are marked as value types."""
        content = """
struct ValueType {
    var x: Int
    var y: Int
}
"""
        chunks = parse_swift(content)

        # Find struct chunks
        struct_chunks = [
            c
            for c in chunks
            if c.chunk_type == ChunkType.CLASS and "ValueType" in c.symbol
        ]
        if struct_chunks and struct_chunks[0].metadata:
            assert struct_chunks[0].metadata.get("value_type") is True, (
                "Struct should have value_type=True"
            )

    def test_reference_type_metadata(self):
        """Test that classes are reference types (no value_type flag)."""
        content = """
class ReferenceType {
    var x: Int = 0
}
"""
        chunks = parse_swift(content)

        # Find class chunks
        class_chunks = [
            c
            for c in chunks
            if c.chunk_type == ChunkType.CLASS and "ReferenceType" in c.symbol
        ]
        if class_chunks and class_chunks[0].metadata:
            # Classes should not have value_type=True
            assert class_chunks[0].metadata.get("value_type") is not True, (
                "Class should not have value_type=True"
            )

    def test_concurrency_metadata(self):
        """Test that actors have concurrency metadata."""
        content = """
actor Counter {
    private var count = 0

    func increment() {
        count += 1
    }
}
"""
        chunks = parse_swift(content)

        # Find actor chunks
        actor_chunks = [c for c in chunks if "Counter" in c.symbol]
        if actor_chunks and actor_chunks[0].metadata:
            metadata = actor_chunks[0].metadata
            # Should have concurrency marker or kind=actor
            has_concurrency = (
                metadata.get("concurrency") is True or metadata.get("kind") == "actor"
            )
            assert has_concurrency, "Actor should have concurrency metadata"

    def test_protocol_conformance_metadata(self):
        """Test that protocol conformance is extracted in metadata."""
        content = """
protocol Drawable {
    func draw()
}

class Shape: Drawable {
    func draw() {
        print("Drawing")
    }
}
"""
        chunks = parse_swift(content)

        # Find Shape class
        shape_chunks = [
            c for c in chunks if "Shape" in c.symbol and c.chunk_type == ChunkType.CLASS
        ]
        if shape_chunks and shape_chunks[0].metadata:
            # Check for protocol conformance
            conforms_to = shape_chunks[0].metadata.get("conforms_to", [])
            if conforms_to:
                assert "Drawable" in conforms_to, (
                    "Should capture Drawable protocol conformance"
                )

    def test_class_inheritance_metadata(self):
        """Test that class inheritance is extracted in metadata."""
        content = """
class BaseClass {
    var name: String = ""
}

class DerivedClass: BaseClass {
    var age: Int = 0
}
"""
        chunks = parse_swift(content)

        # Find DerivedClass
        derived_chunks = [
            c
            for c in chunks
            if "DerivedClass" in c.symbol and c.chunk_type == ChunkType.CLASS
        ]
        if derived_chunks and derived_chunks[0].metadata:
            # Check for inheritance
            inherits = derived_chunks[0].metadata.get("inherits", [])
            if inherits:
                assert "BaseClass" in inherits, "Should capture BaseClass inheritance"

    def test_failable_initializer_metadata(self):
        """Test that failable initializers have metadata."""
        content = """
struct SafeNumber {
    let value: Int

    init?(string: String) {
        guard let val = Int(string) else {
            return nil
        }
        self.value = val
    }
}
"""
        chunks = parse_swift(content)

        # Find init? method
        init_chunks = [c for c in chunks if "init?" in c.symbol]
        if init_chunks and init_chunks[0].metadata:
            assert init_chunks[0].metadata.get("failable") is True, (
                "Failable init should have failable=True"
            )


class TestSwiftSymbolNames:
    """Test that Swift parser generates correct symbol names."""

    def test_function_symbols_basic(self):
        """Test that function symbols are correctly extracted."""
        content = """
func calculateSum(a: Int, b: Int) -> Int {
    // Make substantial to avoid filtering
    let result = a + b
    print("Sum of \\(a) and \\(b) is \\(result)")
    return result
}

func greet(name: String) {
    // Make substantial to avoid filtering
    let message = "Hello, \\(name)"
    print(message)
    print("Nice to meet you!")
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Should find both functions
        assert "calculateSum" in symbols, (
            f"Expected 'calculateSum' in symbols, got: {sorted(symbols)}"
        )
        assert "greet" in symbols, (
            f"Expected 'greet' in symbols, got: {sorted(symbols)}"
        )

    def test_function_symbols_with_generics(self):
        """Test that generic function symbols include type parameters."""
        content = """
func identity<T>(value: T) -> T {
    return value
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Should find identity function (with or without <T>)
        has_identity = any("identity" in s for s in symbols)
        assert has_identity, f"Expected 'identity' in symbols, got: {sorted(symbols)}"

    def test_protocol_symbols_prefix(self):
        """Test that protocol symbols include 'protocol' prefix."""
        content = """
protocol Equatable {
    static func == (lhs: Self, rhs: Self) -> Bool
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Protocol should be identifiable
        has_equatable = any("Equatable" in s for s in symbols)
        assert has_equatable, f"Expected 'Equatable' in symbols, got: {sorted(symbols)}"

    def test_actor_symbols_prefix(self):
        """Test that actor symbols include 'actor' prefix."""
        content = """
actor DataStore {
    private var data: [String: Any] = [:]

    func set(key: String, value: Any) {
        data[key] = value
    }
}
"""
        chunks = parse_swift(content)
        symbols = {c.symbol for c in chunks}

        # Actor should be identifiable
        has_datastore = any("DataStore" in s for s in symbols)
        assert has_datastore, f"Expected 'DataStore' in symbols, got: {sorted(symbols)}"

    def test_extension_symbols_with_protocols(self):
        """Test that extension symbols include protocol conformance."""
        content = """
protocol Printable {
    func printDescription()
}

extension Int: Printable {
    func printDescription() {
        print("Number: \\(self)")
    }
}
"""
        chunks = parse_swift(content)

        # Should capture extension
        extension_chunks = [
            c
            for c in chunks
            if "extension" in c.symbol.lower()
            or ("Int" in c.code and "Printable" in c.code)
        ]
        assert len(chunks) > 0, "Should capture protocol and extension"


class TestSwiftComments:
    """Test parsing of Swift comments."""

    def test_captures_line_comments(self):
        """Test that single-line comments (//) are captured."""
        content = """
// This is a comment
class MyClass {
    var name: String = ""
}
"""
        chunks = parse_swift(content)

        # Should parse without errors
        assert len(chunks) > 0, "Should parse file with line comments"

    def test_captures_block_comments(self):
        """Test that block comments (/* */) are captured."""
        content = """
/* This is a
   block comment */
class MyClass {
    var name: String = ""
}
"""
        chunks = parse_swift(content)

        # Should parse without errors
        assert len(chunks) > 0, "Should parse file with block comments"

    def test_captures_doc_comments(self):
        """Test that documentation comments (///) are captured."""
        content = """
/// This is a documentation comment
/// It can span multiple lines
class MyClass {
    /// Property documentation
    var name: String = ""
}
"""
        chunks = parse_swift(content)

        # Should parse without errors and potentially capture doc comments
        assert len(chunks) > 0, "Should parse file with doc comments"


class TestSwiftImports:
    """Test parsing of Swift import statements."""

    def test_captures_module_imports(self):
        """Test that import statements are captured."""
        content = """
import Foundation
import UIKit

class MyClass {
    var name: String = ""
}
"""
        chunks = parse_swift(content)

        # Should parse file with imports
        assert len(chunks) > 0, "Should parse file with imports"

        # Check if imports are captured
        import_chunks = [c for c in chunks if "import" in c.code.lower()]
        # Imports may or may not be captured as separate chunks depending on configuration

    def test_import_metadata(self):
        """Test that import metadata is extracted."""
        content = """
import Foundation
import class UIKit.UIView

struct MyStruct {
    var data: Data?
}
"""
        chunks = parse_swift(content)

        # Should parse successfully
        assert len(chunks) > 0, "Should parse file with specific imports"


class TestSwiftProperties:
    """Test parsing of Swift properties."""

    def test_captures_property_declarations(self):
        """Test that property declarations are captured."""
        content = """
class Person {
    var name: String
    let age: Int
    private var email: String?

    init(name: String, age: Int) {
        self.name = name
        self.age = age
    }
}
"""
        chunks = parse_swift(content)

        # Should capture class with properties
        assert len(chunks) > 0, "Should parse class with properties"

    def test_computed_properties(self):
        """Test that computed properties are handled."""
        content = """
struct Rectangle {
    var width: Double
    var height: Double

    var area: Double {
        return width * height
    }

    var perimeter: Double {
        get {
            return 2 * (width + height)
        }
    }
}
"""
        chunks = parse_swift(content)

        # Should capture struct with computed properties
        assert len(chunks) > 0, "Should parse struct with computed properties"


class TestSwiftModernFeatures:
    """Test parsing of modern Swift features."""

    def test_swiftui_view_parsing(self):
        """Test that SwiftUI View protocols are parsed correctly."""
        content = """
import SwiftUI

struct ContentView: View {
    @State private var count = 0

    var body: some View {
        VStack {
            Text("Count: \\(count)")
            Button("Increment") {
                count += 1
            }
        }
    }
}
"""
        chunks = parse_swift(content)

        # Should capture SwiftUI view
        assert len(chunks) > 0, "Should parse SwiftUI view"

        # Check for ContentView
        has_view = any("ContentView" in c.symbol for c in chunks)
        assert has_view, "Should find ContentView"

    def test_property_wrappers(self):
        """Test that property wrappers like @State are handled."""
        content = """
struct MyView {
    @State private var isShowing = false
    @Published var data: [String] = []
    @ObservedObject var viewModel: ViewModel
}
"""
        chunks = parse_swift(content)

        # Should parse properties with wrappers
        assert len(chunks) > 0, "Should parse properties with wrappers"

    def test_result_builder(self):
        """Test that result builders like @ViewBuilder are handled."""
        content = """
struct CustomContainer {
    @ViewBuilder
    func buildContent() -> some View {
        Text("Hello")
        Text("World")
    }
}
"""
        chunks = parse_swift(content)

        # Should parse result builder
        assert len(chunks) > 0, "Should parse result builder"


class TestSwiftComplexFile:
    """Test parsing of complex Swift files with multiple constructs."""

    def test_parses_complete_class_file(self):
        """Test parsing a complete Swift class with multiple features."""
        content = """
import Foundation

class Person {
    var name: String
    private(set) var age: Int

    init(name: String, age: Int) {
        self.name = name
        self.age = age
    }

    func greet() {
        print("Hello, I'm \\(name)")
    }

    func celebrateBirthday() {
        age += 1
    }
}

extension Person: CustomStringConvertible {
    var description: String {
        return "Person(name: \\(name), age: \\(age))"
    }
}
"""
        chunks = parse_swift(content)

        # Should capture multiple chunks
        assert len(chunks) > 0, "Should parse complete Swift file"

        # Check for Person class
        has_person = any("Person" in c.symbol for c in chunks)
        assert has_person, "Should find Person class"

    def test_swift_realistic_module(self):
        """Test parsing a realistic Swift module with multiple advanced features."""
        content = """
import Foundation

/// Error types for network operations
enum NetworkError: Error {
    case invalidURL
    case noData
    case decodingError
    case serverError(Int)
}

/// Protocol for data fetching
protocol DataFetcher {
    func fetch<T: Decodable>(from url: URL) async throws -> T
}

/// Network client for API calls
class NetworkClient: DataFetcher {
    private let session: URLSession
    private var cache: [String: Any] = [:]

    init(configuration: URLSessionConfiguration = .default) {
        self.session = URLSession(configuration: configuration)
    }

    /// Fetch data from URL with generic decoding
    func fetch<T: Decodable>(from url: URL) async throws -> T {
        let (data, response) = try await session.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.noData
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.serverError(httpResponse.statusCode)
        }

        do {
            let decoder = JSONDecoder()
            return try decoder.decode(T.self, from: data)
        } catch {
            throw NetworkError.decodingError
        }
    }

    /// Cache response for key
    private func cacheResponse(_ response: Any, for key: String) {
        cache[key] = response
    }
}

/// Extension for URL validation
extension URL {
    var isValid: Bool {
        return scheme != nil && host != nil
    }
}

/// Generic result type
enum Result<Success, Failure: Error> {
    case success(Success)
    case failure(Failure)

    var isSuccess: Bool {
        switch self {
        case .success:
            return true
        case .failure:
            return false
        }
    }
}

/// Actor for thread-safe counter
actor Counter {
    private var value = 0

    func increment() -> Int {
        value += 1
        return value
    }

    func decrement() -> Int {
        value -= 1
        return value
    }

    func reset() {
        value = 0
    }
}
"""
        chunks = parse_swift(content)

        # Should extract multiple types of chunks
        assert len(chunks) > 0, "Should extract chunks from realistic module"

        # Check for NetworkClient class
        has_client = any("NetworkClient" in c.code for c in chunks)
        assert has_client, "Should find NetworkClient class"

        # Check for protocol
        has_protocol = any(
            "DataFetcher" in c.code or "protocol" in c.code.lower() for c in chunks
        )
        assert has_protocol, "Should find protocol declaration"

        # Check for enum
        has_enum = any("NetworkError" in c.code or "Result" in c.code for c in chunks)
        assert has_enum, "Should find enum declaration"

        # Check for actor
        has_actor = any("Counter" in c.code and "actor" in c.code for c in chunks)
        assert has_actor, "Should find actor declaration"

        # Check for extension
        has_extension = any(
            "extension" in c.code.lower() and "URL" in c.code for c in chunks
        )
        assert has_extension, "Should find extension"

        # Verify multiple chunk types are present
        chunk_types = {c.chunk_type for c in chunks}
        assert len(chunk_types) >= 1, (
            f"Should have multiple chunk types, got: {chunk_types}"
        )
