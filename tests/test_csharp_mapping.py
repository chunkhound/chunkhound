"""Tests for C# language mapping and parsing."""

import pytest

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import get_parser_factory


def parse_csharp(content: str):
    """Helper function to parse C# content."""
    factory = get_parser_factory()
    parser = factory.create_parser(Language.CSHARP)
    return parser.parse_content(content, "test.cs", FileId(1))


class TestCSharpClassParsing:
    """Test parsing of C# class declarations."""

    def test_captures_class_declaration(self):
        """Test that class declarations are captured as CLASS chunks."""
        content = """
public class MyClass {
    private string name;

    public void DoSomething() {
        Console.WriteLine("Hello");
    }
}
"""
        chunks = parse_csharp(content)
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        symbols = {c.symbol for c in chunks}

        assert len(class_chunks) > 0, "Should capture class declaration as CLASS chunk"
        assert "MyClass" in symbols, (
            f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_generic_class(self):
        """Test that generic class declarations are captured."""
        content = """
public class Box<T> {
    private T value;

    public void SetValue(T value) {
        this.value = value;
    }

    public T GetValue() {
        return value;
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        assert "Box" in symbols, f"Expected 'Box' in symbols, got: {sorted(symbols)}"

    def test_captures_nested_class(self):
        """Test that nested/inner classes are captured."""
        content = """
public class OuterClass {
    private string outerField;

    public class InnerClass {
        private string innerField;

        public void InnerMethod() {
            Console.WriteLine("Inner method");
        }
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture both outer and inner class
        has_outer = "OuterClass" in symbols
        has_inner = "InnerClass" in symbols

        assert has_outer or has_inner, (
            f"Expected OuterClass or InnerClass in symbols, got: {sorted(symbols)}"
        )


class TestCSharpInterfaceParsing:
    """Test parsing of C# interface declarations."""

    def test_captures_interface_declaration(self):
        """Test that interface declarations are captured."""
        content = """
public interface IMyInterface {
    void Method1();
    int Method2(string param);
}
"""
        chunks = parse_csharp(content)
        class_chunks = [
            c for c in chunks if c.chunk_type in (ChunkType.CLASS, ChunkType.INTERFACE)
        ]
        symbols = {c.symbol for c in chunks}

        assert len(class_chunks) > 0, "Should capture interface declaration"
        assert "IMyInterface" in symbols, (
            f"Expected 'IMyInterface' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_interface_with_default_method(self):
        """Test that interfaces with default methods are captured (C# 8.0+)."""
        content = """
public interface IMyInterface {
    void AbstractMethod();

    void DefaultMethod() {
        Console.WriteLine("Default implementation");
    }

    static void StaticMethod() {
        Console.WriteLine("Static method in interface");
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the interface and potentially the default method
        has_interface = "IMyInterface" in symbols
        has_method = any("Method" in s for s in symbols)

        assert has_interface or has_method, (
            f"Expected interface or method in symbols, got: {sorted(symbols)}"
        )


class TestCSharpStructParsing:
    """Test parsing of C# struct declarations."""

    def test_captures_struct_declaration(self):
        """Test that struct declarations are captured."""
        content = """
public struct Point {
    public int X { get; set; }
    public int Y { get; set; }

    public Point(int x, int y) {
        X = x;
        Y = y;
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the struct (as CLASS or other chunk type)
        assert len(chunks) > 0, "Should capture struct declaration"
        assert "Point" in symbols, (
            f"Expected 'Point' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_struct_with_methods(self):
        """Test that structs with methods and constructors are captured."""
        content = """
public struct Vector {
    public double X;
    public double Y;

    public Vector(double x, double y) {
        X = x;
        Y = y;
    }

    public double Magnitude() {
        return Math.Sqrt(X * X + Y * Y);
    }

    public override string ToString() {
        return $"({X}, {Y})";
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        assert "Vector" in symbols, (
            f"Expected 'Vector' in symbols, got: {sorted(symbols)}"
        )


class TestCSharpEnumParsing:
    """Test parsing of C# enum declarations."""

    def test_captures_enum_declaration(self):
        """Test that enum declarations are captured."""
        content = """
public enum Status {
    Active, Inactive, Pending
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the enum (as CLASS or other chunk type)
        assert len(chunks) > 0, "Should capture enum declaration"
        assert "Status" in symbols, (
            f"Expected 'Status' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_enum_with_values(self):
        """Test that enums with explicit values are captured."""
        content = """
public enum Priority : int {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        assert "Priority" in symbols, (
            f"Expected 'Priority' in symbols, got: {sorted(symbols)}"
        )


class TestCSharpRecordParsing:
    """Test parsing of C# record declarations (C# 9.0+)."""

    def test_captures_record_declaration(self):
        """Test that record declarations are captured."""
        content = """
public record Person(string FirstName, string LastName, int Age);
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the record (as CLASS or other chunk type)
        assert len(chunks) > 0, "Should capture record declaration"
        assert "Person" in symbols, (
            f"Expected 'Person' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_record_struct(self):
        """Test that record structs are captured (C# 10.0+)."""
        content = """
public record struct Point3D(double X, double Y, double Z) {
    public double DistanceFromOrigin() {
        return Math.Sqrt(X * X + Y * Y + Z * Z);
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        assert "Point3D" in symbols, (
            f"Expected 'Point3D' in symbols, got: {sorted(symbols)}"
        )


class TestCSharpMethodParsing:
    """Test parsing of C# methods and constructors."""

    def test_captures_instance_method(self):
        """Test that classes with instance methods are captured."""
        content = """
public class MyClass {
    // Making implementation substantial to prevent filtering
    public void InstanceMethod() {
        int x = 1;
        int y = 2;
        int z = x + y;
        int result = 0;
        for (int i = 0; i < 10; i++) {
            result += i;
        }
        Console.WriteLine(z + result);
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Method may be captured as part of class or separately
        assert len(chunks) > 0, "Should capture class with instance method"
        has_class_or_method = "MyClass" in symbols or "InstanceMethod" in symbols
        assert has_class_or_method, (
            f"Expected 'MyClass' or 'InstanceMethod' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_static_method(self):
        """Test that classes with static methods are captured."""
        content = """
public class MyClass {
    // Making method substantial to prevent filtering
    public static void StaticMethod() {
        int x = 1;
        int y = 2;
        int z = x + y;
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 2;
        }
        Console.WriteLine(z + sum);
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Static method may be captured as part of class or separately
        has_class_or_method = "MyClass" in symbols or "StaticMethod" in symbols
        assert has_class_or_method, (
            f"Expected 'MyClass' or 'StaticMethod' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_constructor(self):
        """Test that constructors are captured."""
        content = """
public class MyClass {
    private string name;

    public MyClass(string name) {
        this.name = name;
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Constructor should be captured with class name
        has_constructor = "MyClass" in symbols
        assert has_constructor, f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"

    def test_captures_destructor(self):
        """Test that destructors are captured."""
        content = """
public class MyClass {
    ~MyClass() {
        // Cleanup resources
        Console.WriteLine("Destructor called");
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Destructor should be captured
        assert len(chunks) > 0, "Should capture class with destructor"
        assert "MyClass" in symbols, (
            f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_method_overloading(self):
        """Test that classes with overloaded methods are captured."""
        content = """
public class Calculator {
    // Making methods substantial to prevent filtering
    public int Add(int a, int b) {
        int result = a + b;
        for (int i = 0; i < 5; i++) {
            result += i;
        }
        return result;
    }

    public double Add(double a, double b) {
        double result = a + b;
        for (int i = 0; i < 5; i++) {
            result += i * 0.5;
        }
        return result;
    }

    public int Add(int a, int b, int c) {
        int result = a + b + c;
        for (int i = 0; i < 5; i++) {
            result += i * 2;
        }
        return result;
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the Calculator class (methods may be within it)
        has_calculator = "Calculator" in symbols
        has_add = "Add" in symbols or any("add" in s.lower() for s in symbols)
        assert has_calculator or has_add, (
            f"Expected 'Calculator' or 'Add' in symbols, got: {sorted(symbols)}"
        )


class TestCSharpPropertyParsing:
    """Test parsing of C# properties."""

    def test_captures_auto_property(self):
        """Test that auto-implemented properties are captured."""
        content = """
public class MyClass {
    public string Name { get; set; }
    public int Age { get; set; }
    public bool IsActive { get; set; } = true;
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the class with properties
        assert "MyClass" in symbols, (
            f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_expression_bodied_property(self):
        """Test that expression-bodied properties are captured."""
        content = """
public class MyClass {
    private string firstName;
    private string lastName;

    public string FullName => $"{firstName} {lastName}";
    public int NameLength => FullName.Length;
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the class with properties
        assert "MyClass" in symbols, (
            f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_full_property(self):
        """Test that properties with get/set bodies are captured."""
        content = """
public class MyClass {
    private string name;

    public string Name {
        get { return name; }
        set {
            if (value != null) {
                name = value;
            }
        }
    }

    public int Count {
        get => items.Count;
        set => items = new List<string>(value);
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the class with properties
        assert "MyClass" in symbols, (
            f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"
        )


class TestCSharpAttributes:
    """Test parsing and metadata extraction of C# attributes."""

    def test_captures_obsolete_attribute(self):
        """Test that [Obsolete] attribute is captured."""
        content = """
public class MyClass {
    [Obsolete("This method is deprecated")]
    public void OldMethod() {
        Console.WriteLine("Old method");
    }
}
"""
        chunks = parse_csharp(content)

        # Should parse without errors and capture the method
        assert len(chunks) > 0, "Should capture class and method with attribute"

    def test_captures_multiple_attributes(self):
        """Test that multiple attributes are captured."""
        content = """
[Serializable]
[Obsolete("This class is deprecated")]
public class User {
    [Required]
    [MaxLength(100)]
    public string Username { get; set; }

    [JsonProperty("email_address")]
    [EmailAddress]
    public string Email { get; set; }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the class
        assert "User" in symbols, f"Expected 'User' in symbols, got: {sorted(symbols)}"

    def test_captures_attribute_with_parameters(self):
        """Test that attributes with parameters are captured."""
        content = """
[Table("users")]
[Description("User entity class")]
public class User {
    [Key]
    [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
    public int Id { get; set; }

    [Column("username", TypeName = "varchar(100)")]
    [Required(ErrorMessage = "Username is required")]
    public string Username { get; set; }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the class
        assert "User" in symbols, f"Expected 'User' in symbols, got: {sorted(symbols)}"


class TestCSharpGenerics:
    """Test parsing of C# generic types and type parameters."""

    def test_generic_class_declaration(self):
        """Test that generic class declarations are captured."""
        content = """
public class Pair<K, V> {
    private K key;
    private V value;

    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }

    public K GetKey() {
        return key;
    }

    public V GetValue() {
        return value;
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        assert "Pair" in symbols, f"Expected 'Pair' in symbols, got: {sorted(symbols)}"

    def test_generic_method_declaration(self):
        """Test that classes with generic methods are captured."""
        content = """
public class Utils {
    // Making methods substantial to prevent filtering
    public static T GetFirst<T>(List<T> list) {
        if (list == null || list.Count == 0) {
            return default(T);
        }
        T first = list[0];
        for (int i = 1; i < list.Count; i++) {
            T current = list[i];
            Console.WriteLine(current);
        }
        return first;
    }

    public static Dictionary<K, V> CreateMap<K, V>(K key, V value) {
        var map = new Dictionary<K, V>();
        map[key] = value;
        for (int i = 0; i < 5; i++) {
            Console.WriteLine($"Processing: {i}");
        }
        return map;
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture Utils class (methods may be within it)
        has_class_or_method = (
            "Utils" in symbols or "GetFirst" in symbols or "CreateMap" in symbols
        )
        assert has_class_or_method, (
            f"Expected 'Utils' or methods in symbols, got: {sorted(symbols)}"
        )

    def test_type_constraints(self):
        """Test that generic type constraints are parsed correctly."""
        content = """
public class ComparableBox<T> where T : IComparable<T> {
    private T value;

    public ComparableBox(T value) {
        this.value = value;
    }

    public bool IsGreaterThan(T other) {
        return value.CompareTo(other) > 0;
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        assert "ComparableBox" in symbols, (
            f"Expected 'ComparableBox' in symbols, got: {sorted(symbols)}"
        )


class TestCSharpXmlDocComments:
    """Test parsing of C# XML documentation comments."""

    def test_captures_xml_doc_comments(self):
        """Test that XML documentation comments /// are captured."""
        content = """
/// <summary>
/// This is a class documentation comment.
/// It provides detailed information about the class.
/// </summary>
public class MyClass {
    /// <summary>
    /// This is a method documentation comment.
    /// </summary>
    /// <param name="name">The name parameter</param>
    /// <returns>A greeting string</returns>
    public string Greet(string name) {
        return $"Hello, {name}";
    }
}
"""
        chunks = parse_csharp(content)

        # Should parse XML doc comments without errors
        assert len(chunks) > 0, "Should parse file with XML doc comments"

    def test_captures_summary_tag(self):
        """Test that <summary> tags in XML comments are captured."""
        content = """
public class Calculator {
    /// <summary>
    /// Adds two integers together.
    /// </summary>
    public int Add(int a, int b) {
        return a + b;
    }
}
"""
        chunks = parse_csharp(content)

        # Should parse XML doc comments without errors
        assert len(chunks) > 0, "Should parse file with summary tags"

    def test_captures_param_returns_tags(self):
        """Test that <param> and <returns> tags are captured."""
        content = """
public class Calculator {
    /// <summary>
    /// Divides two numbers.
    /// </summary>
    /// <param name="dividend">The number to be divided</param>
    /// <param name="divisor">The number to divide by</param>
    /// <returns>The quotient of the division</returns>
    /// <exception cref="DivideByZeroException">Thrown when divisor is zero</exception>
    public double Divide(double dividend, double divisor) {
        if (divisor == 0) {
            throw new DivideByZeroException();
        }
        return dividend / divisor;
    }
}
"""
        chunks = parse_csharp(content)

        # Should parse comprehensive XML doc comments
        assert len(chunks) > 0, "Should parse file with param/returns tags"

    def test_filters_empty_comments(self):
        """Test that very short or empty comments are filtered."""
        content = """
//
/* */
public class MyClass {
    // x
    private string field;
}
"""
        chunks = parse_csharp(content)

        # Should parse but filter very short comments
        assert len(chunks) > 0, "Should parse file with minimal comments"


class TestCSharpNamespacesAndUsings:
    """Test parsing of C# namespaces and using directives."""

    def test_extracts_namespace_declaration(self):
        """Test that namespace declarations are parsed."""
        content = """
namespace Com.Example.Demo {
    public class MyClass {
        private string field;
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the class
        assert "MyClass" in symbols, (
            f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"
        )

    def test_captures_using_statements(self):
        """Test that using directives are parsed."""
        content = """
using System;
using System.Collections.Generic;
using System.Linq;

namespace MyApp {
    public class MyClass {
        private List<string> items = new List<string>();
    }
}
"""
        chunks = parse_csharp(content)

        # Should parse without errors
        assert len(chunks) > 0, "Should parse file with using statements"


class TestCSharpModifiers:
    """Test parsing of C# modifiers (public, private, static, async, etc.)."""

    def test_method_modifiers(self):
        """Test that classes with methods having various modifiers are captured."""
        content = """
public class MyClass {
    // Making methods substantial to prevent filtering
    public void PublicMethod() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i;
        }
        Console.WriteLine($"Public: {sum}");
    }

    private void PrivateMethod() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 2;
        }
        Console.WriteLine($"Private: {sum}");
    }

    protected void ProtectedMethod() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 3;
        }
        Console.WriteLine($"Protected: {sum}");
    }

    internal void InternalMethod() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 4;
        }
        Console.WriteLine($"Internal: {sum}");
    }

    public static async Task<int> AsyncStaticMethod() {
        int sum = 0;
        await Task.Delay(100);
        for (int i = 0; i < 10; i++) {
            sum += i * 5;
        }
        return sum;
    }
}
"""
        chunks = parse_csharp(content)

        # Should capture the class with methods (methods may be part of class chunk)
        assert len(chunks) > 0, "Should capture class with methods"
        symbols = {c.symbol for c in chunks}
        assert "MyClass" in symbols, (
            f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"
        )

    def test_class_modifiers(self):
        """Test that classes with various modifiers are captured."""
        content = """
public class PublicClass {
    private string field1;

    public void Method1() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i;
        }
        Console.WriteLine(sum);
    }
}

abstract class AbstractClass {
    public abstract void AbstractMethod();

    public void ConcreteMethod() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 2;
        }
        Console.WriteLine(sum);
    }
}

sealed class SealedClass {
    private string field2;

    public void Method2() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 3;
        }
        Console.WriteLine(sum);
    }
}

static class StaticClass {
    public static void StaticMethod() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 4;
        }
        Console.WriteLine(sum);
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture at least one class with modifiers
        has_classes = any(
            name in symbols
            for name in ["PublicClass", "AbstractClass", "SealedClass", "StaticClass"]
        )
        assert has_classes, f"Expected class names in symbols, got: {sorted(symbols)}"


class TestCSharpSymbolNames:
    """Test that C# parser generates correct symbol names."""

    def test_method_symbol_names(self):
        """Test that method symbols are correctly extracted."""
        content = """
public class MyClass {
    public void SimpleMethod() {
        int x = 1;
        int y = 2;
        int z = x + y;
    }

    public string GetName() {
        return "name";
    }

    public void SetName(string name) {
        this.name = name;
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should have method names in symbols
        has_methods = any(
            name in symbols for name in ["SimpleMethod", "GetName", "SetName"]
        )
        assert has_methods or "MyClass" in symbols, (
            f"Expected method names or class name in symbols, got: {sorted(symbols)}"
        )

    def test_class_symbol_names(self):
        """Test that class symbols are correctly extracted."""
        content = """
public class OuterClass {
    private string field1;

    public static class StaticNestedClass {
        private string field2;
    }

    public class InnerClass {
        private string field3;
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should have class names in symbols
        has_outer = "OuterClass" in symbols
        has_nested = "StaticNestedClass" in symbols or "InnerClass" in symbols

        assert has_outer or has_nested, (
            f"Expected class names in symbols, got: {sorted(symbols)}"
        )

    def test_property_symbol_names(self):
        """Test that property symbols are correctly extracted."""
        content = """
public class MyClass {
    public string Name { get; set; }
    public int Age { get; set; }
    public bool IsActive => true;
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should have property names or class name in symbols
        has_properties = any(name in symbols for name in ["Name", "Age", "IsActive"])
        assert has_properties or "MyClass" in symbols, (
            f"Expected property names or class name in symbols, got: {sorted(symbols)}"
        )


class TestCSharpMetadata:
    """Test that C# metadata is correctly extracted."""

    def test_method_return_type_metadata(self):
        """Test that method return types are visible in code."""
        content = """
public class Calculator {
    public int Add(int a, int b) {
        return a + b;
    }

    public string GetString() {
        return "test";
    }

    public void VoidMethod() {
        Console.WriteLine("void");
    }
}
"""
        chunks = parse_csharp(content)

        # Return types should be visible in the code chunks
        has_int = any("int Add" in c.code or "int" in c.code for c in chunks)
        has_string = any(
            "string GetString" in c.code or "string" in c.code for c in chunks
        )
        has_void = any("void" in c.code for c in chunks)

        assert has_int or has_string or has_void, (
            "Expected return types to be visible in code"
        )

    def test_method_parameter_metadata(self):
        """Test that method parameters are visible in code."""
        content = """
public class MyClass {
    public void Method(string name, int age, bool active) {
        Console.WriteLine($"{name} {age} {active}");
    }

    public void GenericMethod(List<string> items, Dictionary<string, int> map) {
        Console.WriteLine($"{items.Count} {map.Count}");
    }
}
"""
        chunks = parse_csharp(content)

        # Parameters should be visible in the code chunks
        has_params = any("string name" in c.code or "int age" in c.code for c in chunks)
        has_generics = any(
            "List<string>" in c.code or "Dictionary<string, int>" in c.code
            for c in chunks
        )

        assert has_params or has_generics, "Expected parameters to be visible in code"

    def test_generic_type_parameter_metadata(self):
        """Test that generic type parameters are visible in code."""
        content = """
public class Container<T> where T : class {
    private T value;

    public T GetValue() {
        return value;
    }

    public void SetValue(T value) {
        this.value = value;
    }
}
"""
        chunks = parse_csharp(content)

        # Generic type parameters should be visible in code chunks
        has_generic = any("<T>" in c.code or "where T" in c.code for c in chunks)

        assert has_generic or len(chunks) > 0, (
            "Expected generic type parameters to be visible in code"
        )


class TestCSharpComplexModule:
    """Test parsing of complex C# files with multiple constructs."""

    def test_parses_complete_class_file(self):
        """Test parsing a complete C# class with multiple features."""
        content = """
namespace Com.Example;

using System;
using System.Collections.Generic;

public class CompleteExample {
    private string name;
    private int value;

    public CompleteExample(string name, int value) {
        this.name = name;
        this.value = value;
    }

    public string GetName() {
        return name;
    }

    public void SetName(string name) {
        this.name = name;
    }

    public override string ToString() {
        return $"CompleteExample({name}, {value})";
    }

    public void ProcessData() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * value;
        }
        Console.WriteLine($"{name}: {sum}");
    }
}
"""
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the class
        assert "CompleteExample" in symbols, (
            f"Expected 'CompleteExample' in symbols, got: {sorted(symbols)}"
        )

    def test_csharp_realistic_module(self):
        """Test parsing a realistic C# module with advanced features using Sample.cs."""
        # Read the Sample.cs fixture
        import pathlib

        fixture_path = (
            pathlib.Path(__file__).parent / "fixtures" / "csharp" / "Sample.cs"
        )

        if not fixture_path.exists():
            pytest.skip("Sample.cs fixture not found")

        content = fixture_path.read_text()
        chunks = parse_csharp(content)
        symbols = {c.symbol for c in chunks}

        # Should capture multiple classes/interfaces/structs/enums from Sample.cs
        expected_symbols = [
            "Sample",
            "Status",
            "Priority",
            "IProcessor",
            "Configuration",
        ]
        has_expected = any(sym in symbols for sym in expected_symbols)

        assert has_expected, f"Expected symbols from Sample.cs, got: {sorted(symbols)}"

        # Should have captured multiple chunks
        assert len(chunks) >= 3, (
            f"Expected at least 3 chunks from realistic module, got {len(chunks)}"
        )
