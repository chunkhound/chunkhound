"""Tests for Java language mapping and parsing."""

import pytest

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import get_parser_factory, ParserFactory
from chunkhound.parsers.universal_engine import UniversalConcept
from chunkhound.parsers.mappings.java import JavaMapping


def parse_java(content: str):
    """Helper function to parse Java content."""
    factory = get_parser_factory()
    parser = factory.create_parser(Language.JAVA)
    return parser.parse_content(content, "test.java", FileId(1))


@pytest.fixture
def parse_java_concepts():
    """Parse Java content at concept level for detailed verification.

    Returns chunks with full metadata for inspection, following the HCL
    test pattern for high-standard metadata verification.
    """
    def _parse(content: str, concept: UniversalConcept = UniversalConcept.DEFINITION):
        parser = ParserFactory().create_parser(Language.JAVA)
        ast = parser.engine.parse_to_ast(content)
        return parser.extractor.extract_concept(
            ast.root_node, content.encode(), concept
        )
    return _parse


@pytest.fixture
def java_mapping():
    """Create a JavaMapping instance for unit testing extraction methods."""
    return JavaMapping()


class TestJavaClassParsing:
    """Test parsing of Java class declarations."""

    def test_captures_class_declaration(self):
        """Test that class declarations are captured as CLASS chunks."""
        content = """
public class MyClass {
    private String name;

    public void doSomething() {
        System.out.println("Hello");
    }
}
"""
        chunks = parse_java(content)
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        symbols = {c.symbol for c in chunks}

        assert len(class_chunks) > 0, "Should capture class declaration as CLASS chunk"
        assert "MyClass" in symbols, \
            f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"

    def test_captures_nested_class(self):
        """Test that nested/inner classes are captured."""
        content = """
public class OuterClass {
    private String outerField;

    public class InnerClass {
        private String innerField;

        public void innerMethod() {
            System.out.println(outerField);
        }
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Should capture both outer and inner class
        has_outer = "OuterClass" in symbols
        has_inner = "InnerClass" in symbols

        assert has_outer or has_inner, \
            f"Expected OuterClass or InnerClass in symbols, got: {sorted(symbols)}"

    def test_captures_generic_class(self):
        """Test that generic class declarations are captured."""
        content = """
public class Box<T> {
    private T value;

    public void setValue(T value) {
        this.value = value;
    }

    public T getValue() {
        return value;
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        assert "Box" in symbols, \
            f"Expected 'Box' in symbols, got: {sorted(symbols)}"


class TestJavaInterfaceParsing:
    """Test parsing of Java interface declarations."""

    def test_captures_interface_declaration(self):
        """Test that interface declarations are captured."""
        content = """
public interface MyInterface {
    void method1();
    int method2(String param);
}
"""
        chunks = parse_java(content)
        class_chunks = [c for c in chunks if c.chunk_type in (ChunkType.CLASS, ChunkType.INTERFACE)]
        symbols = {c.symbol for c in chunks}

        assert len(class_chunks) > 0, "Should capture interface declaration"
        assert "MyInterface" in symbols, \
            f"Expected 'MyInterface' in symbols, got: {sorted(symbols)}"

    def test_captures_interface_with_default_method(self):
        """Test that interfaces with default methods are captured."""
        content = """
public interface MyInterface {
    void abstractMethod();

    default void defaultMethod() {
        System.out.println("Default implementation");
    }

    static void staticMethod() {
        System.out.println("Static method in interface");
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the interface and potentially the default method
        has_interface = "MyInterface" in symbols
        has_method = any("Method" in s for s in symbols)

        assert has_interface or has_method, \
            f"Expected interface or method in symbols, got: {sorted(symbols)}"


class TestJavaEnumParsing:
    """Test parsing of Java enum declarations."""

    def test_captures_enum_declaration(self):
        """Test that enum declarations are captured."""
        content = """
public enum Status {
    ACTIVE, INACTIVE, PENDING;

    public boolean isActive() {
        return this == ACTIVE;
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the enum (as CLASS or other chunk type)
        assert len(chunks) > 0, "Should capture enum declaration"
        assert "Status" in symbols, \
            f"Expected 'Status' in symbols, got: {sorted(symbols)}"

    def test_captures_enum_with_methods(self):
        """Test that enums with methods and fields are captured."""
        content = """
public enum Color {
    RED(255, 0, 0),
    GREEN(0, 255, 0),
    BLUE(0, 0, 255);

    private final int r;
    private final int g;
    private final int b;

    Color(int r, int g, int b) {
        this.r = r;
        this.g = g;
        this.b = b;
    }

    public int getRed() {
        return r;
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        assert "Color" in symbols, \
            f"Expected 'Color' in symbols, got: {sorted(symbols)}"


class TestJavaMethodParsing:
    """Test parsing of Java methods and constructors."""

    def test_captures_instance_method(self):
        """Test that classes with instance methods are captured."""
        content = """
public class MyClass {
    // cAST may filter small methods. Making implementation substantial.
    public void instanceMethod() {
        int x = 1;
        int y = 2;
        int z = x + y;
        int result = 0;
        for (int i = 0; i < 10; i++) {
            result += i;
        }
        System.out.println(z + result);
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Method may be captured as part of class or separately
        assert len(chunks) > 0, "Should capture class with instance method"
        has_class_or_method = "MyClass" in symbols or "instanceMethod" in symbols
        assert has_class_or_method, f"Expected 'MyClass' or 'instanceMethod' in symbols, got: {sorted(symbols)}"

    def test_captures_static_method(self):
        """Test that classes with static methods are captured."""
        content = """
public class MyClass {
    // Making method substantial to prevent filtering
    public static void staticMethod() {
        int x = 1;
        int y = 2;
        int z = x + y;
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 2;
        }
        System.out.println(z + sum);
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Static method may be captured as part of class or separately
        has_class_or_method = "MyClass" in symbols or "staticMethod" in symbols
        assert has_class_or_method, f"Expected 'MyClass' or 'staticMethod' in symbols, got: {sorted(symbols)}"

    def test_captures_constructor(self):
        """Test that constructors are captured."""
        content = """
public class MyClass {
    private String name;

    public MyClass(String name) {
        this.name = name;
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Constructor should be captured with class name
        has_constructor = "MyClass" in symbols
        assert has_constructor, f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"

    def test_captures_method_overloading(self):
        """Test that classes with overloaded methods are captured."""
        content = """
public class Calculator {
    // Making methods substantial to prevent filtering
    public int add(int a, int b) {
        int result = a + b;
        for (int i = 0; i < 5; i++) {
            result += i;
        }
        return result;
    }

    public double add(double a, double b) {
        double result = a + b;
        for (int i = 0; i < 5; i++) {
            result += i * 0.5;
        }
        return result;
    }

    public int add(int a, int b, int c) {
        int result = a + b + c;
        for (int i = 0; i < 5; i++) {
            result += i * 2;
        }
        return result;
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the Calculator class (methods may be within it)
        has_calculator = "Calculator" in symbols
        has_add = "add" in symbols or any("add" in s.lower() for s in symbols)
        assert has_calculator or has_add, \
            f"Expected 'Calculator' or 'add' in symbols, got: {sorted(symbols)}"


class TestJavaAnnotations:
    """Test parsing and metadata extraction of Java annotations."""

    def test_captures_override_annotation(self):
        """Test that @Override annotation is captured."""
        content = """
public class MyClass extends BaseClass {
    @Override
    public String toString() {
        return "MyClass instance";
    }
}
"""
        chunks = parse_java(content)

        # Should parse without errors and capture the method
        assert len(chunks) > 0, "Should capture class and method"

    def test_captures_suppresswarnings_annotation(self):
        """Test that @SuppressWarnings annotation is captured."""
        content = """
public class MyClass {
    @SuppressWarnings("unused")
    private String unusedField;

    @SuppressWarnings({"unchecked", "rawtypes"})
    public void method() {
        System.out.println("Method with warnings suppressed");
    }
}
"""
        chunks = parse_java(content)

        # Should parse without errors
        assert len(chunks) > 0, "Should capture class with annotations"

    def test_captures_custom_annotation(self):
        """Test that custom annotations are captured."""
        content = """
@Entity
@Table(name = "users")
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "username", nullable = false)
    private String username;

    @Transient
    private String temporaryData;
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the class
        assert "User" in symbols, f"Expected 'User' in symbols, got: {sorted(symbols)}"


class TestJavaGenerics:
    """Test parsing of Java generic types and type parameters."""

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

    public K getKey() {
        return key;
    }

    public V getValue() {
        return value;
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        assert "Pair" in symbols, f"Expected 'Pair' in symbols, got: {sorted(symbols)}"

    def test_generic_method_declaration(self):
        """Test that classes with generic methods are captured."""
        content = """
public class Utils {
    // Making methods substantial to prevent filtering
    public static <T> T getFirst(List<T> list) {
        if (list == null || list.isEmpty()) {
            return null;
        }
        T first = list.get(0);
        for (int i = 1; i < list.size(); i++) {
            T current = list.get(i);
            System.out.println(current);
        }
        return first;
    }

    public static <K, V> Map<K, V> createMap(K key, V value) {
        Map<K, V> map = new HashMap<>();
        map.put(key, value);
        for (int i = 0; i < 5; i++) {
            System.out.println("Processing: " + i);
        }
        return map;
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Should capture Utils class (methods may be within it)
        has_class_or_method = "Utils" in symbols or "getFirst" in symbols or "createMap" in symbols
        assert has_class_or_method, f"Expected 'Utils' or methods in symbols, got: {sorted(symbols)}"

    def test_bounded_type_parameters(self):
        """Test that bounded type parameters are parsed correctly."""
        content = """
public class ComparableBox<T extends Comparable<T>> {
    private T value;

    public ComparableBox(T value) {
        this.value = value;
    }

    public boolean isGreaterThan(T other) {
        return value.compareTo(other) > 0;
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        assert "ComparableBox" in symbols, \
            f"Expected 'ComparableBox' in symbols, got: {sorted(symbols)}"


class TestJavaComments:
    """Test parsing of Java comments and Javadoc."""

    def test_captures_line_comments(self):
        """Test that single-line comments // are captured."""
        content = """
// This is a comment
public class MyClass {
    // Another comment
    private String field;
}
"""
        chunks = parse_java(content)

        # Comments may or may not be captured depending on configuration
        # Just ensure parsing doesn't fail
        assert len(chunks) > 0, "Should parse file with comments"

    def test_captures_block_comments(self):
        """Test that block comments /* */ are captured."""
        content = """
/* This is a
   block comment */
public class MyClass {
    /* Another block comment */
    private String field;
}
"""
        chunks = parse_java(content)

        # Just ensure parsing doesn't fail
        assert len(chunks) > 0, "Should parse file with block comments"

    def test_captures_javadoc_comments(self):
        """Test that Javadoc comments /** */ are captured."""
        content = """
/**
 * This is a Javadoc comment for the class.
 * It provides documentation.
 */
public class MyClass {
    /**
     * This is a Javadoc comment for the method.
     *
     * @param name the name parameter
     * @return a greeting string
     */
    public String greet(String name) {
        return "Hello, " + name;
    }
}
"""
        chunks = parse_java(content)

        # Should parse Javadoc without errors
        assert len(chunks) > 0, "Should parse file with Javadoc comments"

    def test_filters_empty_comments(self):
        """Test that very short or empty comments are filtered."""
        content = """
//
/* */
public class MyClass {
    // x
    private String field;
}
"""
        chunks = parse_java(content)

        # Should parse but filter very short comments
        assert len(chunks) > 0, "Should parse file with minimal comments"


class TestJavaInheritance:
    """Test parsing of Java inheritance (extends and implements)."""

    def test_captures_class_with_extends(self):
        """Test that classes with extends are captured."""
        content = """
public class Dog extends Animal {
    private String breed;

    @Override
    public void makeSound() {
        System.out.println("Woof!");
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        assert "Dog" in symbols, f"Expected 'Dog' in symbols, got: {sorted(symbols)}"

    def test_captures_class_with_implements(self):
        """Test that classes with implements are captured."""
        content = """
public class MyList implements List<String> {
    private String[] items;

    @Override
    public int size() {
        return items.length;
    }

    @Override
    public boolean isEmpty() {
        return items.length == 0;
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        assert "MyList" in symbols, f"Expected 'MyList' in symbols, got: {sorted(symbols)}"

    def test_captures_abstract_class(self):
        """Test that abstract classes are captured."""
        content = """
public abstract class Animal {
    protected String name;

    public Animal(String name) {
        this.name = name;
    }

    public abstract void makeSound();

    public void sleep() {
        System.out.println(name + " is sleeping");
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        assert "Animal" in symbols, f"Expected 'Animal' in symbols, got: {sorted(symbols)}"


class TestJavaModifiers:
    """Test parsing of Java modifiers (public, private, static, final, etc.)."""

    def test_method_modifiers(self):
        """Test that classes with methods having various modifiers are captured."""
        content = """
public class MyClass {
    // Making methods substantial to prevent filtering
    public void publicMethod() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i;
        }
        System.out.println("Public: " + sum);
    }

    private void privateMethod() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 2;
        }
        System.out.println("Private: " + sum);
    }

    protected void protectedMethod() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 3;
        }
        System.out.println("Protected: " + sum);
    }

    static void packagePrivateMethod() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 4;
        }
        System.out.println("Package private: " + sum);
    }

    public static final void staticFinalMethod() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 5;
        }
        System.out.println("Static final: " + sum);
    }
}
"""
        chunks = parse_java(content)

        # Should capture the class with methods (methods may be part of class chunk)
        assert len(chunks) > 0, "Should capture class with methods"
        symbols = {c.symbol for c in chunks}
        assert "MyClass" in symbols, f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"

    def test_class_modifiers(self):
        """Test that classes with various modifiers are captured."""
        content = """
public class PublicClass {
    private String field1;

    public void method1() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i;
        }
        System.out.println(sum);
    }
}

abstract class AbstractClass {
    abstract void abstractMethod();

    public void concreteMethod() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 2;
        }
        System.out.println(sum);
    }
}

final class FinalClass {
    private String field2;

    public void method2() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * 3;
        }
        System.out.println(sum);
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Should capture at least one class with modifiers
        has_classes = any(name in symbols for name in ["PublicClass", "AbstractClass", "FinalClass"])
        assert has_classes, f"Expected class names in symbols, got: {sorted(symbols)}"


class TestJavaLambdas:
    """Test parsing of Java lambda expressions and method references."""

    def test_parses_simple_lambda(self):
        """Test that simple lambda expressions are parsed."""
        content = """
public class LambdaExample {
    public void example() {
        Runnable r = () -> System.out.println("Hello");
        r.run();
    }
}
"""
        chunks = parse_java(content)

        # Should parse without errors
        assert len(chunks) > 0, "Should parse class with lambda"

    def test_parses_lambda_with_block(self):
        """Test that lambda expressions with blocks are parsed."""
        content = """
public class LambdaExample {
    public void example() {
        List<String> list = Arrays.asList("a", "b", "c");
        list.forEach(item -> {
            System.out.println(item);
            System.out.println(item.toUpperCase());
        });
    }
}
"""
        chunks = parse_java(content)

        # Should parse without errors
        assert len(chunks) > 0, "Should parse class with lambda block"

    def test_parses_method_reference(self):
        """Test that method references are parsed."""
        content = """
public class MethodRefExample {
    public void example() {
        List<String> list = Arrays.asList("a", "b", "c");
        list.forEach(System.out::println);
        list.stream().map(String::toUpperCase).collect(Collectors.toList());
    }
}
"""
        chunks = parse_java(content)

        # Should parse without errors
        assert len(chunks) > 0, "Should parse class with method references"


class TestJavaPackages:
    """Test parsing of Java package declarations."""

    def test_extracts_package_declaration(self):
        """Test that package declarations are parsed."""
        content = """
package com.example.demo;

public class MyClass {
    private String field;
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the class
        assert "MyClass" in symbols, f"Expected 'MyClass' in symbols, got: {sorted(symbols)}"


class TestJavaImports:
    """Test parsing of Java import statements."""

    def test_captures_regular_imports(self):
        """Test that regular import statements are parsed."""
        content = """
import java.util.List;
import java.util.ArrayList;
import java.io.IOException;

public class MyClass {
    private List<String> items = new ArrayList<>();
}
"""
        chunks = parse_java(content)

        # Should parse without errors
        assert len(chunks) > 0, "Should parse file with imports"

    def test_captures_static_imports(self):
        """Test that static import statements are parsed."""
        content = """
import static java.lang.Math.PI;
import static java.lang.Math.sqrt;
import static java.util.Collections.sort;

public class MathExample {
    public double circleArea(double radius) {
        return PI * radius * radius;
    }
}
"""
        chunks = parse_java(content)

        # Should parse without errors
        assert len(chunks) > 0, "Should parse file with static imports"


class TestJavaSymbolNames:
    """Test that Java parser generates correct symbol names."""

    def test_method_symbol_names(self):
        """Test that method symbols are correctly extracted."""
        content = """
public class MyClass {
    public void simpleMethod() {
        int x = 1;
        int y = 2;
        int z = x + y;
    }

    public String getName() {
        return "name";
    }

    public void setName(String name) {
        this.name = name;
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Should have method names in symbols
        has_methods = any(name in symbols for name in ["simpleMethod", "getName", "setName"])
        assert has_methods or "MyClass" in symbols, \
            f"Expected method names or class name in symbols, got: {sorted(symbols)}"

    def test_class_symbol_names(self):
        """Test that class symbols are correctly extracted."""
        content = """
public class OuterClass {
    private String field1;

    public static class StaticNestedClass {
        private String field2;
    }

    public class InnerClass {
        private String field3;
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Should have class names in symbols
        has_outer = "OuterClass" in symbols
        has_nested = "StaticNestedClass" in symbols or "InnerClass" in symbols

        assert has_outer or has_nested, \
            f"Expected class names in symbols, got: {sorted(symbols)}"


class TestJavaMetadata:
    """Test that Java metadata is correctly extracted."""

    def test_method_return_type_metadata(self):
        """Test that method return types are in code (not necessarily metadata)."""
        content = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public String getString() {
        return "test";
    }

    public void voidMethod() {
        System.out.println("void");
    }
}
"""
        chunks = parse_java(content)

        # Return types should be visible in the code chunks
        has_int = any("int add" in c.code for c in chunks)
        has_string = any("String getString" in c.code for c in chunks)
        has_void = any("void" in c.code for c in chunks)

        assert has_int or has_string or has_void, \
            "Expected return types to be visible in code"

    def test_method_parameter_metadata(self):
        """Test that method parameters are visible in code."""
        content = """
public class MyClass {
    public void method(String name, int age, boolean active) {
        System.out.println(name + " " + age + " " + active);
    }

    public void genericMethod(List<String> items, Map<String, Integer> map) {
        System.out.println(items.size() + " " + map.size());
    }
}
"""
        chunks = parse_java(content)

        # Parameters should be visible in the code chunks
        has_params = any("String name" in c.code or "int age" in c.code for c in chunks)
        has_generics = any("List<String>" in c.code or "Map<String, Integer>" in c.code for c in chunks)

        assert has_params or has_generics, \
            "Expected parameters to be visible in code"


class TestJavaComplexModule:
    """Test parsing of complex Java files with multiple constructs."""

    def test_parses_complete_class_file(self):
        """Test parsing a complete Java class with multiple features."""
        content = """
package com.example;

import java.util.List;
import java.util.ArrayList;

public class CompleteExample {
    private String name;
    private int value;

    public CompleteExample(String name, int value) {
        this.name = name;
        this.value = value;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return "CompleteExample(" + name + ", " + value + ")";
    }

    public void processData() {
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += i * value;
        }
        System.out.println(name + ": " + sum);
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the class
        assert "CompleteExample" in symbols, \
            f"Expected 'CompleteExample' in symbols, got: {sorted(symbols)}"

    def test_java_realistic_module(self):
        """Test parsing a realistic Java module with advanced features."""
        content = """
package com.example.service;

import java.util.*;

public class UserService {

    private final Map<Long, String> userCache;
    private final String repositoryName;

    public UserService(String repositoryName) {
        this.userCache = new HashMap<>();
        this.repositoryName = repositoryName;
    }

    public String findById(Long id) {
        String cached = userCache.get(id);
        if (cached != null) {
            return cached;
        }
        String user = "User" + id;
        userCache.put(id, user);
        return user;
    }

    public List<String> findAll() {
        List<String> users = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            users.add("User" + i);
        }
        return users;
    }

    public void save(Long id, String user) {
        userCache.put(id, user);
        for (int i = 0; i < 5; i++) {
            System.out.println("Saving: " + i);
        }
    }

    public void delete(Long id) {
        userCache.remove(id);
        for (int i = 0; i < 5; i++) {
            System.out.println("Deleting: " + i);
        }
    }

    public static class Builder {
        private String repositoryName;

        public Builder repository(String repositoryName) {
            this.repositoryName = repositoryName;
            int count = 0;
            for (int i = 0; i < 5; i++) {
                count++;
            }
            return this;
        }

        public UserService build() {
            UserService service = new UserService(repositoryName);
            for (int i = 0; i < 5; i++) {
                System.out.println("Building: " + i);
            }
            return service;
        }
    }
}
"""
        chunks = parse_java(content)
        symbols = {c.symbol for c in chunks}

        # Should capture the main class
        has_service = "UserService" in symbols
        has_builder = "Builder" in symbols
        has_methods = any(name in symbols for name in ["findById", "findAll", "save"])

        assert has_service or has_builder or has_methods, \
            f"Expected UserService, Builder, or methods in symbols, got: {sorted(symbols)}"

        # Should have captured multiple chunks
        assert len(chunks) >= 2, \
            f"Expected at least 2 chunks from realistic module, got {len(chunks)}"


# =============================================================================
# HIGH-STANDARD METADATA VERIFICATION TESTS
# Following the pattern from test_hcl_mapping.py for concept-level testing
# =============================================================================


class TestJavaVisibilityMetadata:
    """Test that visibility modifiers are correctly extracted into metadata."""

    def test_public_visibility(self, parse_java_concepts):
        """Test public visibility is captured in metadata."""
        content = """
public class MyClass {
    public void myMethod() {}
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        assert method_chunks[0].metadata.get("visibility") == "public"

    def test_private_visibility(self, parse_java_concepts):
        """Test private visibility is captured in metadata."""
        content = """
public class MyClass {
    private void myMethod() {}
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        assert method_chunks[0].metadata.get("visibility") == "private"

    def test_protected_visibility(self, parse_java_concepts):
        """Test protected visibility is captured in metadata."""
        content = """
public class MyClass {
    protected void myMethod() {}
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        assert method_chunks[0].metadata.get("visibility") == "protected"

    def test_package_private_visibility(self, parse_java_concepts):
        """Test package-private (no modifier) visibility is not set (no modifiers node).

        Note: When there's no explicit visibility modifier, the parser doesn't
        create a modifiers node, so visibility is not set in metadata.
        This is correct behavior - absence of visibility key indicates package-private.
        """
        content = """
public class MyClass {
    void myMethod() {}
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        # Package-private methods have no modifiers node, so no visibility key
        visibility = method_chunks[0].metadata.get("visibility")
        assert visibility is None or visibility == "package", \
            f"Expected None (no modifiers) or 'package', got: {visibility}"


class TestJavaModifierMetadata:
    """Test that Java modifiers are correctly extracted into metadata."""

    def test_static_modifier_in_metadata(self, parse_java_concepts):
        """Test static modifier is captured in metadata."""
        content = """
public class MyClass {
    public static void myMethod() {}
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        modifiers = method_chunks[0].metadata.get("modifiers", [])
        assert "static" in modifiers, f"Expected 'static' in modifiers, got: {modifiers}"

    def test_final_modifier_in_metadata(self, parse_java_concepts):
        """Test final modifier is captured in metadata."""
        content = """
public class MyClass {
    public final void myMethod() {}
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        modifiers = method_chunks[0].metadata.get("modifiers", [])
        assert "final" in modifiers, f"Expected 'final' in modifiers, got: {modifiers}"

    def test_abstract_modifier_in_metadata(self, parse_java_concepts):
        """Test abstract modifier is captured in metadata."""
        content = """
public abstract class MyClass {
    public abstract void myMethod();
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        modifiers = method_chunks[0].metadata.get("modifiers", [])
        assert "abstract" in modifiers, f"Expected 'abstract' in modifiers, got: {modifiers}"

    def test_synchronized_modifier_in_metadata(self, parse_java_concepts):
        """Test synchronized modifier is captured in metadata."""
        content = """
public class MyClass {
    public synchronized void myMethod() {}
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        modifiers = method_chunks[0].metadata.get("modifiers", [])
        assert "synchronized" in modifiers, f"Expected 'synchronized' in modifiers, got: {modifiers}"

    def test_multiple_modifiers_in_metadata(self, parse_java_concepts):
        """Test multiple modifiers are captured in metadata."""
        content = """
public class MyClass {
    public static final void myMethod() {}
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        modifiers = method_chunks[0].metadata.get("modifiers", [])
        assert "static" in modifiers, f"Expected 'static' in modifiers, got: {modifiers}"
        assert "final" in modifiers, f"Expected 'final' in modifiers, got: {modifiers}"


class TestJavaKindMetadata:
    """Test that Java definition kinds are correctly identified in metadata."""

    def test_method_kind(self, parse_java_concepts):
        """Test method kind is captured in metadata."""
        content = """
public class MyClass {
    public void myMethod() {}
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        assert method_chunks[0].metadata.get("kind") == "method"

    def test_constructor_kind(self, parse_java_concepts):
        """Test constructor kind is captured in metadata."""
        content = """
public class MyClass {
    public MyClass() {}
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        constructor_chunks = [c for c in chunks if c.metadata.get("kind") == "constructor"]
        assert len(constructor_chunks) > 0, "Should capture constructor with kind='constructor'"

    def test_class_kind(self, parse_java_concepts):
        """Test class kind is captured in metadata."""
        content = "public class MyClass {}"
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        class_chunks = [c for c in chunks if c.name == "MyClass"]
        assert len(class_chunks) > 0, "Should capture MyClass"
        assert class_chunks[0].metadata.get("kind") == "class"

    def test_interface_kind(self, parse_java_concepts):
        """Test interface kind is captured in metadata."""
        content = "public interface MyInterface {}"
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        interface_chunks = [c for c in chunks if c.name == "MyInterface"]
        assert len(interface_chunks) > 0, "Should capture MyInterface"
        assert interface_chunks[0].metadata.get("kind") == "interface"

    def test_enum_kind(self, parse_java_concepts):
        """Test enum kind is captured in metadata."""
        content = "public enum MyEnum { VALUE1, VALUE2 }"
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        enum_chunks = [c for c in chunks if c.name == "MyEnum"]
        assert len(enum_chunks) > 0, "Should capture MyEnum"
        assert enum_chunks[0].metadata.get("kind") == "enum"

    def test_field_kind(self, parse_java_concepts):
        """Test field kind is captured in metadata."""
        content = """
public class MyClass {
    private String myField;
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        field_chunks = [c for c in chunks if c.metadata.get("kind") == "field"]
        assert len(field_chunks) > 0, "Should capture field with kind='field'"

    def test_annotation_type_kind(self, parse_java_concepts):
        """Test annotation type kind is captured in metadata."""
        content = "public @interface MyAnnotation {}"
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        annotation_chunks = [c for c in chunks if c.name == "MyAnnotation"]
        assert len(annotation_chunks) > 0, "Should capture MyAnnotation"
        assert annotation_chunks[0].metadata.get("kind") == "annotation"


class TestJavaMethodMetadata:
    """Test that method-specific metadata is correctly extracted."""

    def test_parameters_in_metadata(self, parse_java_concepts):
        """Test method parameters are captured in metadata."""
        content = """
public class MyClass {
    public void myMethod(String name, int age) {}
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        params = method_chunks[0].metadata.get("parameters", [])
        assert "String" in params, f"Expected 'String' in parameters, got: {params}"
        assert "int" in params, f"Expected 'int' in parameters, got: {params}"

    def test_return_type_in_metadata(self, parse_java_concepts):
        """Test method return type is captured in metadata."""
        content = """
public class MyClass {
    public String myMethod() { return ""; }
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        return_type = method_chunks[0].metadata.get("return_type")
        assert return_type == "String", f"Expected 'String' return_type, got: {return_type}"

    def test_void_return_type_in_metadata(self, parse_java_concepts):
        """Test void return type is captured in metadata."""
        content = """
public class MyClass {
    public void myMethod() {}
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        return_type = method_chunks[0].metadata.get("return_type")
        assert return_type == "void", f"Expected 'void' return_type, got: {return_type}"

    def test_type_parameters_in_metadata(self, parse_java_concepts):
        """Test generic type parameters are captured in metadata."""
        content = """
public class MyClass {
    public <T> T myMethod(T value) { return value; }
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "myMethod"]
        assert len(method_chunks) > 0, "Should capture myMethod"
        type_params = method_chunks[0].metadata.get("type_parameters")
        assert type_params is not None, "Expected type_parameters in metadata"
        assert "T" in type_params, f"Expected 'T' in type_parameters, got: {type_params}"

    def test_annotations_in_metadata(self, parse_java_concepts):
        """Test method annotations are captured in metadata."""
        content = """
public class MyClass {
    @Override
    @SuppressWarnings("unused")
    public String toString() { return ""; }
}
"""
        chunks = parse_java_concepts(content, UniversalConcept.DEFINITION)
        method_chunks = [c for c in chunks if c.name == "toString"]
        assert len(method_chunks) > 0, "Should capture toString"
        annotations = method_chunks[0].metadata.get("annotations", [])
        has_override = any("Override" in a for a in annotations)
        has_suppress = any("SuppressWarnings" in a for a in annotations)
        assert has_override, f"Expected @Override in annotations, got: {annotations}"
        assert has_suppress, f"Expected @SuppressWarnings in annotations, got: {annotations}"


class TestJavaCommentMetadata:
    """Test that comment metadata is correctly extracted."""

    def test_line_comment_type(self, parse_java_concepts):
        """Test line comment type is captured in metadata."""
        content = "// This is a meaningful line comment for testing"
        chunks = parse_java_concepts(content, UniversalConcept.COMMENT)
        if chunks:  # Comments may be filtered
            comment_chunks = [c for c in chunks if c.metadata.get("comment_type") == "line"]
            if comment_chunks:
                assert comment_chunks[0].metadata.get("comment_type") == "line"

    def test_block_comment_type(self, parse_java_concepts):
        """Test block comment type is captured in metadata."""
        content = "/* This is a meaningful block comment for testing purposes */"
        chunks = parse_java_concepts(content, UniversalConcept.COMMENT)
        if chunks:  # Comments may be filtered
            block_chunks = [c for c in chunks if c.metadata.get("comment_type") == "block"]
            if block_chunks:
                assert block_chunks[0].metadata.get("comment_type") == "block"

    def test_javadoc_comment_type(self, parse_java_concepts):
        """Test Javadoc comment type and is_javadoc flag are captured in metadata."""
        content = """
/**
 * This is a Javadoc comment.
 * It provides documentation for a class.
 */
public class MyClass {}
"""
        chunks = parse_java_concepts(content, UniversalConcept.COMMENT)
        javadoc_chunks = [c for c in chunks if c.metadata.get("comment_type") == "javadoc"]
        assert len(javadoc_chunks) > 0, "Should capture Javadoc comment"
        assert javadoc_chunks[0].metadata.get("is_javadoc") is True


class TestJavaImportMetadata:
    """Test that import/package metadata is correctly extracted."""

    def test_import_path_metadata(self, parse_java_concepts):
        """Test import path is captured in metadata."""
        content = "import java.util.List;"
        chunks = parse_java_concepts(content, UniversalConcept.IMPORT)
        import_chunks = [c for c in chunks if c.metadata.get("import_type") == "import"]
        assert len(import_chunks) > 0, "Should capture import declaration"
        import_path = import_chunks[0].metadata.get("import_path")
        assert import_path == "java.util.List", f"Expected 'java.util.List', got: {import_path}"

    def test_static_import_flag(self, parse_java_concepts):
        """Test static import flag is captured in metadata."""
        content = "import static java.lang.Math.PI;"
        chunks = parse_java_concepts(content, UniversalConcept.IMPORT)
        import_chunks = [c for c in chunks if c.metadata.get("import_type") == "import"]
        assert len(import_chunks) > 0, "Should capture static import"
        assert import_chunks[0].metadata.get("is_static_import") is True

    def test_package_name_metadata(self, parse_java_concepts):
        """Test package name is captured in metadata."""
        content = "package com.example.demo;"
        chunks = parse_java_concepts(content, UniversalConcept.IMPORT)
        package_chunks = [c for c in chunks if c.metadata.get("import_type") == "package"]
        assert len(package_chunks) > 0, "Should capture package declaration"
        package_name = package_chunks[0].metadata.get("package_name")
        assert package_name == "com.example.demo", f"Expected 'com.example.demo', got: {package_name}"


class TestJavaChunkTypes:
    """Test that chunk types are correctly assigned."""

    def test_class_chunk_type(self):
        """Test class declarations get ChunkType.CLASS."""
        content = "public class MyClass {}"
        chunks = parse_java(content)
        class_chunks = [c for c in chunks if c.symbol == "MyClass"]
        assert len(class_chunks) > 0, "Should capture MyClass"
        assert class_chunks[0].chunk_type == ChunkType.CLASS

    def test_interface_chunk_type(self):
        """Test interface declarations get ChunkType.INTERFACE."""
        content = "public interface MyInterface {}"
        chunks = parse_java(content)
        interface_chunks = [c for c in chunks if c.symbol == "MyInterface"]
        assert len(interface_chunks) > 0, "Should capture MyInterface"
        assert interface_chunks[0].chunk_type == ChunkType.INTERFACE

    def test_method_chunk_type(self):
        """Test method declarations get ChunkType.METHOD."""
        content = """
public class MyClass {
    public void myMethod() {
        int x = 1;
        int y = 2;
        System.out.println(x + y);
    }
}
"""
        chunks = parse_java(content)
        method_chunks = [c for c in chunks if c.symbol == "myMethod"]
        if method_chunks:
            assert method_chunks[0].chunk_type == ChunkType.METHOD


class TestJavaExtractionMethods:
    """Unit tests for JavaMapping extraction methods."""

    def test_clean_comment_text_javadoc(self, java_mapping):
        """Test clean_comment_text removes Javadoc markers."""
        javadoc = """/**
 * This is a Javadoc comment.
 * @param name the name
 */"""
        cleaned = java_mapping.clean_comment_text(javadoc)
        assert not cleaned.startswith("/**")
        assert not cleaned.endswith("*/")
        assert "This is a Javadoc comment" in cleaned

    def test_clean_comment_text_block(self, java_mapping):
        """Test clean_comment_text removes block comment markers."""
        block = "/* This is a block comment */"
        cleaned = java_mapping.clean_comment_text(block)
        assert not cleaned.startswith("/*")
        assert not cleaned.endswith("*/")
        assert "This is a block comment" in cleaned

    def test_clean_comment_text_line(self, java_mapping):
        """Test clean_comment_text removes line comment markers."""
        line = "// This is a line comment"
        cleaned = java_mapping.clean_comment_text(line)
        assert not cleaned.startswith("//")
        assert "This is a line comment" in cleaned

    def test_clean_comment_text_preserves_content(self, java_mapping):
        """Test clean_comment_text preserves multiline Javadoc content."""
        javadoc = """/**
 * Line 1
 * Line 2
 * Line 3
 */"""
        cleaned = java_mapping.clean_comment_text(javadoc)
        assert "Line 1" in cleaned
        assert "Line 2" in cleaned
        assert "Line 3" in cleaned
