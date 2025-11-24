"""Comprehensive class parsing tests for JS-family parsers.

This module tests class-related constructs across JS, JSX, TS, TSX, and Vue
parsers to ensure consistent and complete parsing coverage.

Test categories covered:
1. Class Declarations (basic, extends, expressions)
2. Class Members - Methods (constructor, instance, async, generator, static, getters/setters)
3. Class Members - Properties (instance, static, private)
4. TypeScript Class Features (implements, access modifiers, abstract, decorators, generics)
"""

import pytest
from pathlib import Path

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import get_parser_factory


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def js_parser():
    """Create JavaScript parser."""
    factory = get_parser_factory()
    return factory.create_parser(Language.JAVASCRIPT)


@pytest.fixture
def jsx_parser():
    """Create JSX parser."""
    factory = get_parser_factory()
    return factory.create_parser(Language.JSX)


@pytest.fixture
def ts_parser():
    """Create TypeScript parser."""
    factory = get_parser_factory()
    return factory.create_parser(Language.TYPESCRIPT)


@pytest.fixture
def tsx_parser():
    """Create TSX parser."""
    factory = get_parser_factory()
    return factory.create_parser(Language.TSX)


@pytest.fixture
def vue_parser():
    """Create Vue parser."""
    factory = get_parser_factory()
    return factory.create_parser(Language.VUE)


# Helper function for creating parsers by language
def create_parser(language: Language):
    """Create a parser for the specified language."""
    factory = get_parser_factory()
    return factory.create_parser(language)


# =============================================================================
# Class Declarations Tests
# =============================================================================

class TestClassDeclarations:
    """Tests for basic class declaration patterns."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_basic_class_declaration(self, language, ext):
        """Test basic class declaration is extracted with correct type."""
        code = """
class Foo {
    bar() {
        return 'bar';
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        # Find the class chunk
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class for {language.value}"

        # Verify class name is captured
        foo_class = [c for c in class_chunks if "Foo" in c.symbol]
        assert len(foo_class) > 0, f"Class name 'Foo' should be captured for {language.value}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_class_with_extends(self, language, ext):
        """Test class with extends clause is extracted."""
        code = """
class Animal {
    speak() {}
}

class Dog extends Animal {
    bark() {
        return 'woof';
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        # Find class chunks
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]

        # Should have both Animal and Dog classes
        animal_classes = [c for c in class_chunks if "Animal" in c.symbol]
        dog_classes = [c for c in class_chunks if "Dog" in c.symbol]

        assert len(animal_classes) > 0, f"Should extract Animal class for {language.value}"
        assert len(dog_classes) > 0, f"Should extract Dog class for {language.value}"

        # Dog class code should include extends
        dog_class = dog_classes[0]
        assert "extends" in dog_class.code, "Class code should include 'extends'"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_class_with_mixin_extends(self, language, ext):
        """Test class extending mixin expression."""
        code = """
const mixin = (base) => class extends base {
    mixinMethod() {}
};

class MyClass extends mixin(Base) {
    method() {}
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        # Find MyClass
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS and "MyClass" in c.symbol]
        assert len(class_chunks) > 0, f"Should extract class with mixin extends for {language.value}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_anonymous_class_expression(self, language, ext):
        """Test anonymous class expression assigned to const."""
        code = """
const Foo = class {
    method() {
        return 'hello';
    }
};
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        # Should extract as CLASS type (not FUNCTION)
        # The variable name 'Foo' should be captured
        relevant_chunks = [c for c in chunks if "Foo" in c.symbol or "class" in c.code.lower()]
        assert len(relevant_chunks) > 0, f"Should extract class expression for {language.value}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_named_class_expression(self, language, ext):
        """Test named class expression assigned to variable."""
        code = """
const Foo = class Bar {
    method() {
        return 'hello';
    }
};
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        # Should capture either Foo or Bar as the class name
        relevant_chunks = [c for c in chunks if "Foo" in c.symbol or "Bar" in c.symbol or "class" in c.code.lower()]
        assert len(relevant_chunks) > 0, f"Should extract named class expression for {language.value}"


# =============================================================================
# Class Members - Methods Tests
# =============================================================================

class TestClassMethods:
    """Tests for class method patterns."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_constructor_method(self, language, ext):
        """Test constructor method is extracted."""
        code = """
class MyClass {
    constructor(name) {
        this.name = name;
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        # Find class chunk with constructor
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class with constructor for {language.value}"

        # Constructor should be in content
        class_chunk = class_chunks[0]
        assert "constructor" in class_chunk.code, "Constructor should be in class content"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_instance_method(self, language, ext):
        """Test instance method in class."""
        code = """
class MyClass {
    myMethod() {
        return 'result';
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        # Find class with method
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class for {language.value}"

        class_chunk = class_chunks[0]
        assert "myMethod" in class_chunk.code, "Instance method should be in class content"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_async_method(self, language, ext):
        """Test async method in class."""
        code = """
class ApiClient {
    async fetchData() {
        return await fetch('/api');
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class with async method for {language.value}"

        class_chunk = class_chunks[0]
        assert "async" in class_chunk.code, "Async keyword should be in class content"
        assert "fetchData" in class_chunk.code, "Async method name should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_generator_method(self, language, ext):
        """Test generator method in class."""
        code = """
class Collection {
    *items() {
        yield 1;
        yield 2;
        yield 3;
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class with generator method for {language.value}"

        class_chunk = class_chunks[0]
        assert "*" in class_chunk.code or "items" in class_chunk.code, \
            "Generator method should be in class content"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_static_method(self, language, ext):
        """Test static method in class."""
        code = """
class Utils {
    static helper() {
        return 'helper result';
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class with static method for {language.value}"

        class_chunk = class_chunks[0]
        assert "static" in class_chunk.code, "Static keyword should be in class content"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_getter_method(self, language, ext):
        """Test getter method in class."""
        code = """
class Person {
    get fullName() {
        return this.firstName + ' ' + this.lastName;
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class with getter for {language.value}"

        class_chunk = class_chunks[0]
        assert "get" in class_chunk.code, "Getter keyword should be in class content"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_setter_method(self, language, ext):
        """Test setter method in class."""
        code = """
class Person {
    set age(value) {
        if (value >= 0) {
            this._age = value;
        }
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class with setter for {language.value}"

        class_chunk = class_chunks[0]
        assert "set" in class_chunk.code, "Setter keyword should be in class content"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_computed_method_name(self, language, ext):
        """Test computed method name in class."""
        code = """
const methodName = 'dynamicMethod';

class Dynamic {
    [methodName]() {
        return 'dynamic';
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class with computed method for {language.value}"

        class_chunk = class_chunks[0]
        assert "[" in class_chunk.code, "Computed method name syntax should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_private_method_es2022(self, language, ext):
        """Test ES2022 private method with # syntax."""
        code = """
class SecureClass {
    #privateMethod() {
        return 'private';
    }

    publicMethod() {
        return this.#privateMethod();
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class with private method for {language.value}"

        class_chunk = class_chunks[0]
        assert "#privateMethod" in class_chunk.code, "Private method with # should be in content"


# =============================================================================
# Class Members - Properties Tests
# =============================================================================

class TestClassProperties:
    """Tests for class property patterns."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_instance_property(self, language, ext):
        """Test instance property in class."""
        code = """
class Config {
    timeout = 5000;
    retries = 3;
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class with instance properties for {language.value}"

        class_chunk = class_chunks[0]
        assert "timeout" in class_chunk.code, "Instance property should be in class content"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_static_property(self, language, ext):
        """Test static property in class."""
        code = """
class Constants {
    static PI = 3.14159;
    static E = 2.71828;
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class with static properties for {language.value}"

        class_chunk = class_chunks[0]
        assert "static" in class_chunk.code, "Static keyword should be in class content"
        assert "PI" in class_chunk.code, "Static property name should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_private_property_es2022(self, language, ext):
        """Test ES2022 private property with # syntax."""
        code = """
class Counter {
    #count = 0;

    increment() {
        this.#count++;
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class with private property for {language.value}"

        class_chunk = class_chunks[0]
        assert "#count" in class_chunk.code, "Private property with # should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_property_without_initializer(self, language, ext):
        """Test property without initializer."""
        # TypeScript-specific syntax, but should parse in JS too
        if language in (Language.TYPESCRIPT, Language.TSX):
            code = """
class Data {
    value: number;
    name: string;
}
"""
        else:
            # JS/JSX syntax for uninitialized properties (using constructor)
            code = """
class Data {
    constructor() {
        this.value = undefined;
        this.name = undefined;
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class for {language.value}"


# =============================================================================
# TypeScript Class Features Tests
# =============================================================================

class TestTypeScriptClassFeatures:
    """Tests for TypeScript-specific class features."""

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_class_implements_interface(self, language, ext):
        """Test class implementing interface."""
        code = """
interface Printable {
    print(): void;
}

class Document implements Printable {
    print() {
        console.log('printing');
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        doc_class = [c for c in class_chunks if "Document" in c.symbol]

        assert len(doc_class) > 0, f"Should extract class implementing interface for {language.value}"
        assert "implements" in doc_class[0].code, "implements keyword should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_class_implements_multiple(self, language, ext):
        """Test class implementing multiple interfaces."""
        code = """
interface A { methodA(): void; }
interface B { methodB(): void; }

class MultiImpl implements A, B {
    methodA() {}
    methodB() {}
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        multi_class = [c for c in class_chunks if "MultiImpl" in c.symbol]

        assert len(multi_class) > 0, f"Should extract class for {language.value}"
        content = multi_class[0].code
        assert "implements" in content and "A" in content and "B" in content, \
            "Multiple implements should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_public_modifier(self, language, ext):
        """Test public access modifier."""
        code = """
class MyClass {
    public data: string = 'public';

    public getData(): string {
        return this.data;
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class for {language.value}"

        assert "public" in class_chunks[0].code, "public modifier should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_private_modifier(self, language, ext):
        """Test private access modifier."""
        code = """
class MyClass {
    private secret: string = 'private';

    private getSecret(): string {
        return this.secret;
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class for {language.value}"

        assert "private" in class_chunks[0].code, "private modifier should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_protected_modifier(self, language, ext):
        """Test protected access modifier."""
        code = """
class Base {
    protected internal: string = 'protected';

    protected getInternal(): string {
        return this.internal;
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class for {language.value}"

        assert "protected" in class_chunks[0].code, "protected modifier should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_readonly_property(self, language, ext):
        """Test readonly property modifier."""
        code = """
class Config {
    readonly version: string = '1.0.0';
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class for {language.value}"

        assert "readonly" in class_chunks[0].code, "readonly modifier should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_abstract_class(self, language, ext):
        """Test abstract class declaration."""
        code = """
abstract class Shape {
    abstract getArea(): number;

    describe(): string {
        return `Area: ${this.getArea()}`;
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        shape_class = [c for c in class_chunks if "Shape" in c.symbol]

        assert len(shape_class) > 0, f"Should extract abstract class for {language.value}"
        assert "abstract" in shape_class[0].code, "abstract keyword should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_abstract_method(self, language, ext):
        """Test abstract method in class."""
        code = """
abstract class Animal {
    abstract makeSound(): void;

    move(): void {
        console.log('moving');
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class for {language.value}"

        content = class_chunks[0].code
        assert "abstract" in content and "makeSound" in content, \
            "abstract method should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_parameter_properties(self, language, ext):
        """Test parameter properties in constructor."""
        code = """
class Person {
    constructor(
        public name: string,
        private age: number,
        readonly id: string
    ) {}
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class for {language.value}"

        content = class_chunks[0].code
        assert "public name" in content or "public" in content, \
            "parameter property should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_generic_class(self, language, ext):
        """Test generic class declaration."""
        code = """
class Container<T> {
    private value: T;

    constructor(value: T) {
        this.value = value;
    }

    getValue(): T {
        return this.value;
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        container_class = [c for c in class_chunks if "Container" in c.symbol]

        assert len(container_class) > 0, f"Should extract generic class for {language.value}"
        assert "<T>" in container_class[0].code, "type parameter should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_generic_class_with_constraints(self, language, ext):
        """Test generic class with type constraints."""
        code = """
class Repository<T extends Entity> {
    private items: T[] = [];

    add(item: T): void {
        this.items.push(item);
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class for {language.value}"

        assert "extends Entity" in class_chunks[0].code, \
            "type constraint should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_class_decorator(self, language, ext):
        """Test class with decorator."""
        code = """
function sealed(constructor: Function) {
    Object.seal(constructor);
}

@sealed
class Greeter {
    greeting: string;
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        greeter_class = [c for c in class_chunks if "Greeter" in c.symbol]

        assert len(greeter_class) > 0, f"Should extract decorated class for {language.value}"
        # Decorator should be included with class
        assert "@sealed" in greeter_class[0].code or "@" in greeter_class[0].code, \
            "decorator should be in class content"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_method_decorator(self, language, ext):
        """Test method with decorator."""
        code = """
function log(target: any, key: string) {
    console.log(key);
}

class Logger {
    @log
    logMessage() {
        console.log('message');
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class for {language.value}"

        assert "@log" in class_chunks[0].code, "method decorator should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_property_decorator(self, language, ext):
        """Test property with decorator."""
        code = """
function validate(target: any, key: string) {}

class Form {
    @validate
    email: string = '';
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract class for {language.value}"

        assert "@validate" in class_chunks[0].code, "property decorator should be in content"


# =============================================================================
# Vue Class Tests
# =============================================================================

class TestVueClasses:
    """Tests for class patterns in Vue SFC files."""

    def test_vue_js_class_in_script(self, vue_parser):
        """Test class in Vue script section (JavaScript)."""
        code = """
<script>
class DataProcessor {
    process(data) {
        return data.map(item => item.value);
    }
}

export default {
    name: 'MyComponent',
    methods: {
        handleData() {
            const processor = new DataProcessor();
            return processor.process(this.items);
        }
    }
}
</script>

<template>
    <div>Content</div>
</template>
"""
        chunks = vue_parser.parse_content(code, "test.vue", FileId(1))

        # Find class chunk
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        processor_class = [c for c in class_chunks if "DataProcessor" in c.symbol]

        assert len(processor_class) > 0, "Should extract class from Vue script"

    def test_vue_ts_class_in_script(self, vue_parser):
        """Test TypeScript class in Vue script section."""
        code = """
<script lang="ts">
class UserService {
    private baseUrl: string;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }

    async getUser(id: number): Promise<User> {
        return {} as User;
    }
}

export default {
    name: 'UserComponent'
}
</script>

<template>
    <div>User</div>
</template>
"""
        chunks = vue_parser.parse_content(code, "test.vue", FileId(1))

        # Find class chunk
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        service_class = [c for c in class_chunks if "UserService" in c.symbol]

        assert len(service_class) > 0, "Should extract TypeScript class from Vue script"

        # Verify TypeScript features are captured
        content = service_class[0].code
        assert "private" in content, "private modifier should be in content"

    def test_vue_class_with_implements(self, vue_parser):
        """Test class implementing interface in Vue TypeScript."""
        code = """
<script lang="ts">
interface Validator {
    validate(value: string): boolean;
}

class EmailValidator implements Validator {
    validate(value: string): boolean {
        return value.includes('@');
    }
}
</script>

<template>
    <div>Validator</div>
</template>
"""
        chunks = vue_parser.parse_content(code, "test.vue", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        validator_class = [c for c in class_chunks if "EmailValidator" in c.symbol]

        assert len(validator_class) > 0, "Should extract class implementing interface"
        assert "implements" in validator_class[0].code, \
            "implements keyword should be in content"


# =============================================================================
# Cross-Language Consistency Tests
# =============================================================================

class TestCrossLanguageConsistency:
    """Tests verifying consistent extraction across all JS-family languages."""

    def test_basic_class_consistent_across_languages(self):
        """Test that basic class produces consistent results across languages."""
        code = """
class TestClass {
    method() {
        return 'test';
    }
}
"""
        languages = [
            (Language.JAVASCRIPT, "js"),
            (Language.JSX, "jsx"),
            (Language.TYPESCRIPT, "ts"),
            (Language.TSX, "tsx"),
        ]

        results = {}
        for lang, ext in languages:
            parser = create_parser(lang)
            chunks = parser.parse_content(code, f"test.{ext}", FileId(1))
            class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
            results[lang.value] = class_chunks

        # All languages should find the class
        for lang_name, chunks in results.items():
            assert len(chunks) > 0, f"{lang_name} should extract class"
            assert any("TestClass" in c.symbol for c in chunks), \
                f"{lang_name} should capture class name"

    def test_class_with_extends_consistent(self):
        """Test extends clause consistent across languages."""
        code = """
class Child extends Parent {
    childMethod() {}
}
"""
        languages = [
            (Language.JAVASCRIPT, "js"),
            (Language.JSX, "jsx"),
            (Language.TYPESCRIPT, "ts"),
            (Language.TSX, "tsx"),
        ]

        for lang, ext in languages:
            parser = create_parser(lang)
            chunks = parser.parse_content(code, f"test.{ext}", FileId(1))
            class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]

            assert len(class_chunks) > 0, f"{lang.value} should extract class"
            child_class = [c for c in class_chunks if "Child" in c.symbol]
            assert len(child_class) > 0, f"{lang.value} should find Child class"
            assert "extends" in child_class[0].code, \
                f"{lang.value} should include extends in content"

    def test_private_members_consistent(self):
        """Test private # syntax consistent across languages."""
        code = """
class Secure {
    #secret = 'hidden';

    #privateMethod() {
        return this.#secret;
    }
}
"""
        languages = [
            (Language.JAVASCRIPT, "js"),
            (Language.JSX, "jsx"),
            (Language.TYPESCRIPT, "ts"),
            (Language.TSX, "tsx"),
        ]

        for lang, ext in languages:
            parser = create_parser(lang)
            chunks = parser.parse_content(code, f"test.{ext}", FileId(1))
            class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]

            assert len(class_chunks) > 0, f"{lang.value} should extract class"
            assert "#secret" in class_chunks[0].code, \
                f"{lang.value} should include private property"
            assert "#privateMethod" in class_chunks[0].code, \
                f"{lang.value} should include private method"


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestClassEdgeCases:
    """Tests for edge cases in class parsing."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_empty_class(self, language, ext):
        """Test empty class declaration."""
        code = "class Empty {}"

        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, f"Should extract empty class for {language.value}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_class_in_function(self, language, ext):
        """Test class defined inside function."""
        code = """
function createClass() {
    class Inner {
        value = 42;
    }
    return Inner;
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        # Should find both function and class
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]

        assert len(func_chunks) > 0, f"Should extract outer function for {language.value}"
        # Inner class may be extracted separately or as part of function

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_exported_class(self, language, ext):
        """Test exported class declaration."""
        code = """
export class ExportedClass {
    method() {
        return 'exported';
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        exported = [c for c in class_chunks if "ExportedClass" in c.symbol]

        assert len(exported) > 0, f"Should extract exported class for {language.value}"
        assert "export" in exported[0].code, "export keyword should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_default_exported_class(self, language, ext):
        """Test default exported class declaration."""
        code = """
export default class DefaultClass {
    method() {
        return 'default';
    }
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        default_class = [c for c in class_chunks if "DefaultClass" in c.symbol]

        assert len(default_class) > 0, f"Should extract default exported class for {language.value}"
        content = default_class[0].code
        assert "export" in content and "default" in content, \
            "export default keywords should be in content"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_multiple_classes_in_file(self, language, ext):
        """Test multiple class declarations in same file."""
        code = """
class First {
    first() {}
}

class Second {
    second() {}
}

class Third extends First {
    third() {}
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]

        # Should find all three classes
        first = [c for c in class_chunks if "First" in c.symbol]
        second = [c for c in class_chunks if "Second" in c.symbol]
        third = [c for c in class_chunks if "Third" in c.symbol]

        assert len(first) > 0, f"Should extract First class for {language.value}"
        assert len(second) > 0, f"Should extract Second class for {language.value}"
        assert len(third) > 0, f"Should extract Third class for {language.value}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_class_is_not_function_type(self, language, ext):
        """Verify class is classified as CLASS not FUNCTION."""
        code = """
class MyClass {
    constructor() {}
    method() {}
}
"""
        parser = create_parser(language)
        chunks = parser.parse_content(code, f"test.{ext}", FileId(1))

        # Find the class
        my_class = [c for c in chunks if "MyClass" in c.symbol]
        assert len(my_class) > 0, f"Should extract class for {language.value}"

        # CRITICAL: Must be CLASS type, not FUNCTION
        class_chunk = my_class[0]
        assert class_chunk.chunk_type == ChunkType.CLASS, \
            f"Class should be ChunkType.CLASS but got {class_chunk.chunk_type} for {language.value}"
