"""Function-related edge cases for JS-family parsers.

This module tests advanced function patterns that are not covered by the
main comprehensive function tests, including:
- Object method shorthand (including async and generator methods)
- Getter and setter accessor functions
- Legacy constructor functions
- Functions as object properties (both function expressions and arrows)

These tests focus on edge cases and modern JavaScript/TypeScript patterns
that may require special handling in the parser.
"""

import pytest
from pathlib import Path

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import get_parser_factory


# =============================================================================
# Helper Functions
# =============================================================================

def create_parser(language: Language):
    """Create a parser for the specified language."""
    factory = get_parser_factory()
    return factory.create_parser(language)


def parse_code(code: str, filename: str, language: Language):
    """Helper to parse code and return chunks."""
    parser = create_parser(language)
    return parser.parse_content(code, filename, FileId(1))


# =============================================================================
# Test Classes
# =============================================================================


class TestObjectMethodShorthand:
    """Test ES6 method shorthand syntax in object literals.

    Modern JavaScript allows defining methods in objects using shorthand:
    { method() { ... } } instead of { method: function() { ... } }

    This includes async methods, generator methods, and regular methods.
    """

    def test_method_shorthand(self):
        """Test basic method shorthand in object literal.

        Pattern: const obj = { method() { return 42; } }
        Expected: Method should be extracted with proper name
        """
        code = """
const obj = {
    method() {
        return 42;
    }
};
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Look for chunks containing the method name
        method_chunks = [c for c in chunks if "method" in c.code]
        assert len(method_chunks) > 0, "Should find method shorthand"
        # Verify it contains the return statement
        assert any("return 42" in c.code for c in method_chunks), "Should preserve method body"

    def test_async_method_shorthand(self):
        """Test async method shorthand in object literal.

        Pattern: const obj = { async fetch() { await ... } }
        Expected: Async method should be extracted with async keyword preserved
        """
        code = """
const obj = {
    async fetch() {
        const response = await fetch('/api');
        return response.json();
    }
};
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Look for async method
        async_chunks = [c for c in chunks if "fetch" in c.code and "async" in c.code]
        assert len(async_chunks) > 0, "Should find async method shorthand"
        # Verify await is preserved
        assert any("await" in c.code for c in async_chunks), "Should preserve await keyword"

    @pytest.mark.xfail(reason="Generator method shorthand not yet extracted by JS-family parsers")
    def test_generator_method_shorthand(self):
        """Test generator method shorthand in object literal.

        Pattern: const obj = { *generate() { yield 1; } }
        Expected: Generator method should be extracted with * syntax preserved
        """
        code = """
const obj = {
    *generate() {
        yield 1;
        yield 2;
        yield 3;
    }
};
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Look for generator method
        gen_chunks = [c for c in chunks if "generate" in c.code]
        assert len(gen_chunks) > 0, "Should find generator method"
        # Verify generator syntax is preserved
        assert any("*generate" in c.code or "* generate" in c.code for c in gen_chunks), \
            "Should preserve generator method syntax"
        assert any("yield" in c.code for c in gen_chunks), "Should preserve yield statements"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_method_shorthand_cross_language(self, language, ext):
        """Test method shorthand works consistently across JS-family languages."""
        code = """
const utils = {
    calculate() {
        return 100;
    }
};
"""
        chunks = parse_code(code, f"test.{ext}", language)

        assert len(chunks) > 0, f"Should extract chunk for {language.value}"
        assert any("calculate" in c.code for c in chunks), \
            f"Should find method shorthand in {language.value}"


class TestAccessorFunctions:
    """Test getter and setter accessor functions.

    JavaScript getters and setters allow property-like access to functions:
    obj.prop instead of obj.getProp()

    These use special syntax: get prop() { ... } and set prop(value) { ... }
    """

    @pytest.mark.xfail(reason="Getter functions not yet extracted by JS-family parsers")
    def test_getter(self):
        """Test getter accessor function.

        Pattern: const obj = { get prop() { return this._prop; } }
        Expected: Getter should be extracted with 'get' keyword preserved
        """
        code = """
const obj = {
    _value: 42,
    get prop() {
        return this._value;
    }
};
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Look for getter
        getter_chunks = [c for c in chunks if "get prop" in c.code or "get  prop" in c.code]
        assert len(getter_chunks) > 0, "Should find getter function"
        # Verify it's a getter, not a regular method
        assert any("get" in c.code and "prop" in c.code for c in getter_chunks), \
            "Should preserve 'get' keyword"

    @pytest.mark.xfail(reason="Setter functions not yet extracted by JS-family parsers")
    def test_setter(self):
        """Test setter accessor function.

        Pattern: const obj = { set prop(value) { this._prop = value; } }
        Expected: Setter should be extracted with 'set' keyword preserved
        """
        code = """
const obj = {
    _value: 0,
    set prop(value) {
        this._value = value;
    }
};
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Look for setter
        setter_chunks = [c for c in chunks if "set prop" in c.code or "set  prop" in c.code]
        assert len(setter_chunks) > 0, "Should find setter function"
        # Verify it's a setter with parameter
        assert any("set" in c.code and "prop" in c.code and "value" in c.code
                   for c in setter_chunks), "Should preserve 'set' keyword and parameter"

    @pytest.mark.xfail(reason="Getter/setter pairs not yet extracted by JS-family parsers")
    def test_getter_and_setter(self):
        """Test object with both getter and setter for same property.

        Pattern: Both getter and setter for a single property
        Expected: Both should be extracted, possibly as separate chunks
        """
        code = """
const obj = {
    _value: 0,
    get value() {
        return this._value;
    },
    set value(val) {
        this._value = val;
    }
};
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Look for both getter and setter
        value_chunks = [c for c in chunks if "value" in c.code]
        assert len(value_chunks) > 0, "Should find value accessors"
        # Should have both get and set somewhere in the chunks
        has_getter = any("get" in c.code and "value" in c.code for c in chunks)
        has_setter = any("set" in c.code and "value" in c.code for c in chunks)
        assert has_getter, "Should extract getter"
        assert has_setter, "Should extract setter"

    @pytest.mark.xfail(reason="TypeScript getters not yet extracted")
    def test_typescript_getter_with_type(self):
        """Test TypeScript getter with return type annotation.

        Pattern: get prop(): string { return this._prop; }
        Expected: Type annotation should be preserved
        """
        code = """
class MyClass {
    private _name: string = 'test';
    get name(): string {
        return this._name;
    }
}
"""
        chunks = parse_code(code, "test.ts", Language.TYPESCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Look for typed getter
        getter_chunks = [c for c in chunks if "get name" in c.code]
        assert len(getter_chunks) > 0, "Should find typed getter"
        # Verify type annotation is preserved
        assert any(": string" in c.code for c in getter_chunks), \
            "Should preserve return type annotation"


class TestConstructorFunctions:
    """Test legacy constructor function pattern.

    Before ES6 classes, JavaScript used constructor functions:
    function Person(name) { this.name = name; }

    These are regular functions but used with 'new' keyword.
    Convention: Capitalized names indicate constructor intent.
    """

    def test_constructor_function(self):
        """Test legacy constructor function pattern.

        Pattern: function Person(name) { this.name = name; }
        Expected: Should be extracted as a regular function
        Note: Parser may not distinguish constructors from regular functions
              without analyzing usage context (new keyword)
        """
        code = """
function Person(name, age) {
    this.name = name;
    this.age = age;
    this.greet = function() {
        return 'Hello, ' + this.name;
    };
}
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Look for constructor function
        person_chunks = [c for c in chunks if "Person" in c.code]
        assert len(person_chunks) > 0, "Should find constructor function"
        # Verify it contains 'this' assignments
        assert any("this.name" in c.code for c in person_chunks), \
            "Should preserve this assignments"

    @pytest.mark.xfail(reason="Prototype method assignments not yet extracted separately")
    def test_constructor_with_prototype(self):
        """Test constructor function with prototype methods.

        Pattern:
        function Person(name) { this.name = name; }
        Person.prototype.greet = function() { ... }

        Expected: Both constructor and prototype assignments should be extracted
        Note: Prototype assignments are typically not extracted as separate chunks
        """
        code = """
function Person(name) {
    this.name = name;
}

Person.prototype.greet = function() {
    return 'Hello, ' + this.name;
};

Person.prototype.farewell = function() {
    return 'Goodbye, ' + this.name;
};
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should find the constructor
        assert any("Person" in c.code and "this.name" in c.code for c in chunks), \
            "Should find constructor function"
        # Should find prototype methods
        assert any("greet" in c.code for c in chunks), "Should find prototype method"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_constructor_function_cross_language(self, language, ext):
        """Test constructor function pattern across languages.

        Note: TypeScript has better alternatives (classes), but constructor
        functions are still valid JavaScript syntax.
        """
        code = """
function Vehicle(type) {
    this.type = type;
    this.drive = function() {
        return this.type + ' is driving';
    };
}
"""
        chunks = parse_code(code, f"test.{ext}", language)

        assert len(chunks) > 0, f"Should extract chunk for {language.value}"
        assert any("Vehicle" in c.code for c in chunks), \
            f"Should find constructor in {language.value}"


class TestFunctionEdgeCases:
    """Test edge cases for functions as object properties.

    Functions can be assigned as object properties in multiple ways:
    - Traditional function expressions: { fn: function() {} }
    - Arrow functions: { fn: () => {} }
    - Named function expressions: { fn: function named() {} }
    """

    def test_function_as_object_property(self):
        """Test function expression as object property.

        Pattern: const obj = { fn: function() {} }
        Expected: Should extract the object with function property
        """
        code = """
const obj = {
    handler: function(event) {
        console.log('Handling:', event);
        return true;
    }
};
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Look for function property
        handler_chunks = [c for c in chunks if "handler" in c.code]
        assert len(handler_chunks) > 0, "Should find function property"
        # Verify it contains the function keyword
        assert any("function" in c.code for c in handler_chunks), \
            "Should preserve function expression"

    def test_arrow_as_object_property(self):
        """Test arrow function as object property.

        Pattern: const obj = { fn: () => {} }
        Expected: Should extract the object with arrow function
        """
        code = """
const obj = {
    compute: (x, y) => {
        const result = x * y;
        return result;
    }
};
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Look for arrow function property
        compute_chunks = [c for c in chunks if "compute" in c.code]
        assert len(compute_chunks) > 0, "Should find arrow property"
        # Verify arrow syntax is preserved
        assert any("=>" in c.code for c in compute_chunks), \
            "Should preserve arrow function syntax"

    def test_named_function_as_property(self):
        """Test named function expression as object property.

        Pattern: const obj = { prop: function named() {} }
        Expected: Should extract with both property name and function name
        """
        code = """
const obj = {
    callback: function handleCallback(data) {
        return data.processed;
    }
};
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Look for the property
        callback_chunks = [c for c in chunks if "callback" in c.code]
        assert len(callback_chunks) > 0, "Should find function property"
        # May or may not preserve the inner function name
        # At minimum should have the property name

    def test_mixed_function_properties(self):
        """Test object with multiple function property styles.

        Combines:
        - Method shorthand
        - Function expression
        - Arrow function
        """
        code = """
const api = {
    get() {
        return this.data;
    },
    set: function(value) {
        this.data = value;
    },
    clear: () => {
        this.data = null;
    }
};
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should find the object with various function properties
        assert any("get" in c.code for c in chunks), "Should find method shorthand"
        assert any("set" in c.code for c in chunks), "Should find function expression"
        assert any("clear" in c.code for c in chunks), "Should find arrow function"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_function_properties_cross_language(self, language, ext):
        """Test function as object property across languages."""
        code = """
const config = {
    validate: function(data) {
        return data !== null;
    }
};
"""
        chunks = parse_code(code, f"test.{ext}", language)

        assert len(chunks) > 0, f"Should extract chunk for {language.value}"
        assert any("validate" in c.code for c in chunks), \
            f"Should find function property in {language.value}"

    def test_typescript_typed_function_property(self):
        """Test TypeScript typed function as object property.

        Pattern: const obj = { fn: (x: string): number => x.length }
        Expected: Type annotations should be preserved
        """
        code = """
const utils = {
    transform: (input: string): number => {
        return input.length;
    }
};
"""
        chunks = parse_code(code, "test.ts", Language.TYPESCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Look for typed arrow function
        transform_chunks = [c for c in chunks if "transform" in c.code]
        assert len(transform_chunks) > 0, "Should find typed function property"
        # Verify types are preserved
        assert any(": string" in c.code for c in transform_chunks), \
            "Should preserve parameter type"
        assert any(": number" in c.code for c in transform_chunks), \
            "Should preserve return type"


# =============================================================================
# Integration Tests
# =============================================================================


class TestFunctionEdgeCasesIntegration:
    """Integration tests combining multiple edge case patterns.

    Real-world code often combines multiple patterns, so we test
    that the parser handles complex scenarios correctly.
    """

    def test_complex_object_with_multiple_function_styles(self):
        """Test object combining multiple function definition styles.

        Realistic scenario: API client with various method types
        """
        code = """
const apiClient = {
    baseUrl: 'https://api.example.com',

    get(endpoint) {
        return fetch(this.baseUrl + endpoint);
    },

    post: function(endpoint, data) {
        return fetch(this.baseUrl + endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    delete: (endpoint) => {
        return fetch(this.baseUrl + endpoint, { method: 'DELETE' });
    },

    async fetchJson(endpoint) {
        const response = await this.get(endpoint);
        return response.json();
    }
};
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should find the object with all methods
        assert any("apiClient" in c.code for c in chunks), "Should find object"
        # Verify various function styles are present
        api_chunks = [c for c in chunks if "apiClient" in c.code or "get" in c.code or "post" in c.code]
        assert len(api_chunks) > 0, "Should extract API client methods"

    def test_constructor_with_method_properties(self):
        """Test constructor function with methods as properties.

        Pattern: Old-style JavaScript OOP
        """
        code = """
function Animal(name) {
    this.name = name;

    this.speak = function() {
        return this.name + ' makes a sound';
    };

    this.eat = () => {
        return this.name + ' is eating';
    };
}

Animal.prototype.sleep = function() {
    return this.name + ' is sleeping';
};
"""
        chunks = parse_code(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should find constructor
        assert any("Animal" in c.code and "this.name" in c.code for c in chunks), \
            "Should find constructor"
        # Should find methods
        assert any("speak" in c.code or "eat" in c.code or "sleep" in c.code for c in chunks), \
            "Should find methods"

    @pytest.mark.xfail(reason="Complex accessor patterns not yet fully supported")
    def test_typescript_class_with_accessors(self):
        """Test TypeScript class with getter/setter and regular methods.

        Realistic scenario: Class with computed properties
        """
        code = """
class User {
    private _firstName: string;
    private _lastName: string;

    constructor(first: string, last: string) {
        this._firstName = first;
        this._lastName = last;
    }

    get fullName(): string {
        return this._firstName + ' ' + this._lastName;
    }

    set fullName(value: string) {
        const parts = value.split(' ');
        this._firstName = parts[0];
        this._lastName = parts[1];
    }

    greet(): string {
        return 'Hello, ' + this.fullName;
    }
}
"""
        chunks = parse_code(code, "test.ts", Language.TYPESCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should find class
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should find class"
        # Should extract methods and accessors
        assert any("fullName" in c.code for c in chunks), "Should find accessor"
        assert any("greet" in c.code for c in chunks), "Should find method"
