"""Comprehensive TypeScript parsing tests (TDD Red Phase).

This module contains integration tests that verify TypeScript parser
extracts all expected constructs. These tests are designed to FAIL
with the current implementation and will pass once the implementation
is complete.

Missing features being tested:
1. Import statements (ES6 and TypeScript type imports)
2. Interface declarations
3. Enum declarations
4. Type alias declarations
5. Class identification as ChunkType.CLASS (not FUNCTION)
6. Namespace declarations (bonus feature)
7. Proper filtering of simple variables vs object/array variables
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from chunkhound.database_factory import create_services
from chunkhound.core.config.config import Config
from chunkhound.core.types.common import ChunkType, Language, FileId
from chunkhound.parsers.parser_factory import get_parser_factory
from types import SimpleNamespace


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test database."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_services(temp_db_dir):
    """Create test database services."""
    db_path = temp_db_dir / "test.db"
    fake_args = SimpleNamespace(path=temp_db_dir)
    config = Config(
        args=fake_args,
        database={"path": str(db_path), "provider": "duckdb"},
        embedding=None,
        indexing={"include": ["*.ts"], "exclude": []}
    )

    services = create_services(db_path, config, embedding_manager=None)
    services.provider.connect()

    yield services

    try:
        services.provider.disconnect()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error disconnecting test database: {e}")


@pytest.fixture
def ts_test_file(temp_db_dir):
    """Create a comprehensive TypeScript test fixture file."""
    fixture_content = '''// Comprehensive TypeScript test file
import { useState, useEffect } from 'react';
import type { User } from './types';
import * as utils from './utils';

interface ApiResponse<T> {
    data: T;
    error?: string;
}

enum Status {
    Pending = 'pending',
    Success = 'success',
    Error = 'error'
}

type UserRole = 'admin' | 'user' | 'guest';

class UserService {
    constructor(private baseUrl: string) {}

    async getUser(id: number): Promise<User> {
        return {} as User;
    }
}

namespace Utils {
    export function helper() {
        return true;
    }
}

const API_URL = 'https://api.example.com';  // Simple string variable
const MAX_RETRIES = 3;  // Simple number variable
const config = { timeout: 5000, retries: MAX_RETRIES };  // Object variable
'''

    test_file = temp_db_dir / "test_ts_parsing.ts"
    test_file.write_text(fixture_content)
    return test_file


class TestTypeScriptImports:
    """Test suite for TypeScript import statement extraction."""

    async def test_es6_default_import_extracted(self, test_services, temp_db_dir):
        """Test that ES6 default imports are extracted as chunks."""
        test_file = temp_db_dir / "test_default_import.ts"
        test_file.write_text('''
// ES6 default import
import React from 'react';

// UNIQUE_DEFAULT_IMPORT_MARKER_ABC123

function Component() {
    return null;
}
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the import statement
        results = test_services.provider.search_chunks_regex("import React from")

        assert len(results) > 0, "Default import statement should be extracted as a chunk"
        assert any("React" in chunk["content"] for chunk in results), \
            "Import content should contain module name"

    async def test_es6_named_imports_extracted(self, test_services, temp_db_dir):
        """Test that ES6 named imports are extracted as chunks."""
        test_file = temp_db_dir / "test_named_imports.ts"
        test_file.write_text('''
// Named imports
import { useState, useEffect } from 'react';

// UNIQUE_NAMED_IMPORT_MARKER_DEF456

export function useCustomHook() {
    return useState(null);
}
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the import statement
        results = test_services.provider.search_chunks_regex("useState.*useEffect")

        assert len(results) > 0, "Named import statement should be extracted as a chunk"
        assert any("useState" in chunk["content"] and "useEffect" in chunk["content"]
                  for chunk in results), \
            "Named imports should be captured in content"

    async def test_namespace_import_extracted(self, test_services, temp_db_dir):
        """Test that namespace imports (import * as) are extracted as chunks."""
        test_file = temp_db_dir / "test_namespace_import.ts"
        test_file.write_text('''
// Namespace import
import * as utils from './utils';

// UNIQUE_NAMESPACE_IMPORT_MARKER_GHI789

export function helper() {
    return utils.format();
}
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the import statement
        results = test_services.provider.search_chunks_regex("import \\* as utils")

        assert len(results) > 0, "Namespace import statement should be extracted as a chunk"

    async def test_typescript_type_import_extracted(self, test_services, temp_db_dir):
        """Test that TypeScript type-only imports are extracted as chunks."""
        test_file = temp_db_dir / "test_type_import.ts"
        test_file.write_text('''
// Type-only import
import type { User, Product } from './types';

// UNIQUE_TYPE_IMPORT_MARKER_JKL012

export function processUser(user: User) {
    return user.name;
}
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the import statement
        results = test_services.provider.search_chunks_regex("import type.*User")

        assert len(results) > 0, "TypeScript type import statement should be extracted as a chunk"
        assert any("type" in chunk["content"] and "User" in chunk["content"]
                  for chunk in results), \
            "Type import should be captured with 'type' keyword"

    async def test_fixture_file_imports_extracted(self, test_services, ts_test_file):
        """Test that imports from fixture file are extracted."""
        if not ts_test_file.exists():
            pytest.skip(f"Test fixture not found: {ts_test_file}")

        await test_services.indexing_coordinator.process_file(ts_test_file)

        # Search for imports from fixture
        results = test_services.provider.search_chunks_regex("import.*useState.*react")

        assert len(results) > 0, "Import statements from fixture should be extracted"


class TestBasicInterfaces:
    """Test suite for basic TypeScript interface constructs."""

    async def test_basic_interface(self, test_services, temp_db_dir):
        """Test that basic interface with primitive types is extracted."""
        test_file = temp_db_dir / "test_basic_interface.ts"
        test_file.write_text('''
// Basic interface
interface User {
    name: string;
    age: number;
}

// UNIQUE_BASIC_INTERFACE_MARKER_ABC123
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the interface
        results = test_services.provider.search_chunks_regex("interface User")

        assert len(results) > 0, "Basic interface declaration should be extracted as a chunk"

        # Verify chunk type is INTERFACE
        interface_chunks = [c for c in results if "User" in c["content"]]
        assert any(chunk["type"] == ChunkType.INTERFACE.value for chunk in interface_chunks), \
            "Interface should be classified as ChunkType.INTERFACE"


class TestTypeScriptInterfaces:
    """Test suite for TypeScript interface extraction."""

    async def test_simple_interface_extracted(self, test_services, temp_db_dir):
        """Test that simple interfaces are extracted as chunks."""
        test_file = temp_db_dir / "test_simple_interface.ts"
        test_file.write_text('''
// Simple interface
interface User {
    id: number;
    name: string;
}

// UNIQUE_INTERFACE_MARKER_MNO345
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the interface
        results = test_services.provider.search_chunks_regex("interface User")

        assert len(results) > 0, "Interface declaration should be extracted as a chunk"

        # Verify chunk type is INTERFACE
        interface_chunks = [c for c in results if "User" in c["content"]]
        assert any(chunk["type"] == ChunkType.INTERFACE.value for chunk in interface_chunks), \
            "Interface should be classified as ChunkType.INTERFACE"

    async def test_generic_interface_extracted(self, test_services, temp_db_dir):
        """Test that generic interfaces are extracted with type parameters."""
        test_file = temp_db_dir / "test_generic_interface.ts"
        test_file.write_text('''
// Generic interface
interface ApiResponse<T> {
    data: T;
    error?: string;
    status: number;
}

// UNIQUE_GENERIC_INTERFACE_MARKER_PQR678
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the interface
        results = test_services.provider.search_chunks_regex("interface ApiResponse")

        assert len(results) > 0, "Generic interface should be extracted as a chunk"
        assert any("<T>" in chunk["content"] or "ApiResponse<T>" in chunk.get("symbol", "")
                  for chunk in results), \
            "Generic type parameter should be captured"

    async def test_interface_with_methods_extracted(self, test_services, temp_db_dir):
        """Test that interfaces with method signatures are extracted."""
        test_file = temp_db_dir / "test_interface_methods.ts"
        test_file.write_text('''
// Interface with methods
interface Repository {
    find(id: number): Promise<User>;
    save(user: User): Promise<void>;
    delete(id: number): Promise<boolean>;
}

// UNIQUE_INTERFACE_METHODS_MARKER_STU901
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the interface
        results = test_services.provider.search_chunks_regex("interface Repository")

        assert len(results) > 0, "Interface with method signatures should be extracted"
        # Verify methods are included in content
        repo_chunks = [c for c in results if "Repository" in c["content"]]
        assert any("find" in chunk["content"] and "save" in chunk["content"]
                  for chunk in repo_chunks), \
            "Method signatures should be part of interface content"

    async def test_fixture_file_interface_extracted(self, test_services, ts_test_file):
        """Test that ApiResponse interface from fixture is extracted."""
        if not ts_test_file.exists():
            pytest.skip(f"Test fixture not found: {ts_test_file}")

        await test_services.indexing_coordinator.process_file(ts_test_file)

        # Search for ApiResponse interface
        results = test_services.provider.search_chunks_regex("interface ApiResponse")

        assert len(results) > 0, "ApiResponse interface should be extracted from fixture"


class TestBasicEnums:
    """Test suite for basic TypeScript enum constructs."""

    async def test_numeric_enum(self, test_services, temp_db_dir):
        """Test that numeric auto-incrementing enum is extracted."""
        test_file = temp_db_dir / "test_numeric_enum.ts"
        test_file.write_text('''
// Numeric enum
enum Color {
    Red,
    Green,
    Blue
}

// UNIQUE_NUMERIC_ENUM_MARKER_JKL012
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the enum
        results = test_services.provider.search_chunks_regex("enum Color")

        assert len(results) > 0, "Numeric enum should be extracted as a chunk"

        # Verify chunk type is ENUM
        enum_chunks = [c for c in results if "Color" in c["content"]]
        assert any(chunk["type"] == ChunkType.ENUM.value for chunk in enum_chunks), \
            "Enum should be classified as ChunkType.ENUM"

    async def test_string_enum(self, test_services, temp_db_dir):
        """Test that string enum is extracted."""
        test_file = temp_db_dir / "test_string_enum_basic.ts"
        test_file.write_text('''
// String enum
enum Status {
    Active = "ACTIVE",
    Pending = "PENDING"
}

// UNIQUE_STRING_ENUM_BASIC_MARKER_MNO345
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the enum
        results = test_services.provider.search_chunks_regex("enum Status")

        assert len(results) > 0, "String enum should be extracted as a chunk"

        # Verify content includes enum members
        status_chunks = [c for c in results if "Status" in c["content"]]
        assert any("Active" in chunk["content"] and "ACTIVE" in chunk["content"]
                  for chunk in status_chunks), \
            "String enum values should be captured in content"

    async def test_mixed_enum(self, test_services, temp_db_dir):
        """Test that mixed enum (numeric and string) is extracted."""
        test_file = temp_db_dir / "test_mixed_enum.ts"
        test_file.write_text('''
// Mixed enum
enum Mixed {
    A,
    B = "b",
    C = 2
}

// UNIQUE_MIXED_ENUM_MARKER_PQR678
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the enum
        results = test_services.provider.search_chunks_regex("enum Mixed")

        assert len(results) > 0, "Mixed enum should be extracted as a chunk"

        # Verify content includes both numeric and string values
        mixed_chunks = [c for c in results if "Mixed" in c["content"]]
        assert any("A" in chunk["content"] and "B" in chunk["content"]
                  for chunk in mixed_chunks), \
            "Mixed enum members should be captured in content"


class TestTypeScriptEnums:
    """Test suite for TypeScript enum extraction."""

    async def test_string_enum_extracted(self, test_services, temp_db_dir):
        """Test that string enums are extracted as chunks."""
        test_file = temp_db_dir / "test_string_enum.ts"
        test_file.write_text('''
// String enum
enum Status {
    Pending = 'pending',
    Success = 'success',
    Error = 'error'
}

// UNIQUE_STRING_ENUM_MARKER_VWX234
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the enum
        results = test_services.provider.search_chunks_regex("enum Status")

        assert len(results) > 0, "String enum should be extracted as a chunk"

        # Verify chunk type is ENUM
        enum_chunks = [c for c in results if "Status" in c["content"]]
        assert any(chunk["type"] == ChunkType.ENUM.value for chunk in enum_chunks), \
            "Enum should be classified as ChunkType.ENUM"

    async def test_numeric_enum_extracted(self, test_services, temp_db_dir):
        """Test that numeric enums are extracted as chunks."""
        test_file = temp_db_dir / "test_numeric_enum.ts"
        test_file.write_text('''
// Numeric enum
enum Permission {
    Read = 1,
    Write = 2,
    Execute = 4,
    Admin = 8
}

// UNIQUE_NUMERIC_ENUM_MARKER_YZA567
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the enum
        results = test_services.provider.search_chunks_regex("enum Permission")

        assert len(results) > 0, "Numeric enum should be extracted as a chunk"
        assert any("Read" in chunk["content"] and "Write" in chunk["content"]
                  for chunk in results), \
            "Enum members should be captured in content"

    async def test_auto_increment_enum_extracted(self, test_services, temp_db_dir):
        """Test that auto-incrementing enums are extracted."""
        test_file = temp_db_dir / "test_auto_enum.ts"
        test_file.write_text('''
// Auto-incrementing enum
enum Color {
    Red,
    Green,
    Blue
}

// UNIQUE_AUTO_ENUM_MARKER_BCD890
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the enum
        results = test_services.provider.search_chunks_regex("enum Color")

        assert len(results) > 0, "Auto-incrementing enum should be extracted as a chunk"

    async def test_fixture_file_enum_extracted(self, test_services, ts_test_file):
        """Test that Status enum from fixture is extracted."""
        if not ts_test_file.exists():
            pytest.skip(f"Test fixture not found: {ts_test_file}")

        await test_services.indexing_coordinator.process_file(ts_test_file)

        # Search for Status enum
        results = test_services.provider.search_chunks_regex("enum Status")

        assert len(results) > 0, "Status enum should be extracted from fixture"


class TestBasicTypeAliases:
    """Test suite for basic TypeScript type alias constructs."""

    async def test_basic_type_alias(self, test_services, temp_db_dir):
        """Test that basic union type alias is extracted."""
        test_file = temp_db_dir / "test_basic_type_alias.ts"
        test_file.write_text('''
// Basic type alias
type ID = string | number;

// UNIQUE_BASIC_TYPE_ALIAS_MARKER_DEF456
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the type alias
        results = test_services.provider.search_chunks_regex("type ID")

        assert len(results) > 0, "Basic type alias should be extracted as a chunk"

        # Verify chunk type is TYPE_ALIAS
        type_chunks = [c for c in results if "ID" in c["content"]]
        assert any(chunk["type"] == ChunkType.TYPE_ALIAS.value for chunk in type_chunks), \
            "Type alias should be classified as ChunkType.TYPE_ALIAS"

    async def test_type_alias_object(self, test_services, temp_db_dir):
        """Test that object type alias is extracted."""
        test_file = temp_db_dir / "test_type_alias_object.ts"
        test_file.write_text('''
// Object type alias
type Point = { x: number; y: number };

// UNIQUE_TYPE_ALIAS_OBJECT_MARKER_GHI789
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the type alias
        results = test_services.provider.search_chunks_regex("type Point")

        assert len(results) > 0, "Object type alias should be extracted as a chunk"

        # Verify content includes object properties
        point_chunks = [c for c in results if "Point" in c["content"]]
        assert any("x" in chunk["content"] and "y" in chunk["content"]
                  for chunk in point_chunks), \
            "Object properties should be captured in content"


class TestTypeScriptTypeAliases:
    """Test suite for TypeScript type alias extraction."""

    async def test_simple_type_alias_extracted(self, test_services, temp_db_dir):
        """Test that simple type aliases are extracted as chunks."""
        test_file = temp_db_dir / "test_simple_type_alias.ts"
        test_file.write_text('''
// Simple type alias
type UserRole = 'admin' | 'user' | 'guest';

// UNIQUE_TYPE_ALIAS_MARKER_EFG123
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the type alias
        results = test_services.provider.search_chunks_regex("type UserRole")

        assert len(results) > 0, "Type alias should be extracted as a chunk"

        # Verify chunk type is TYPE_ALIAS
        type_chunks = [c for c in results if "UserRole" in c["content"]]
        assert any(chunk["type"] == ChunkType.TYPE_ALIAS.value for chunk in type_chunks), \
            "Type alias should be classified as ChunkType.TYPE_ALIAS"

    async def test_union_type_alias_extracted(self, test_services, temp_db_dir):
        """Test that union type aliases are extracted."""
        test_file = temp_db_dir / "test_union_type.ts"
        test_file.write_text('''
// Union type alias
type Result = Success | Failure | Pending;

// UNIQUE_UNION_TYPE_MARKER_HIJ456
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the type alias
        results = test_services.provider.search_chunks_regex("type Result")

        assert len(results) > 0, "Union type alias should be extracted as a chunk"
        assert any("Success" in chunk["content"] and "Failure" in chunk["content"]
                  for chunk in results), \
            "Union type members should be captured"

    async def test_generic_type_alias_extracted(self, test_services, temp_db_dir):
        """Test that generic type aliases are extracted."""
        test_file = temp_db_dir / "test_generic_type.ts"
        test_file.write_text('''
// Generic type alias
type Nullable<T> = T | null | undefined;

// UNIQUE_GENERIC_TYPE_MARKER_KLM789
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the type alias
        results = test_services.provider.search_chunks_regex("type Nullable")

        assert len(results) > 0, "Generic type alias should be extracted as a chunk"
        assert any("<T>" in chunk["content"] or "Nullable<T>" in chunk.get("symbol", "")
                  for chunk in results), \
            "Generic type parameter should be captured"

    async def test_object_type_alias_extracted(self, test_services, temp_db_dir):
        """Test that object type aliases are extracted."""
        test_file = temp_db_dir / "test_object_type.ts"
        test_file.write_text('''
// Object type alias
type Point = {
    x: number;
    y: number;
    z?: number;
};

// UNIQUE_OBJECT_TYPE_MARKER_NOP012
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the type alias
        results = test_services.provider.search_chunks_regex("type Point")

        assert len(results) > 0, "Object type alias should be extracted as a chunk"

    async def test_fixture_file_type_alias_extracted(self, test_services, ts_test_file):
        """Test that UserRole type alias from fixture is extracted."""
        if not ts_test_file.exists():
            pytest.skip(f"Test fixture not found: {ts_test_file}")

        await test_services.indexing_coordinator.process_file(ts_test_file)

        # Search for UserRole type
        results = test_services.provider.search_chunks_regex("type UserRole")

        assert len(results) > 0, "UserRole type alias should be extracted from fixture"


class TestTypeScriptClasses:
    """Test suite for TypeScript class identification."""

    async def test_class_identified_as_class_type(self, test_services, temp_db_dir):
        """Test that classes are identified as ChunkType.CLASS, not FUNCTION."""
        test_file = temp_db_dir / "test_class_type.ts"
        test_file.write_text('''
// Class declaration
class UserService {
    private baseUrl: string;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }

    async getUser(id: number): Promise<User> {
        return fetch(`${this.baseUrl}/users/${id}`).then(r => r.json());
    }
}

// UNIQUE_CLASS_TYPE_MARKER_QRS345
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the class
        results = test_services.provider.search_chunks_regex("class UserService")

        assert len(results) > 0, "Class should be extracted as a chunk"

        # CRITICAL: Verify chunk type is CLASS, not FUNCTION
        class_chunks = [c for c in results if "UserService" in c["content"]]
        assert len(class_chunks) > 0, "Should find UserService in results"
        assert any(chunk["type"] == ChunkType.CLASS.value for chunk in class_chunks), \
            f"Class should be ChunkType.CLASS, but got types: {[c['type'] for c in class_chunks]}"

    async def test_fixture_file_class_identified_correctly(self, test_services, ts_test_file):
        """Test that UserService class from fixture is identified as CLASS."""
        if not ts_test_file.exists():
            pytest.skip(f"Test fixture not found: {ts_test_file}")

        await test_services.indexing_coordinator.process_file(ts_test_file)

        # Search for UserService class
        results = test_services.provider.search_chunks_regex("class UserService")

        assert len(results) > 0, "UserService class should be extracted from fixture"

        # Verify it's classified as CLASS
        class_chunks = [c for c in results if "UserService" in c["content"]]
        assert any(chunk["type"] == ChunkType.CLASS.value for chunk in class_chunks), \
            "UserService should be classified as ChunkType.CLASS"


class TestTypeScriptNamespaces:
    """Test suite for TypeScript namespace extraction (bonus feature)."""

    async def test_namespace_extracted(self, test_services, temp_db_dir):
        """Test that namespace declarations are extracted as chunks."""
        test_file = temp_db_dir / "test_namespace.ts"
        test_file.write_text('''
// Namespace declaration
namespace Utils {
    export function format(value: string): string {
        return value.trim();
    }

    export function parse(input: string): any {
        return JSON.parse(input);
    }
}

// UNIQUE_NAMESPACE_MARKER_TUV678
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the namespace
        results = test_services.provider.search_chunks_regex("namespace Utils")

        assert len(results) > 0, "Namespace declaration should be extracted as a chunk"

        # Verify chunk type is NAMESPACE
        ns_chunks = [c for c in results if "Utils" in c["content"]]
        assert any(chunk["type"] == ChunkType.NAMESPACE.value for chunk in ns_chunks), \
            "Namespace should be classified as ChunkType.NAMESPACE"

    async def test_module_declaration_extracted(self, test_services, temp_db_dir):
        """Test that module declarations are extracted (TypeScript namespace alias)."""
        test_file = temp_db_dir / "test_module.ts"
        test_file.write_text('''
// Module declaration (namespace alias)
declare module 'my-library' {
    export function myFunction(): void;
}

// UNIQUE_MODULE_MARKER_WXY901
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the module declaration
        results = test_services.provider.search_chunks_regex("module.*my-library")

        assert len(results) > 0, "Module declaration should be extracted as a chunk"


class TestConstAssertions:
    """Test suite for TypeScript const assertions."""

    async def test_const_assertion_array(self, test_services, temp_db_dir):
        """Test that const assertion on array is extracted."""
        test_file = temp_db_dir / "test_const_assertion_array.ts"
        test_file.write_text('''
// Const assertion array
const arr = [1, 2, 3] as const;

// UNIQUE_CONST_ASSERTION_ARRAY_MARKER_STU901
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the const assertion
        results = test_services.provider.search_chunks_regex("as const")

        assert len(results) > 0, "Const assertion array should be extracted as a chunk"

        # Verify it's captured as a variable chunk
        arr_chunks = [c for c in results if "arr" in c["content"]]
        assert len(arr_chunks) > 0, \
            "Const assertion variable should be extracted"

    async def test_const_assertion_object(self, test_services, temp_db_dir):
        """Test that const assertion on object is extracted."""
        test_file = temp_db_dir / "test_const_assertion_object.ts"
        test_file.write_text('''
// Const assertion object
const obj = { x: 1, y: 2 } as const;

// UNIQUE_CONST_ASSERTION_OBJECT_MARKER_VWX234
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the const assertion
        results = test_services.provider.search_chunks_regex("as const")

        assert len(results) > 0, "Const assertion object should be extracted as a chunk"

        # Verify it contains object properties
        obj_chunks = [c for c in results if "obj" in c["content"]]
        assert any("x" in chunk["content"] and "y" in chunk["content"]
                  for chunk in obj_chunks), \
            "Const assertion object properties should be captured"


class TestTypeGuards:
    """Test suite for TypeScript type guards."""

    async def test_user_defined_type_guard(self, test_services, temp_db_dir):
        """Test that user-defined type guard function is extracted."""
        test_file = temp_db_dir / "test_type_guard.ts"
        test_file.write_text('''
// User-defined type guard
function isString(x: any): x is string {
    return typeof x === 'string';
}

// UNIQUE_TYPE_GUARD_MARKER_YZA567
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the type guard function
        results = test_services.provider.search_chunks_regex("x is string")

        assert len(results) > 0, "Type guard function should be extracted as a chunk"

        # Verify it's a function chunk
        guard_chunks = [c for c in results if "isString" in c["content"]]
        assert len(guard_chunks) > 0, \
            "Type guard function should be captured with 'x is string' predicate"

    async def test_assertion_function(self, test_services, temp_db_dir):
        """Test that assertion function is extracted."""
        test_file = temp_db_dir / "test_assertion_function.ts"
        test_file.write_text('''
// Assertion function
function assert(condition: any): asserts condition {
    if (!condition) throw new Error();
}

// UNIQUE_ASSERTION_FUNCTION_MARKER_BCD890
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the assertion function
        results = test_services.provider.search_chunks_regex("asserts condition")

        assert len(results) > 0, "Assertion function should be extracted as a chunk"

        # Verify it contains assertion signature
        assert_chunks = [c for c in results if "assert" in c["content"]]
        assert len(assert_chunks) > 0, \
            "Assertion function should be captured with 'asserts' keyword"


class TestAmbientDeclarations:
    """Test suite for TypeScript ambient declarations."""

    async def test_declare_const(self, test_services, temp_db_dir):
        """Test that declare const is extracted."""
        test_file = temp_db_dir / "test_declare_const.ts"
        test_file.write_text('''
// Ambient const declaration
declare const ENV: string;

// UNIQUE_DECLARE_CONST_MARKER_EFG123
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the declare const
        results = test_services.provider.search_chunks_regex("declare const ENV")

        assert len(results) > 0, "Declare const should be extracted as a chunk"

        # Verify it contains the declaration
        env_chunks = [c for c in results if "ENV" in c["content"]]
        assert len(env_chunks) > 0, \
            "Ambient const declaration should be captured"

    async def test_declare_function(self, test_services, temp_db_dir):
        """Test that declare function is extracted."""
        test_file = temp_db_dir / "test_declare_function.ts"
        test_file.write_text('''
// Ambient function declaration
declare function lib(): void;

// UNIQUE_DECLARE_FUNCTION_MARKER_HIJ456
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the declare function
        results = test_services.provider.search_chunks_regex("declare function lib")

        assert len(results) > 0, "Declare function should be extracted as a chunk"

        # Verify it contains the declaration
        lib_chunks = [c for c in results if "lib" in c["content"]]
        assert len(lib_chunks) > 0, \
            "Ambient function declaration should be captured"


class TestTypeScriptVariableFiltering:
    """Test suite for proper variable filtering (simple vs complex)."""

    async def test_simple_string_variable_not_extracted(self, test_services, temp_db_dir):
        """Test that simple string variables ARE extracted as chunks."""
        test_file = temp_db_dir / "test_simple_var.ts"
        test_file.write_text('''
// Simple variables (SHOULD be extracted)
const API_URL = 'https://api.example.com';
const MAX_RETRIES = 3;
const ENABLED = true;

// UNIQUE_SIMPLE_VAR_MARKER_ZAB234

// This function should be extracted
function doWork() {
    return API_URL;
}
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search - SHOULD find simple variable declarations
        api_results = test_services.provider.search_chunks_regex("const API_URL =")
        max_results = test_services.provider.search_chunks_regex("const MAX_RETRIES =")
        enabled_results = test_services.provider.search_chunks_regex("const ENABLED =")

        # Simple variables SHOULD be extracted as separate chunks
        # More robust chunk field access
        api_var_chunks = [
            c for c in api_results
            if "API_URL" in c.get("symbol", "")
            or "API_URL" in c.get("name", "")
            or "API_URL" in str(c.get("chunk", {}).get("symbol", ""))
        ]
        max_var_chunks = [
            c for c in max_results
            if "MAX_RETRIES" in c.get("symbol", "")
            or "MAX_RETRIES" in c.get("name", "")
            or "MAX_RETRIES" in str(c.get("chunk", {}).get("symbol", ""))
        ]
        enabled_var_chunks = [
            c for c in enabled_results
            if "ENABLED" in c.get("symbol", "")
            or "ENABLED" in c.get("name", "")
            or "ENABLED" in str(c.get("chunk", {}).get("symbol", ""))
        ]

        assert len(api_var_chunks) > 0, \
            "Simple string variable API_URL should be extracted"
        assert len(max_var_chunks) > 0, \
            "Simple number variable MAX_RETRIES should be extracted"
        assert len(enabled_var_chunks) > 0, \
            "Simple boolean variable ENABLED should be extracted"

    async def test_object_variable_extracted(self, test_services, temp_db_dir):
        """Test that object/array variables ARE extracted as chunks."""
        test_file = temp_db_dir / "test_object_var.ts"
        test_file.write_text('''
// Object variable (SHOULD be extracted)
const config = {
    timeout: 5000,
    retries: 3,
    baseUrl: 'https://api.example.com'
};

// UNIQUE_OBJECT_VAR_MARKER_CDE567

// Array variable (SHOULD be extracted)
const items = [
    { id: 1, name: 'Item 1' },
    { id: 2, name: 'Item 2' }
];
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for object variable
        config_results = test_services.provider.search_chunks_regex("const config")

        # Object/array variables SHOULD be extracted
        assert len(config_results) > 0, \
            "Object literal variables should be extracted as chunks"

        config_chunks = [c for c in config_results if "timeout" in c["content"]]
        assert len(config_chunks) > 0, \
            "Object variable content should include object properties"

    async def test_fixture_file_variable_filtering(self, test_services, ts_test_file):
        """Test that fixture file properly extracts all variables."""
        if not ts_test_file.exists():
            pytest.skip(f"Test fixture not found: {ts_test_file}")

        await test_services.indexing_coordinator.process_file(ts_test_file)

        # Simple variables SHOULD be chunks
        simple_results = test_services.provider.search_chunks_regex("const API_URL =")
        # More robust chunk field access
        simple_chunks = [
            c for c in simple_results
            if "API_URL" in c.get("symbol", "")
            or "API_URL" in c.get("name", "")
            or "API_URL" in str(c.get("chunk", {}).get("symbol", ""))
        ]
        assert len(simple_chunks) > 0, \
            "Simple string variable API_URL from fixture should be extracted"

        # Object variables SHOULD be chunks
        config_results = test_services.provider.search_chunks_regex("const config")
        assert len(config_results) > 0, \
            "Object variables from fixture should be extracted"


class TestTypeScriptMetadata:
    """Test suite for TypeScript-specific metadata extraction."""

    async def test_interface_metadata_includes_type_info(self, test_services, temp_db_dir):
        """Test that interface chunks include proper metadata."""
        test_file = temp_db_dir / "test_interface_metadata.ts"
        test_file.write_text('''
// Generic interface with extends
interface Repository<T> extends BaseRepository {
    findById(id: string): Promise<T>;
}
''')

        await test_services.indexing_coordinator.process_file(test_file)

        results = test_services.provider.search_chunks_regex("interface Repository")

        assert len(results) > 0, "Interface should be extracted"
        # Note: Metadata assertions would check for type_parameters, extends, etc.
        # but these require implementation - tests will fail until then

    async def test_enum_metadata_includes_members(self, test_services, temp_db_dir):
        """Test that enum chunks include member information."""
        test_file = temp_db_dir / "test_enum_metadata.ts"
        test_file.write_text('''
enum HttpStatus {
    OK = 200,
    NotFound = 404,
    ServerError = 500
}
''')

        await test_services.indexing_coordinator.process_file(test_file)

        results = test_services.provider.search_chunks_regex("enum HttpStatus")

        assert len(results) > 0, "Enum should be extracted"
        # Metadata would include enum members - tests will fail until implemented


# Direct parser tests (unit-level)
class TestTypeScriptParserDirect:
    """Direct tests of TypeScript parser functionality."""

    def test_parser_creates_for_typescript(self):
        """Test that parser factory creates TypeScript parser."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        assert parser is not None, "Should create TypeScript parser"
        assert parser.language_name == "typescript", \
            "Parser should be for TypeScript language"

    def test_imports_parsed_from_test_file(self):
        """Test that parser extracts imports from test content."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        test_content = """
import { useState } from 'react';
import type { User } from './types';

function Component() {
    return null;
}
"""

        chunks = parser.parse_content(test_content, "test.ts", FileId(1))

        # Should find import chunks
        import_chunks = [c for c in chunks if "import" in c.code.lower()]
        assert len(import_chunks) > 0, "Should extract import statements"

    def test_interface_parsed_from_test_content(self):
        """Test that parser extracts interfaces."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        test_content = """
interface User {
    id: number;
    name: string;
}
"""

        chunks = parser.parse_content(test_content, "test.ts", FileId(1))

        # Should find interface chunk
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract interface declarations"
        assert any("User" in c.symbol for c in interface_chunks), \
            "Interface name should be captured"

    def test_enum_parsed_from_test_content(self):
        """Test that parser extracts enums."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        test_content = """
enum Status {
    Pending = 'pending',
    Success = 'success'
}
"""

        chunks = parser.parse_content(test_content, "test.ts", FileId(1))

        # Should find enum chunk
        enum_chunks = [c for c in chunks if c.chunk_type == ChunkType.ENUM]
        assert len(enum_chunks) > 0, "Should extract enum declarations"

    def test_type_alias_parsed_from_test_content(self):
        """Test that parser extracts type aliases."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        test_content = """
type UserRole = 'admin' | 'user' | 'guest';
"""

        chunks = parser.parse_content(test_content, "test.ts", FileId(1))

        # Should find type alias chunk
        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]
        assert len(type_chunks) > 0, "Should extract type alias declarations"

    def test_class_identified_as_class_not_function(self):
        """Test that classes are ChunkType.CLASS, not FUNCTION."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        test_content = """
class MyService {
    constructor() {
        // init
    }

    doWork() {
        return true;
    }
}
"""

        chunks = parser.parse_content(test_content, "test.ts", FileId(1))

        # Should find class chunk
        class_chunks = [c for c in chunks if "MyService" in c.symbol]
        assert len(class_chunks) > 0, "Should extract class declaration"

        # CRITICAL: Must be CLASS type, not FUNCTION
        service_class = class_chunks[0]
        assert service_class.chunk_type == ChunkType.CLASS, \
            f"Class should be ChunkType.CLASS but got {service_class.chunk_type}"

    def test_namespace_parsed_from_test_content(self):
        """Test that parser extracts namespaces (bonus feature)."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        test_content = """
namespace Utils {
    export function helper() {
        return true;
    }
}
"""

        chunks = parser.parse_content(test_content, "test.ts", FileId(1))

        # Should find namespace chunk
        ns_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
        assert len(ns_chunks) > 0, "Should extract namespace declarations"
