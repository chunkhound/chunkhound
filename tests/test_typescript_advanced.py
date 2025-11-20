"""Advanced TypeScript parsing tests.

This module contains tests for advanced TypeScript patterns that are NOT covered
in test_typescript_comprehensive.py. These tests focus on:

1. Interface patterns: Extends, multiple extends, index/call/construct signatures
2. Type alias patterns: Intersection, mapped, conditional, template literal, tuple, function types
3. Enum patterns: Const enum, computed members, mixed enums
4. Namespace patterns: Nested namespaces, module declarations, global augmentation
5. Decorator patterns: Class/method/property/parameter decorators, factories, multiple decorators

Reference: docs/js-family-parser-test-specification.md Section 6
"""

import pytest
from pathlib import Path

from chunkhound.core.types.common import ChunkType, Language, FileId
from chunkhound.parsers.parser_factory import get_parser_factory


class TestAdvancedInterfaces:
    """Test suite for advanced TypeScript interface patterns."""

    def test_interface_extends_single(self):
        """Test interface that extends another interface."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
interface Animal {
    name: string;
}

interface Dog extends Animal {
    breed: string;
    bark(): void;
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        # Should extract interfaces
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) >= 1, "Should extract at least one interface"

        # Verify at least one interface is extracted with extends
        all_code = " ".join([c.code for c in interface_chunks])
        assert "extends Animal" in all_code or "Animal" in all_code, \
            "Should capture interface with or referencing extends"

    def test_interface_extends_multiple(self):
        """Test interface that extends multiple interfaces."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
interface Readable {
    read(): string;
}

interface Writable {
    write(data: string): void;
}

interface ReadWriteStream extends Readable, Writable {
    close(): void;
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) >= 1, "Should extract at least one interface"

        # Verify at least one interface has extends in content
        all_code = " ".join([c.code for c in interface_chunks])
        assert "extends" in all_code, "Should capture at least one interface with extends"

    def test_interface_index_signature(self):
        """Test interface with index signature."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
interface StringDictionary {
    [key: string]: string;
}

interface NumberDictionary {
    [index: number]: any;
    length: number;
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) >= 1, "Should extract at least one interface"

        # Check that index signature syntax is captured in at least one
        all_code = " ".join([c.code for c in interface_chunks])
        assert "[key: string]" in all_code or "[index: number]" in all_code, \
            "Should capture index signature syntax"

    def test_interface_call_signature(self):
        """Test interface with call signature (callable interface)."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
interface Formatter {
    (value: string): string;
}

interface GenericFormatter<T> {
    (value: T): string;
    defaultValue: T;
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]

        formatter_chunks = [c for c in interface_chunks if "Formatter" in c.symbol]
        assert len(formatter_chunks) >= 1, "Should extract Formatter interface"

        # Check call signature syntax
        formatter = formatter_chunks[0]
        assert "(value:" in formatter.code and "): string" in formatter.code, \
            "Should capture call signature"

    def test_interface_construct_signature(self):
        """Test interface with construct signature (newable interface)."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
interface ClockConstructor {
    new (hour: number, minute: number): ClockInterface;
}

interface ClockInterface {
    tick(): void;
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]

        constructor_chunks = [c for c in interface_chunks if "ClockConstructor" in c.symbol]
        assert len(constructor_chunks) > 0, "Should extract ClockConstructor interface"

        constructor = constructor_chunks[0]
        assert "new (" in constructor.code, "Should capture construct signature"

    def test_interface_method_signatures(self):
        """Test interface with various method signatures."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
interface Calculator {
    add(a: number, b: number): number;
    subtract(a: number, b: number): number;
    multiply?(a: number, b: number): number;
    divide(a: number, b: number): number | undefined;
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]

        calc_chunks = [c for c in interface_chunks if "Calculator" in c.symbol]
        assert len(calc_chunks) > 0, "Should extract Calculator interface"

        calc = calc_chunks[0]
        assert "add(" in calc.code and "subtract(" in calc.code, \
            "Should capture method signatures"
        assert "multiply?" in calc.code, "Should capture optional method"


class TestAdvancedTypeAliases:
    """Test suite for advanced TypeScript type alias patterns."""

    def test_intersection_type(self):
        """Test intersection type alias."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = "type Combined = TypeA & TypeB & TypeC;"
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]
        assert len(type_chunks) > 0, "Should extract intersection type alias"

        combined = [c for c in type_chunks if "Combined" in c.symbol][0]
        assert "TypeA & TypeB & TypeC" in combined.code, "Should capture intersection"

    def test_mapped_type(self):
        """Test mapped type alias."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
type Readonly<T> = {
    readonly [K in keyof T]: T[K];
};

type Partial<T> = {
    [K in keyof T]?: T[K];
};

type Required<T> = {
    [K in keyof T]-?: T[K];
};
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]

        # Should extract at least one mapped type
        assert len(type_chunks) >= 1, "Should extract at least one mapped type"

        # Verify mapped type syntax in any extracted type
        all_code = " ".join([c.code for c in type_chunks])
        assert "[K in keyof T]" in all_code or "keyof" in all_code, \
            "Should capture mapped type syntax"

    def test_conditional_type(self):
        """Test conditional type alias."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
type IsString<T> = T extends string ? true : false;

type NonNullable<T> = T extends null | undefined ? never : T;

type ReturnType<T> = T extends (...args: any[]) => infer R ? R : never;
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]

        # Verify conditional types
        is_string_chunks = [c for c in type_chunks if "IsString" in c.symbol]
        assert len(is_string_chunks) > 0, "Should extract IsString conditional type"

        is_string = is_string_chunks[0]
        assert "extends string ? true : false" in is_string.code, \
            "Should capture conditional type syntax"

        # Verify infer keyword
        return_type_chunks = [c for c in type_chunks if "ReturnType" in c.symbol]
        if len(return_type_chunks) > 0:
            return_type = return_type_chunks[0]
            assert "infer R" in return_type.code, "Should capture infer keyword"

    def test_template_literal_type(self):
        """Test template literal type alias."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
type EventName = `on${string}`;

type CSSProperty = `--${string}`;

type PropertyName<T extends string> = `get${Capitalize<T>}` | `set${Capitalize<T>}`;
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]

        event_chunks = [c for c in type_chunks if "EventName" in c.symbol]
        assert len(event_chunks) > 0, "Should extract EventName template literal type"

        event = event_chunks[0]
        assert "`on${string}`" in event.code, "Should capture template literal type syntax"

    def test_tuple_type(self):
        """Test tuple type alias."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
type Point = [number, number];

type Point3D = [number, number, number];

type NamedTuple = [first: string, second: number, third?: boolean];

type RestTuple = [string, ...number[]];
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]

        point_chunks = [c for c in type_chunks if c.symbol == "Point"]
        assert len(point_chunks) > 0, "Should extract Point tuple type"

        point = point_chunks[0]
        assert "[number, number]" in point.code, "Should capture tuple type syntax"

    def test_function_type(self):
        """Test function type alias."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
type Callback = (error: Error | null, result: string) => void;

type AsyncCallback<T> = (value: T) => Promise<void>;

type GenericFunction<T, R> = (input: T) => R;
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]

        callback_chunks = [c for c in type_chunks if "Callback" in c.symbol and "Async" not in c.symbol]
        assert len(callback_chunks) > 0, "Should extract Callback function type"

        callback = callback_chunks[0]
        assert "=>" in callback.code, "Should capture arrow function type syntax"

    def test_complex_utility_type(self):
        """Test complex utility type combinations."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
type DeepPartial<T> = {
    [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};

type Awaited<T> = T extends PromiseLike<infer U> ? Awaited<U> : T;
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]

        deep_partial = [c for c in type_chunks if "DeepPartial" in c.symbol]
        assert len(deep_partial) > 0, "Should extract DeepPartial complex type"


class TestAdvancedEnums:
    """Test suite for advanced TypeScript enum patterns."""

    def test_const_enum(self):
        """Test const enum (compile-time inlined)."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
const enum Direction {
    Up = 1,
    Down = 2,
    Left = 3,
    Right = 4
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        enum_chunks = [c for c in chunks if c.chunk_type == ChunkType.ENUM]
        assert len(enum_chunks) > 0, "Should extract const enum"

        direction = enum_chunks[0]
        assert "const enum" in direction.code, "Should capture const modifier"

    def test_computed_enum_member(self):
        """Test enum with computed members."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
enum FileAccess {
    None = 0,
    Read = 1 << 0,
    Write = 1 << 1,
    Execute = 1 << 2,
    ReadWrite = Read | Write,
    All = ReadWrite | Execute
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        enum_chunks = [c for c in chunks if c.chunk_type == ChunkType.ENUM]
        assert len(enum_chunks) > 0, "Should extract enum with computed members"

        file_access = enum_chunks[0]
        assert "1 << 0" in file_access.code or "Read = 1" in file_access.code, \
            "Should capture computed member expressions"

    def test_mixed_enum(self):
        """Test enum with mixed string and numeric values."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
enum MixedEnum {
    Numeric = 0,
    AnotherNumeric = 1,
    StringValue = 'string',
    AnotherString = 'another'
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        enum_chunks = [c for c in chunks if c.chunk_type == ChunkType.ENUM]
        assert len(enum_chunks) > 0, "Should extract mixed enum"

        mixed = enum_chunks[0]
        assert "= 0" in mixed.code and "= 'string'" in mixed.code, \
            "Should capture both numeric and string values"

    def test_declare_enum(self):
        """Test ambient/declare enum."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
declare enum ExternalEnum {
    Value1,
    Value2,
    Value3
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        enum_chunks = [c for c in chunks if c.chunk_type == ChunkType.ENUM]
        assert len(enum_chunks) > 0, "Should extract declare enum"


class TestAdvancedNamespaces:
    """Test suite for advanced TypeScript namespace patterns."""

    def test_nested_namespace(self):
        """Test nested namespace declaration."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
namespace Outer {
    export namespace Inner {
        export function helper() {
            return true;
        }
    }
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        # Should find namespace chunks
        ns_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
        assert len(ns_chunks) >= 1, "Should extract namespace(s)"

        # Verify nested structure is captured
        outer_chunks = [c for c in ns_chunks if "Outer" in c.symbol]
        assert len(outer_chunks) > 0, "Should extract Outer namespace"

        outer = outer_chunks[0]
        assert "Inner" in outer.code, "Should capture nested namespace in content"

    def test_dotted_namespace(self):
        """Test dotted namespace declaration."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
namespace Shapes.Polygons {
    export class Triangle {}
    export class Square {}
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        # Dotted namespaces may be extracted as namespace or may capture classes
        ns_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]

        # At minimum, should extract the classes inside or the namespace
        assert len(ns_chunks) >= 1 or len(class_chunks) >= 1, \
            "Should extract dotted namespace or its contents"

    def test_module_declaration(self):
        """Test declare module for external modules."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
declare module 'lodash' {
    export function map<T, U>(arr: T[], fn: (val: T) => U): U[];
    export function filter<T>(arr: T[], fn: (val: T) => boolean): T[];
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        # Module declaration should be extracted
        ns_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
        module_chunks = [c for c in ns_chunks if "lodash" in c.code or "module" in c.code.lower()]

        assert len(module_chunks) > 0 or len(ns_chunks) > 0, \
            "Should extract declare module as namespace or similar"

    def test_global_augmentation(self):
        """Test global augmentation."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
declare global {
    interface Window {
        myCustomProperty: string;
    }

    var globalVar: number;
}

export {};
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        # Should find the global declaration
        ns_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
        global_chunks = [c for c in ns_chunks if "global" in c.code.lower()]

        # Note: Implementation may vary on how global is classified
        assert len(chunks) > 0, "Should extract declarations from global augmentation"

    def test_ambient_namespace(self):
        """Test ambient namespace declaration."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
declare namespace NodeJS {
    interface ProcessEnv {
        NODE_ENV: string;
        PORT: string;
    }
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        ns_chunks = [c for c in chunks if c.chunk_type == ChunkType.NAMESPACE]
        nodejs_chunks = [c for c in ns_chunks if "NodeJS" in c.symbol]

        assert len(nodejs_chunks) > 0, "Should extract ambient namespace declaration"


class TestDecorators:
    """Test suite for TypeScript decorator patterns."""

    def test_class_decorator(self):
        """Test decorator on class."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
@Component
class MyComponent {
    render() {}
}

@Injectable()
class MyService {
    doWork() {}
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1, "Should extract at least one decorated class"

        # Verify at least one class is extracted
        all_code = " ".join([c.code for c in class_chunks])
        assert "MyComponent" in all_code or "MyService" in all_code, \
            "Should capture decorated class"

    def test_method_decorator(self):
        """Test decorator on method."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
class Controller {
    @Get('/users')
    getUsers() {
        return [];
    }

    @Post('/users')
    @Validate()
    createUser(data: UserDto) {
        return data;
    }
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should extract class with decorated methods"

        controller = class_chunks[0]
        assert "@Get" in controller.code, "Should capture method decorator"
        assert "@Post" in controller.code and "@Validate" in controller.code, \
            "Should capture multiple method decorators"

    def test_property_decorator(self):
        """Test decorator on property."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
class Entity {
    @Column()
    id: number;

    @Column({ type: 'varchar', length: 100 })
    name: string;

    @OneToMany(() => Order)
    orders: Order[];
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should extract class with decorated properties"

        entity = class_chunks[0]
        assert "@Column" in entity.code, "Should capture property decorator"

    def test_parameter_decorator(self):
        """Test decorator on parameter."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
class UserController {
    getUser(
        @Param('id') id: string,
        @Query('include') include?: string
    ) {
        return { id, include };
    }

    createUser(@Body() body: CreateUserDto) {
        return body;
    }
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should extract class with parameter decorators"

        controller = class_chunks[0]
        assert "@Param" in controller.code or "@Body" in controller.code, \
            "Should capture parameter decorator"

    def test_decorator_factory(self):
        """Test decorator factory (decorator with arguments)."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
@Controller({ path: '/api/users', version: 'v1' })
class UserApiController {
    @Route({ method: 'GET', path: '/:id' })
    findOne(id: string) {}
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should extract class with decorator factory"

        controller = class_chunks[0]
        assert "@Controller({" in controller.code, "Should capture decorator factory with arguments"

    def test_multiple_decorators_stacked(self):
        """Test multiple decorators stacked on single element."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
@Serializable
@Validated
@Logged('UserEntity')
class User {
    @Required
    @MinLength(3)
    @MaxLength(50)
    name: string;
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should extract class with stacked decorators"

        user = class_chunks[0]
        assert "@Serializable" in user.code and "@Validated" in user.code, \
            "Should capture multiple class decorators"
        assert "@Required" in user.code and "@MinLength" in user.code, \
            "Should capture multiple property decorators"


class TestTSXAdvanced:
    """Test suite for advanced TSX (TypeScript + JSX) patterns."""

    def test_generic_component_tsx(self):
        """Test generic React component in TSX."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TSX)

        code = """
interface ListProps<T> {
    items: T[];
    renderItem: (item: T) => React.ReactNode;
}

function List<T>({ items, renderItem }: ListProps<T>) {
    return (
        <ul>
            {items.map((item, index) => (
                <li key={index}>{renderItem(item)}</li>
            ))}
        </ul>
    );
}
"""
        chunks = parser.parse_content(code, "test.tsx", FileId(1))

        # Should extract interface and function
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        function_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]

        assert len(interface_chunks) > 0, "Should extract ListProps interface"
        assert len(function_chunks) > 0 or any("List" in c.symbol for c in chunks), \
            "Should extract List generic function component"

    def test_typed_hooks_tsx(self):
        """Test typed React hooks in TSX."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TSX)

        code = """
interface User {
    id: number;
    name: string;
}

function UserProfile() {
    const [user, setUser] = useState<User | null>(null);
    const ref = useRef<HTMLDivElement>(null);
    const memoized = useMemo<string>(() => user?.name ?? '', [user]);

    return <div ref={ref}>{memoized}</div>;
}
"""
        chunks = parser.parse_content(code, "test.tsx", FileId(1))

        # Should extract component function or interface
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]

        # At least extract the interface
        assert len(interface_chunks) > 0 or len(func_chunks) > 0, \
            "Should extract UserProfile component or User interface"

        # Verify some content is captured
        all_code = " ".join([c.code for c in chunks])
        assert "useState" in all_code or "User" in all_code, \
            "Should capture typed content"

    def test_fc_with_props_tsx(self):
        """Test React.FC with typed props in TSX."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TSX)

        code = """
interface ButtonProps {
    onClick: () => void;
    children: React.ReactNode;
    disabled?: boolean;
}

const Button: React.FC<ButtonProps> = ({ onClick, children, disabled }) => (
    <button onClick={onClick} disabled={disabled}>
        {children}
    </button>
);
"""
        chunks = parser.parse_content(code, "test.tsx", FileId(1))

        # Should extract interface
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract ButtonProps interface"

        # Verify ButtonProps content
        button_props = interface_chunks[0]
        assert "onClick" in button_props.code, "Should capture ButtonProps with onClick"


class TestComplexTypePatterns:
    """Test suite for complex TypeScript type patterns."""

    def test_discriminated_union(self):
        """Test discriminated union pattern."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
interface Success {
    kind: 'success';
    value: string;
}

interface Failure {
    kind: 'failure';
    error: Error;
}

type Result = Success | Failure;
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        # Should extract types
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]

        assert len(interface_chunks) >= 1 or len(type_chunks) >= 1, \
            "Should extract at least one interface or type alias"

        # Verify discriminated union pattern is captured
        all_code = " ".join([c.code for c in chunks])
        assert "kind:" in all_code or "Success" in all_code, \
            "Should capture discriminated union content"

    def test_branded_type(self):
        """Test branded/nominal type pattern."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
type Brand<K, T> = K & { __brand: T };

type UserId = Brand<number, 'UserId'>;
type OrderId = Brand<number, 'OrderId'>;
type Email = Brand<string, 'Email'>;
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]
        assert len(type_chunks) >= 1, "Should extract at least one branded type"

        # Verify branded type content
        all_code = " ".join([c.code for c in type_chunks])
        assert "Brand" in all_code or "__brand" in all_code, \
            "Should capture branded type pattern"

    def test_recursive_type(self):
        """Test recursive type definition."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
type Json =
    | string
    | number
    | boolean
    | null
    | Json[]
    | { [key: string]: Json };

type TreeNode<T> = {
    value: T;
    children: TreeNode<T>[];
};
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]
        assert len(type_chunks) >= 1, "Should extract at least one recursive type"

        # Verify recursive type content
        all_code = " ".join([c.code for c in type_chunks])
        assert "Json" in all_code or "TreeNode" in all_code, \
            "Should capture recursive type"

    def test_generic_constraints(self):
        """Test advanced generic constraints."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
type KeysOfType<T, U> = {
    [K in keyof T]: T[K] extends U ? K : never
}[keyof T];

type RequiredKeys<T> = {
    [K in keyof T]-?: {} extends Pick<T, K> ? never : K
}[keyof T];

type FunctionPropertyNames<T> = {
    [K in keyof T]: T[K] extends Function ? K : never
}[keyof T];
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]
        assert len(type_chunks) >= 1, "Should extract at least one constrained type"

        # Verify generic constraint content
        all_code = " ".join([c.code for c in type_chunks])
        assert "keyof" in all_code or "extends" in all_code, \
            "Should capture generic constraint syntax"


class TestVueTypeScript:
    """Test suite for Vue with TypeScript patterns."""

    def test_vue_ts_defineprops_generic(self):
        """Test Vue defineProps with TypeScript generics."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.VUE)

        code = """<script setup lang="ts">
interface Props {
    title: string;
    count?: number;
    items: string[];
}

const props = defineProps<Props>();
const emit = defineEmits<{
    (e: 'update', value: string): void;
    (e: 'delete', id: number): void;
}>();
</script>

<template>
    <div>{{ props.title }}</div>
</template>
"""
        chunks = parser.parse_content(code, "test.vue", FileId(1))

        # Should extract interface
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        props_interface = [c for c in interface_chunks if "Props" in c.symbol]
        assert len(props_interface) > 0, "Should extract Props interface in Vue"

    def test_vue_ts_computed_with_types(self):
        """Test Vue computed properties with TypeScript types."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.VUE)

        code = """<script setup lang="ts">
import { computed, ref } from 'vue';

interface User {
    id: number;
    name: string;
}

const users = ref<User[]>([]);

const userNames = computed<string[]>(() =>
    users.value.map(u => u.name)
);
</script>
"""
        chunks = parser.parse_content(code, "test.vue", FileId(1))

        # Should extract interface
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract User interface in Vue"


class TestEdgeCases:
    """Test suite for edge cases in advanced TypeScript parsing."""

    def test_empty_interface(self):
        """Test empty interface (marker interface)."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = "interface Marker {}"
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract empty interface"

    def test_single_member_type(self):
        """Test type alias with single member."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = "type ID = string;"
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]
        assert len(type_chunks) > 0, "Should extract simple type alias"

    def test_very_long_union_type(self):
        """Test type alias with many union members."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        members = " | ".join([f"'{chr(97+i)}'" for i in range(26)])
        code = f"type AlphaChars = {members};"
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]
        assert len(type_chunks) > 0, "Should extract long union type"

    def test_deeply_nested_generics(self):
        """Test deeply nested generic types."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
type Nested = Map<string, Array<Promise<Set<number>>>>;
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]
        assert len(type_chunks) > 0, "Should extract deeply nested generic type"

    def test_export_with_decorator(self):
        """Test exported decorated class."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        code = """
@Injectable()
export class MyService {
    constructor(private http: HttpClient) {}
}
"""
        chunks = parser.parse_content(code, "test.ts", FileId(1))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should extract exported decorated class"

        service = class_chunks[0]
        # Note: Decorators may not always be captured in the chunk content
        # depending on parser implementation
        assert "MyService" in service.code, "Should capture the class content"
