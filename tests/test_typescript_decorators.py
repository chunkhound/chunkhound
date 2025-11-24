"""TypeScript decorator parsing tests (TDD Red Phase).

This module contains tests that verify TypeScript parser extracts decorator
patterns correctly. These tests are designed to FAIL with the current
implementation and will pass once decorator support is complete.

Decorator patterns being tested:
1. Class decorators (simple and factory patterns)
2. Method decorators
3. Property decorators
4. Parameter decorators
5. Multiple decorators on single target
6. Decorator factories with configuration objects

TypeScript decorators are special declarations that can be attached to classes,
methods, properties, and parameters. They use the @expression syntax and are
widely used in frameworks like Angular, NestJS, and TypeORM.
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


class TestClassDecorators:
    """Test suite for TypeScript class decorator extraction."""

    async def test_class_decorator(self, test_services, temp_db_dir):
        """Test that simple class decorators are extracted in class chunks.

        Example: @sealed class MyClass {}

        Decorators should appear in the class chunk's content and be searchable.
        """
        test_file = temp_db_dir / "test_class_decorator.ts"
        test_file.write_text('''
// Class decorator
@sealed
class MyClass {
    constructor() {
        this.value = 0;
    }

    getValue() {
        return this.value;
    }
}

// UNIQUE_CLASS_DECORATOR_MARKER_ABC123
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the decorator
        results = test_services.provider.search_chunks_regex("@sealed")

        assert len(results) > 0, "Class decorator should be captured in class chunk"

        # Verify decorator appears with class
        decorator_chunks = [c for c in results if "MyClass" in c["content"]]
        assert len(decorator_chunks) > 0, \
            "Decorator should be part of class chunk content"

        # Verify it's still classified as a class
        assert any(chunk["type"] == ChunkType.CLASS.value for chunk in decorator_chunks), \
            "Decorated class should still be classified as ChunkType.CLASS"

    async def test_decorator_factory(self, test_services, temp_db_dir):
        """Test that decorator factories with arguments are extracted.

        Example: @component({ selector: 'app' }) class MyComponent {}

        Decorator factories are functions that return decorators, commonly used
        in Angular and other frameworks to pass configuration.
        """
        test_file = temp_db_dir / "test_decorator_factory.ts"
        test_file.write_text('''
// Decorator factory with configuration object
@component({
    selector: 'app-root',
    templateUrl: './app.component.html'
})
class MyComponent {
    title = 'My App';

    ngOnInit() {
        console.log('Component initialized');
    }
}

// UNIQUE_DECORATOR_FACTORY_MARKER_DEF456
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the decorator factory
        results = test_services.provider.search_chunks_regex("@component")

        assert len(results) > 0, "Decorator factory should be captured in class chunk"

        # Verify configuration object is included
        config_chunks = [c for c in results if "selector" in c["content"]]
        assert len(config_chunks) > 0, \
            "Decorator factory configuration should be captured in content"

        # Verify class name is present
        component_chunks = [c for c in results if "MyComponent" in c["content"]]
        assert len(component_chunks) > 0, \
            "Decorated class should include decorator in chunk"

    async def test_multiple_decorators(self, test_services, temp_db_dir):
        """Test that multiple decorators on a class are all extracted.

        Example: @decorator1 @decorator2 class MyClass {}

        TypeScript allows stacking multiple decorators, which are applied
        bottom-to-top (reverse order from source).
        """
        test_file = temp_db_dir / "test_multiple_decorators.ts"
        test_file.write_text('''
// Multiple decorators on single class
@injectable()
@singleton
@logged
class UserService {
    constructor(private api: ApiClient) {}

    async getUser(id: number) {
        return this.api.get(`/users/${id}`);
    }
}

// UNIQUE_MULTIPLE_DECORATORS_MARKER_GHI789
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for the class
        results = test_services.provider.search_chunks_regex("class UserService")

        assert len(results) > 0, "Decorated class should be extracted"

        # Verify all decorators are present
        service_chunks = [c for c in results if "UserService" in c["content"]]
        assert len(service_chunks) > 0, "Should find UserService chunks"

        # Check for all three decorators
        content = " ".join(c["content"] for c in service_chunks)
        assert "@injectable" in content, "First decorator should be captured"
        assert "@singleton" in content, "Second decorator should be captured"
        assert "@logged" in content, "Third decorator should be captured"


class TestMethodDecorators:
    """Test suite for TypeScript method decorator extraction."""

    async def test_method_decorator(self, test_services, temp_db_dir):
        """Test that method decorators are extracted in method chunks.

        Example: class C { @logged method() {} }

        Method decorators are applied to method declarations and can be used
        for logging, validation, caching, etc.
        """
        test_file = temp_db_dir / "test_method_decorator.ts"
        test_file.write_text('''
// Method decorator
class Calculator {
    @memoize
    calculate(x: number, y: number): number {
        return x + y;
    }

    @validate
    @logged
    divide(x: number, y: number): number {
        if (y === 0) throw new Error('Division by zero');
        return x / y;
    }
}

// UNIQUE_METHOD_DECORATOR_MARKER_JKL012
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for method decorator
        results = test_services.provider.search_chunks_regex("@memoize")

        assert len(results) > 0, "Method decorator should be captured"

        # Verify decorator appears with method
        calc_chunks = [c for c in results if "calculate" in c["content"]]
        assert len(calc_chunks) > 0, \
            "Method decorator should be part of method chunk content"

        # Check multiple decorators on divide method
        divide_results = test_services.provider.search_chunks_regex("@validate")
        assert len(divide_results) > 0, "Should capture multiple method decorators"

        divide_chunks = [c for c in divide_results if "divide" in c["content"]]
        assert any("@logged" in c["content"] for c in divide_chunks), \
            "Multiple method decorators should be captured together"


class TestPropertyDecorators:
    """Test suite for TypeScript property decorator extraction."""

    async def test_property_decorator(self, test_services, temp_db_dir):
        """Test that property decorators are extracted in class chunks.

        Example: class C { @observable prop = 1 }

        Property decorators are applied to class properties and are commonly
        used in frameworks like MobX for reactive state management.
        """
        test_file = temp_db_dir / "test_property_decorator.ts"
        test_file.write_text('''
// Property decorator
class TodoStore {
    @observable
    todos: Todo[] = [];

    @observable
    @validate
    currentUser: User | null = null;

    @computed
    get completedTodos() {
        return this.todos.filter(t => t.completed);
    }
}

// UNIQUE_PROPERTY_DECORATOR_MARKER_MNO345
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for property decorator
        results = test_services.provider.search_chunks_regex("@observable")

        assert len(results) > 0, "Property decorator should be captured"

        # Verify decorator appears with property
        todo_chunks = [c for c in results if "todos" in c["content"] or "TodoStore" in c["content"]]
        assert len(todo_chunks) > 0, \
            "Property decorator should be captured in class chunk"

        # Check for getter decorator
        computed_results = test_services.provider.search_chunks_regex("@computed")
        assert len(computed_results) > 0, "Getter decorator should be captured"


class TestParameterDecorators:
    """Test suite for TypeScript parameter decorator extraction."""

    async def test_parameter_decorator(self, test_services, temp_db_dir):
        """Test that parameter decorators are extracted in method chunks.

        Example: class C { method(@inject('service') param) {} }

        Parameter decorators are applied to method/constructor parameters and
        are heavily used in dependency injection frameworks like Angular and NestJS.
        """
        test_file = temp_db_dir / "test_parameter_decorator.ts"
        test_file.write_text('''
// Parameter decorator
class AuthController {
    constructor(
        @inject('UserService') private userService: UserService,
        @inject('Logger') private logger: Logger
    ) {}

    async login(
        @body() credentials: LoginDto,
        @request() req: Request
    ): Promise<AuthToken> {
        this.logger.info('Login attempt');
        return this.userService.authenticate(credentials);
    }

    async register(
        @body() data: RegisterDto,
        @query('referral') referralCode?: string
    ): Promise<User> {
        return this.userService.create(data, referralCode);
    }
}

// UNIQUE_PARAMETER_DECORATOR_MARKER_PQR678
''')

        await test_services.indexing_coordinator.process_file(test_file)

        # Search for constructor parameter decorator
        results = test_services.provider.search_chunks_regex("@inject")

        assert len(results) > 0, "Parameter decorator should be captured"

        # Verify decorator appears in constructor or class
        inject_chunks = [c for c in results if "UserService" in c["content"] or "AuthController" in c["content"]]
        assert len(inject_chunks) > 0, \
            "Constructor parameter decorators should be captured"

        # Check for method parameter decorators
        body_results = test_services.provider.search_chunks_regex("@body")
        assert len(body_results) > 0, "Method parameter decorators should be captured"

        # Verify decorators appear in login method
        login_chunks = [c for c in body_results if "login" in c["content"] or "credentials" in c["content"]]
        assert len(login_chunks) > 0, \
            "Method parameter decorators should be in method chunks"

        # Check for decorator with arguments
        query_results = test_services.provider.search_chunks_regex("@query\\('referral'\\)")
        assert len(query_results) > 0, \
            "Parameter decorators with string arguments should be captured"


# Direct parser tests (unit-level)
class TestTypeScriptDecoratorParserDirect:
    """Direct tests of TypeScript decorator parsing functionality."""

    def test_class_decorator_parsed(self):
        """Test that parser extracts class decorators."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        test_content = """
@sealed
class MyClass {
    value = 0;
}
"""

        chunks = parser.parse_content(test_content, "test.ts", FileId(1))

        # Should find class with decorator
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should extract decorated class"

        # Decorator should be in content
        assert any("@sealed" in c.code for c in class_chunks), \
            "Class decorator should be included in chunk code"

    def test_decorator_factory_parsed(self):
        """Test that parser extracts decorator factories with arguments."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        test_content = """
@component({ selector: 'app-root' })
class AppComponent {
    title = 'app';
}
"""

        chunks = parser.parse_content(test_content, "test.ts", FileId(1))

        # Should find class with decorator factory
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should extract class with decorator factory"

        # Decorator factory and arguments should be in content
        assert any("@component" in c.code and "selector" in c.code for c in class_chunks), \
            "Decorator factory with configuration should be included in chunk"

    def test_method_decorator_parsed(self):
        """Test that parser extracts method decorators."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        test_content = """
class Service {
    @logged
    doWork() {
        return true;
    }
}
"""

        chunks = parser.parse_content(test_content, "test.ts", FileId(1))

        # Method decorators are part of class chunks (methods are not extracted separately)
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should extract class containing decorated method"

        # Decorator should be in class content
        assert any("@logged" in c.code and "doWork" in c.code for c in class_chunks), \
            "Method decorator should be included in class chunk code"

    def test_property_decorator_parsed(self):
        """Test that parser extracts property decorators."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        test_content = """
class Store {
    @observable
    count = 0;
}
"""

        chunks = parser.parse_content(test_content, "test.ts", FileId(1))

        # Should find class (property decorators are part of class)
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should extract class with decorated property"

        # Property decorator should be in class content
        assert any("@observable" in c.code and "count" in c.code for c in class_chunks), \
            "Property decorator should be included in class chunk"

    def test_parameter_decorator_parsed(self):
        """Test that parser extracts parameter decorators."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        test_content = """
class Controller {
    constructor(@inject('Service') service: Service) {
        this.service = service;
    }
}
"""

        chunks = parser.parse_content(test_content, "test.ts", FileId(1))

        # Should find class or constructor with parameter decorator
        # Parameter decorators are typically part of constructor/method chunks
        all_chunks = [c for c in chunks]
        assert len(all_chunks) > 0, "Should extract chunks from decorated code"

        # Decorator should appear somewhere in the parsed content
        has_decorator = any("@inject" in c.code for c in all_chunks)
        assert has_decorator, \
            "Parameter decorator should be captured in constructor or class chunk"

    def test_multiple_decorators_parsed(self):
        """Test that parser extracts multiple stacked decorators."""
        factory = get_parser_factory()
        parser = factory.create_parser(Language.TYPESCRIPT)

        test_content = """
@injectable()
@singleton
class UserService {
    @logged
    @cached
    getUser(id: number) {
        return null;
    }
}
"""

        chunks = parser.parse_content(test_content, "test.ts", FileId(1))

        # Should find class with multiple decorators
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should extract class with multiple decorators"

        # All class decorators should be present
        class_code = " ".join(c.code for c in class_chunks)
        assert "@injectable" in class_code, "First class decorator should be captured"
        assert "@singleton" in class_code, "Second class decorator should be captured"

        # Method decorators should be present
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION and "getUser" in c.symbol]
        if method_chunks:
            method_code = " ".join(c.code for c in method_chunks)
            assert "@logged" in method_code or "@cached" in method_code, \
                "Method decorators should be captured in method chunks"
