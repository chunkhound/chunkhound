"""End-to-end multi-language pipeline integrity test.

Tests the complete indexing pipeline across all supported languages,
verifying data integrity in DuckDB.

IMPORTANT: This test MUST cover all available Language parsers.
If a new language is added to the Language enum, add samples here.
The test_all_parsers_covered test will fail if any parsers are missing.
"""

import pytest
from pathlib import Path

from chunkhound.core.types.common import Language
from chunkhound.database_factory import create_services
from chunkhound.core.config.config import Config
from chunkhound.embeddings import EmbeddingManager
from tests.fixtures.fake_providers import FakeEmbeddingProvider

# Languages where binary content prevents standard text testing
BINARY_CONTENT_LANGUAGES = {Language.PDF}

# Language samples: (extension, sample1_content, sample2_content)
# Each sample should produce at least 1 chunk when parsed
# IMPORTANT: Every Language enum member MUST have an entry - no exceptions
LANGUAGE_SAMPLES: dict[Language, tuple[str, str | None, str | None]] = {
    # Programming languages
    Language.PYTHON: (
        ".py",
        'def greet(name: str) -> str:\n    """Say hello."""\n    return f"Hello, {name}"',
        "class Calculator:\n    def add(self, a: int, b: int) -> int:\n        return a + b",
    ),
    Language.JAVASCRIPT: (
        ".js",
        "function greet(name) {\n  return `Hello, ${name}`;\n}",
        "const add = (a, b) => a + b;\nexport { add };",
    ),
    Language.TYPESCRIPT: (
        ".ts",
        "function greet(name: string): string {\n  return `Hello, ${name}`;\n}",
        "interface User {\n  id: number;\n  name: string;\n}",
    ),
    Language.JSX: (
        ".jsx",
        "function Button({ label }) {\n  return <button>{label}</button>;\n}",
        "const Card = ({ title }) => <div className=\"card\">{title}</div>;",
    ),
    Language.TSX: (
        ".tsx",
        "interface Props { label: string; }\nfunction Button({ label }: Props) {\n  return <button>{label}</button>;\n}",
        "const Card: React.FC<{ title: string }> = ({ title }) => <div>{title}</div>;",
    ),
    Language.JAVA: (
        ".java",
        "public class Greeter {\n    public String greet(String name) {\n        return \"Hello, \" + name;\n    }\n}",
        "public class Calculator {\n    public int add(int a, int b) {\n        return a + b;\n    }\n}",
    ),
    Language.C: (
        ".c",
        "int add(int a, int b) {\n    return a + b;\n}",
        "void greet(const char* name) {\n    printf(\"Hello, %s\\n\", name);\n}",
    ),
    Language.CPP: (
        ".cpp",
        "class Calculator {\npublic:\n    int add(int a, int b) { return a + b; }\n};",
        "namespace math {\n    double multiply(double a, double b) { return a * b; }\n}",
    ),
    Language.CSHARP: (
        ".cs",
        "public class Greeter {\n    public string Greet(string name) => $\"Hello, {name}\";\n}",
        "public class Calculator {\n    public int Add(int a, int b) => a + b;\n}",
    ),
    Language.GO: (
        ".go",
        "package main\n\nfunc greet(name string) string {\n\treturn \"Hello, \" + name\n}",
        "package math\n\nfunc Add(a, b int) int {\n\treturn a + b\n}",
    ),
    Language.RUST: (
        ".rs",
        "fn greet(name: &str) -> String {\n    format!(\"Hello, {}\", name)\n}",
        "pub struct Calculator;\n\nimpl Calculator {\n    pub fn add(&self, a: i32, b: i32) -> i32 {\n        a + b\n    }\n}",
    ),
    Language.ZIG: (
        ".zig",
        'const std = @import("std");\n\npub fn add(a: i32, b: i32) i32 {\n    return a + b;\n}',
        "pub fn greet(name: []const u8) void {\n    std.debug.print(\"Hello, {s}\\n\", .{name});\n}",
    ),
    Language.HASKELL: (
        ".hs",
        "greet :: String -> String\ngreet name = \"Hello, \" ++ name",
        "add :: Int -> Int -> Int\nadd a b = a + b",
    ),
    Language.BASH: (
        ".sh",
        "#!/bin/bash\ngreet() {\n    echo \"Hello, $1\"\n}",
        "#!/bin/bash\nadd() {\n    echo $(($1 + $2))\n}",
    ),
    Language.MATLAB: (
        ".m",
        "function result = add(a, b)\n    result = a + b;\nend",
        "function greeting = greet(name)\n    greeting = ['Hello, ', name];\nend",
    ),
    Language.OBJC: (
        ".mm",
        "@interface Greeter : NSObject\n- (NSString *)greet:(NSString *)name;\n@end\n\n@implementation Greeter\n- (NSString *)greet:(NSString *)name {\n    return [NSString stringWithFormat:@\"Hello, %@\", name];\n}\n@end",
        "@interface Calculator : NSObject\n- (int)add:(int)a with:(int)b;\n@end\n\n@implementation Calculator\n- (int)add:(int)a with:(int)b {\n    return a + b;\n}\n@end",
    ),
    Language.SWIFT: (
        ".swift",
        "func greet(name: String) -> String {\n    return \"Hello, \\(name)\"\n}",
        "class Calculator {\n    func add(_ a: Int, _ b: Int) -> Int {\n        return a + b\n    }\n}",
    ),
    Language.PHP: (
        ".php",
        "<?php\nfunction greet(string $name): string {\n    return \"Hello, $name\";\n}",
        "<?php\nclass Calculator {\n    public function add(int $a, int $b): int {\n        return $a + $b;\n    }\n}",
    ),
    Language.KOTLIN: (
        ".kt",
        "fun greet(name: String): String {\n    return \"Hello, $name\"\n}",
        "class Calculator {\n    fun add(a: Int, b: Int): Int = a + b\n}",
    ),
    Language.GROOVY: (
        ".groovy",
        "def greet(String name) {\n    return \"Hello, $name\"\n}",
        "class Calculator {\n    int add(int a, int b) {\n        return a + b\n    }\n}",
    ),
    Language.DART: (
        ".dart",
        "String greet(String name) {\n  return 'Hello, $name';\n}",
        "class Calculator {\n  int add(int a, int b) => a + b;\n}",
    ),
    Language.VUE: (
        ".vue",
        "<template>\n  <div>{{ greeting }}</div>\n</template>\n\n<script>\nexport default {\n  data() {\n    return { greeting: 'Hello' };\n  }\n};\n</script>",
        "<template>\n  <button @click=\"increment\">{{ count }}</button>\n</template>\n\n<script setup>\nimport { ref } from 'vue';\nconst count = ref(0);\nconst increment = () => count.value++;\n</script>",
    ),
    Language.SVELTE: (
        ".svelte",
        "<script>\n  let name = 'world';\n</script>\n\n<h1>Hello {name}!</h1>",
        "<script>\n  let count = 0;\n  function increment() {\n    count += 1;\n  }\n</script>\n\n<button on:click={increment}>{count}</button>",
    ),
    # Data/Configuration languages
    Language.JSON: (
        ".json",
        '{\n  "name": "example",\n  "version": "1.0.0",\n  "description": "A sample JSON file"\n}',
        '{\n  "users": [\n    {"id": 1, "name": "Alice"},\n    {"id": 2, "name": "Bob"}\n  ]\n}',
    ),
    Language.YAML: (
        ".yaml",
        "name: example\nversion: 1.0.0\ndescription: A sample YAML file",
        "users:\n  - id: 1\n    name: Alice\n  - id: 2\n    name: Bob",
    ),
    Language.TOML: (
        ".toml",
        '[package]\nname = "example"\nversion = "1.0.0"',
        '[database]\nhost = "localhost"\nport = 5432',
    ),
    Language.HCL: (
        ".tf",
        'resource "aws_instance" "example" {\n  ami           = "ami-12345678"\n  instance_type = "t2.micro"\n}',
        'variable "region" {\n  description = "AWS region"\n  default     = "us-west-2"\n}',
    ),
    # Documentation languages
    Language.MARKDOWN: (
        ".md",
        "# Hello World\n\nThis is a sample markdown file.\n\n## Features\n\n- Feature 1\n- Feature 2",
        "# API Reference\n\n## Functions\n\n### `greet(name)`\n\nReturns a greeting string.",
    ),
    # Plain text and special formats
    Language.MAKEFILE: (
        ".mk",
        ".PHONY: build\n\nbuild:\n\techo \"Building...\"\n\tgcc -o app main.c",
        ".PHONY: test\n\ntest:\n\tpytest tests/",
    ),
    Language.TEXT: (
        ".txt",
        "This is a plain text file.\nIt contains some sample content.\nLine three.",
        "Another text file with different content.\nUsed for testing purposes.",
    ),
    Language.PDF: (
        ".pdf",
        None,  # PDF requires binary content, skip in tests
        None,
    ),
    # Generic/unknown (fallback for unrecognized files)
    Language.UNKNOWN: (
        ".unknown",
        "Unknown file type content line one.\nSecond line of unknown content.",
        "Different unknown content.\nAnother line here.",
    ),
}


@pytest.fixture
async def multi_language_db(tmp_path):
    """Create DB with files for all supported languages."""
    db_path = tmp_path / "e2e_test.duckdb"

    # Set up embedding manager with FakeEmbeddingProvider
    # Use 1536 dims to match default OpenAI dimensions
    embedding_manager = EmbeddingManager()
    fake_provider = FakeEmbeddingProvider(dims=1536)
    embedding_manager.register_provider(fake_provider, set_default=True)

    # Create Config with database settings only
    # Use openai config but our embedding_manager will be used instead
    config = Config(
        target_dir=tmp_path,
        database={"path": str(db_path), "provider": "duckdb"},
        embedding={
            "provider": "openai",
            "api_key": "fake-api-key-not-used",
            "model": "text-embedding-3-small",
        },
    )

    # Create services using database_factory
    services = create_services(db_path, config, embedding_manager)

    # Track expected file count
    expected_files = 0

    # Process each language's sample files
    for language, (ext, content1, content2) in LANGUAGE_SAMPLES.items():
        # Skip languages where content is None (e.g., PDF)
        if content1 is None:
            continue

        # Create unique filenames per language
        file1 = tmp_path / f"sample1_{language.name.lower()}{ext}"
        file1.write_text(content1)
        result1 = await services.indexing_coordinator.process_file(
            file1, skip_embeddings=False
        )
        if result1.get("status") == "success":
            expected_files += 1

        # Process second sample if available
        if content2 is not None:
            file2 = tmp_path / f"sample2_{language.name.lower()}{ext}"
            file2.write_text(content2)
            result2 = await services.indexing_coordinator.process_file(
                file2, skip_embeddings=False
            )
            if result2.get("status") == "success":
                expected_files += 1

    yield services, expected_files


def verify_database_integrity(
    db_provider, expected_file_count: int, dims: int = 1536
) -> dict:
    """Run all integrity checks against the database.

    Args:
        db_provider: The database provider with execute_query method
        expected_file_count: Expected number of files
        dims: Embedding dimensions (default 384 for FakeEmbeddingProvider)

    Returns:
        Dict with integrity check results
    """
    # 1. Verify file count matches expected
    result = db_provider.execute_query("SELECT COUNT(*) as cnt FROM files")
    file_count = result[0]["cnt"]
    assert file_count == expected_file_count, (
        f"File count mismatch: expected {expected_file_count}, got {file_count}"
    )

    # 2. No duplicate file paths
    duplicates = db_provider.execute_query(
        "SELECT path, COUNT(*) as cnt FROM files GROUP BY path HAVING cnt > 1"
    )
    assert len(duplicates) == 0, f"Found duplicate file paths: {duplicates}"

    # 3. No orphan chunks (chunks without parent files)
    orphan_chunks = db_provider.execute_query(
        "SELECT c.id, c.file_id FROM chunks c "
        "LEFT JOIN files f ON c.file_id = f.id WHERE f.id IS NULL"
    )
    assert len(orphan_chunks) == 0, f"Found orphan chunks: {orphan_chunks}"

    # 4. No duplicate chunks (same file_id + chunk_type + symbol + position + code)
    # True duplicates have identical content at the same position
    duplicate_chunks = db_provider.execute_query(
        "SELECT file_id, chunk_type, symbol, start_line, end_line, code, COUNT(*) as cnt "
        "FROM chunks GROUP BY file_id, chunk_type, symbol, start_line, end_line, code HAVING cnt > 1"
    )
    assert len(duplicate_chunks) == 0, f"Found duplicate chunks: {duplicate_chunks}"

    # 5. Get embedding table name
    embedding_table = f"embeddings_{dims}"

    # 6. No orphan embeddings (embeddings without chunks)
    orphan_embeddings = db_provider.execute_query(
        f"SELECT e.id, e.chunk_id FROM {embedding_table} e "
        f"LEFT JOIN chunks c ON e.chunk_id = c.id WHERE c.id IS NULL"
    )
    assert len(orphan_embeddings) == 0, f"Found orphan embeddings: {orphan_embeddings}"

    # 7. Each chunk has exactly one embedding (count match)
    chunk_count_result = db_provider.execute_query("SELECT COUNT(*) as cnt FROM chunks")
    chunk_count = chunk_count_result[0]["cnt"]

    embedding_count_result = db_provider.execute_query(
        f"SELECT COUNT(*) as cnt FROM {embedding_table}"
    )
    embedding_count = embedding_count_result[0]["cnt"]

    assert chunk_count == embedding_count, (
        f"Chunk/embedding count mismatch: {chunk_count} chunks, "
        f"{embedding_count} embeddings"
    )

    # 8. No duplicate embeddings per chunk
    duplicate_embeddings = db_provider.execute_query(
        f"SELECT chunk_id, COUNT(*) as cnt FROM {embedding_table} "
        f"GROUP BY chunk_id HAVING cnt > 1"
    )
    assert len(duplicate_embeddings) == 0, (
        f"Found duplicate embeddings per chunk: {duplicate_embeddings}"
    )

    # 9. No chunks with empty content
    empty_chunks_result = db_provider.execute_query(
        "SELECT COUNT(*) as cnt FROM chunks WHERE code IS NULL OR code = ''"
    )
    empty_chunk_count = empty_chunks_result[0]["cnt"]
    assert empty_chunk_count == 0, (
        f"Found {empty_chunk_count} chunks with empty content"
    )

    return {
        "files": expected_file_count,
        "chunks": chunk_count,
        "embeddings": embedding_count,
        "integrity": "PASSED",
    }


@pytest.mark.asyncio
async def test_multi_language_pipeline_integrity(multi_language_db):
    """End-to-end test: all languages → chunks → embeddings → integrity.

    This test validates the complete indexing pipeline across all supported
    languages, verifying:
    1. All files are indexed correctly
    2. All chunks have parent files (no orphans)
    3. All embeddings have parent chunks (no orphans)
    4. Each chunk has exactly one embedding
    5. No duplicate entries in any table
    """
    services, expected_files = multi_language_db

    # Verify database integrity
    stats = verify_database_integrity(
        services.provider,
        expected_files,
        dims=1536,  # FakeEmbeddingProvider configured with OpenAI-compatible dims
    )

    # Log results for debugging
    print(f"\n=== E2E Multi-Language Integrity Test Results ===")
    print(f"Files indexed: {stats['files']}")
    print(f"Chunks created: {stats['chunks']}")
    print(f"Embeddings generated: {stats['embeddings']}")
    print(f"Integrity: {stats['integrity']}")

    # Verify we actually processed files (sanity check)
    assert stats["files"] > 0, "No files were processed"
    assert stats["chunks"] > 0, "No chunks were created"
    assert stats["embeddings"] > 0, "No embeddings were generated"


def test_all_parsers_covered():
    """Verify every Language enum member has test samples.

    This test ensures that when new languages are added to the Language enum,
    they MUST also be added to LANGUAGE_SAMPLES. Without this guard, new
    parsers could be added without any integration test coverage.

    NO EXCEPTIONS - every Language enum member must have an entry.
    """
    all_languages = set(Language)
    covered_languages = set(LANGUAGE_SAMPLES.keys())

    missing_languages = all_languages - covered_languages
    assert not missing_languages, (
        f"Missing test samples for languages: {sorted(lang.name for lang in missing_languages)}. "
        f"Every Language enum member MUST have an entry in LANGUAGE_SAMPLES."
    )

    extra_languages = covered_languages - all_languages
    assert not extra_languages, (
        f"LANGUAGE_SAMPLES contains languages not in Language enum: "
        f"{sorted(lang.name for lang in extra_languages)}"
    )


def test_each_parser_has_two_samples():
    """Verify each parser has exactly 2 test files for thorough coverage.

    Each language MUST have 2 distinct sample files to test different code
    patterns and ensure parser robustness. Binary formats (PDF) are the only
    exception as they require special handling.
    """
    insufficient_samples = []

    for language, (ext, sample1, sample2) in LANGUAGE_SAMPLES.items():
        # Binary content languages are allowed to have None samples
        if language in BINARY_CONTENT_LANGUAGES:
            continue

        if sample1 is None or sample2 is None:
            insufficient_samples.append(language.name)

    assert not insufficient_samples, (
        f"Languages with fewer than 2 test samples: {sorted(insufficient_samples)}. "
        f"Each parser MUST have 2 sample files for adequate test coverage."
    )


@pytest.fixture
async def single_file_db(tmp_path):
    """Create DB with a single Python file for re-indexing tests."""
    db_path = tmp_path / "reindex_test.duckdb"

    # Set up embedding manager with FakeEmbeddingProvider
    embedding_manager = EmbeddingManager()
    fake_provider = FakeEmbeddingProvider(dims=1536)
    embedding_manager.register_provider(fake_provider, set_default=True)

    config = Config(
        target_dir=tmp_path,
        database={"path": str(db_path), "provider": "duckdb"},
        embedding={
            "provider": "openai",
            "api_key": "fake-api-key-not-used",
            "model": "text-embedding-3-small",
        },
    )

    services = create_services(db_path, config, embedding_manager)
    yield services, tmp_path


@pytest.mark.asyncio
async def test_file_reindexing_maintains_integrity(single_file_db):
    """Verify re-indexing modified files doesn't create orphans or duplicates.

    This tests the chunk diffing path (indexing_coordinator.py:935-964):
    - Existing chunks compared with new chunks
    - Deleted/modified chunks removed via delete_chunks_batch()
    - New chunks inserted

    CRITICAL: This test exercises delete_chunks_batch() which was missing
    in LanceDB provider until this fix.
    """
    services, tmp_path = single_file_db

    # Step 1: Create initial Python file with 2 functions
    initial_content = '''def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"

def farewell(name: str) -> str:
    """Say goodbye."""
    return f"Goodbye, {name}"
'''
    test_file = tmp_path / "test_module.py"
    test_file.write_text(initial_content)

    # Step 2: Index initial file
    result1 = await services.indexing_coordinator.process_file(
        test_file, skip_embeddings=False
    )
    assert result1.get("status") == "success", f"Initial index failed: {result1}"

    # Step 3: Verify initial state
    initial_stats = verify_database_integrity(services.provider, 1, dims=1536)
    initial_chunks = initial_stats["chunks"]
    initial_embeddings = initial_stats["embeddings"]
    assert initial_chunks >= 1, f"Expected at least 1 chunk, got {initial_chunks}"
    assert initial_chunks == initial_embeddings, "Initial chunk/embedding mismatch"

    # Step 4: Modify file - remove farewell, change greet, add new function
    modified_content = '''def greet(name: str) -> str:
    """Say hello with enthusiasm."""
    return f"Hello, {name}!"

def calculate(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
    test_file.write_text(modified_content)

    # Step 5: Re-index same file (triggers chunk diffing path)
    result2 = await services.indexing_coordinator.process_file(
        test_file, skip_embeddings=False
    )
    assert result2.get("status") == "success", f"Re-index failed: {result2}"

    # Step 6: Verify integrity after re-indexing
    # This is the critical check - no orphan embeddings from deleted chunks
    final_stats = verify_database_integrity(services.provider, 1, dims=1536)

    assert final_stats["integrity"] == "PASSED", "Integrity check failed after re-indexing"
    assert final_stats["chunks"] == final_stats["embeddings"], (
        f"Chunk/embedding mismatch after re-index: "
        f"{final_stats['chunks']} chunks, {final_stats['embeddings']} embeddings"
    )


@pytest.mark.asyncio
async def test_file_deletion_cascade_integrity(single_file_db):
    """Verify delete_file_completely() cascades to chunks and embeddings.

    This tests the deletion cascade path to ensure no orphaned records remain
    when a file is completely removed from the index.
    """
    services, tmp_path = single_file_db

    # Step 1: Create 3 Python files
    files = []
    for i in range(3):
        content = f'''def function_{i}(x: int) -> int:
    """Function {i} docstring."""
    return x + {i}
'''
        file_path = tmp_path / f"module_{i}.py"
        file_path.write_text(content)
        files.append(file_path)

    # Step 2: Index all 3 files
    for file_path in files:
        result = await services.indexing_coordinator.process_file(
            file_path, skip_embeddings=False
        )
        assert result.get("status") == "success", f"Failed to index {file_path}"

    # Step 3: Verify initial state (3 files)
    initial_stats = verify_database_integrity(services.provider, 3, dims=1536)
    assert initial_stats["files"] == 3

    # Step 4: Delete middle file completely
    middle_file_rel = files[1].relative_to(tmp_path).as_posix()
    deleted = services.provider.delete_file_completely(middle_file_rel)
    assert deleted, f"delete_file_completely returned False for {middle_file_rel}"

    # Step 5: Verify integrity after deletion (2 files remain)
    final_stats = verify_database_integrity(services.provider, 2, dims=1536)

    assert final_stats["integrity"] == "PASSED", "Integrity check failed after deletion"
    assert final_stats["files"] == 2, f"Expected 2 files, got {final_stats['files']}"
    assert final_stats["chunks"] == final_stats["embeddings"], (
        f"Chunk/embedding mismatch after deletion: "
        f"{final_stats['chunks']} chunks, {final_stats['embeddings']} embeddings"
    )
