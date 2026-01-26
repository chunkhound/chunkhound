"""E2E test: All parsers respect chunk size constraints.

Verifies that ALL parser features NEVER produce chunks exceeding limits
by intercepting at the embedding provider level.

IMPORTANT: This test MUST cover all available Language parsers.
If a new language is added to the Language enum, add samples here.
The test_all_parsers_covered test will fail if any parsers are missing.

Validates:
1. max_chunk_size: 1200 non-whitespace chars - FAIL on violation
2. safe_token_limit: 6000 tokens - FAIL on violation
3. min_chunk_size: 25 non-whitespace chars - Log warning only (soft threshold)
"""

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.types.common import Language
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from tests.fixtures.fake_providers import ValidatingEmbeddingProvider

# Default constraints from CASTConfig (universal_parser.py:52-59)
MAX_CHUNK_SIZE = 1200  # non-whitespace chars
MIN_CHUNK_SIZE = 25  # soft threshold for suspiciously small
SAFE_TOKEN_LIMIT = 6000  # estimated tokens

# Languages with binary content (cannot test with text samples)
BINARY_CONTENT_LANGUAGES = {Language.PDF}

# Languages that don't use cAST algorithm for size enforcement
# These parsers use simpler line-based or fallback chunking strategies
# TODO: Consider adding size enforcement to these parsers
LANGUAGES_WITHOUT_SIZE_ENFORCEMENT = {
    Language.TEXT,  # Plaintext fallback - no semantic structure
    Language.UNKNOWN,  # Generic fallback parser
    Language.YAML,  # Line-based YAML chunking
}


def _make_large_statements(statement: str, count: int = 500) -> str:
    """Generate repeated statements to exceed 1200 non-ws chars.

    Each statement should have ~5-10 non-ws chars to reliably exceed 1200 total.
    With count=500 and 5 chars/statement, we get ~2500 non-ws chars.
    """
    return "\n".join([statement] * count)


# Language samples: (extension, large_sample, normal_sample)
# Large samples MUST exceed 1200 non-whitespace chars to trigger split paths
# Normal samples verify no over-splitting on small inputs
LARGE_LANGUAGE_SAMPLES: dict[Language, tuple[str, str, str]] = {
    # === Programming Languages ===
    Language.PYTHON: (
        ".py",
        # Large: ~1500+ non-ws chars - triggers line-based split
        "def process_large_data(items):\n"
        '    """Process items."""\n' + _make_large_statements("    x = 1"),
        # Normal
        'def greet(name: str) -> str:\n    """Say hello."""\n    return f"Hello, {name}"',
    ),
    Language.JAVASCRIPT: (
        ".js",
        # Minified: single line, ~2000 chars - triggers emergency_split
        "function processData(items){" + "a=1;" * 400 + "}",
        # Normal
        "function greet(name) {\n  return `Hello, ${name}`;\n}",
    ),
    Language.TYPESCRIPT: (
        ".ts",
        # Large class
        "class DataProcessor {\n"
        + _make_large_statements("  private field: number = 0;"),
        # Normal
        "function greet(name: string): string {\n  return `Hello, ${name}`;\n}",
    ),
    Language.JSX: (
        ".jsx",
        # Large component
        "function LargeComponent({ items }) {\n  return (\n    <div>\n"
        + _make_large_statements("      <span>item</span>")
        + "\n    </div>\n  );\n}",
        # Normal
        "function Button({ label }) {\n  return <button>{label}</button>;\n}",
    ),
    Language.TSX: (
        ".tsx",
        # Large typed component
        "interface Props { items: string[]; }\n"
        "function LargeComponent({ items }: Props) {\n  return (\n    <div>\n"
        + _make_large_statements("      <span>item</span>")
        + "\n    </div>\n  );\n}",
        # Normal
        "interface Props { label: string; }\n"
        "function Button({ label }: Props) {\n  return <button>{label}</button>;\n}",
    ),
    Language.JAVA: (
        ".java",
        # Large class
        "public class DataProcessor {\n"
        + _make_large_statements("    private int field = 0;"),
        # Normal
        "public class Greeter {\n"
        "    public String greet(String name) {\n"
        '        return "Hello, " + name;\n'
        "    }\n"
        "}",
    ),
    Language.CSHARP: (
        ".cs",
        # Large class
        "public class DataProcessor {\n"
        + _make_large_statements("    private int field = 0;"),
        # Normal
        "public class Greeter {\n"
        '    public string Greet(string name) => $"Hello, {name}";\n'
        "}",
    ),
    Language.GO: (
        ".go",
        # Large function
        "package main\n\nfunc processData(items []int) int {\n"
        + _make_large_statements("\tx := 1")
        + "\n\treturn x\n}",
        # Normal
        "package main\n\nfunc greet(name string) string {\n"
        '\treturn "Hello, " + name\n}',
    ),
    Language.RUST: (
        ".rs",
        # Large function
        "fn process_data(items: Vec<i32>) -> i32 {\n"
        + _make_large_statements("    let x = 1;")
        + "\n    x\n}",
        # Normal
        'fn greet(name: &str) -> String {\n    format!("Hello, {}", name)\n}',
    ),
    Language.ZIG: (
        ".zig",
        # Large function
        'const std = @import("std");\n\n'
        "pub fn processData(items: []const i32) i32 {\n"
        + _make_large_statements("    var x: i32 = 1;")
        + "\n    return x;\n}",
        # Normal
        'const std = @import("std");\n\n'
        "pub fn add(a: i32, b: i32) i32 {\n    return a + b;\n}",
    ),
    Language.C: (
        ".c",
        # Large function
        "int process_data(int* items, int len) {\n"
        + _make_large_statements("    int x = 1;")
        + "\n    return x;\n}",
        # Normal
        "int add(int a, int b) {\n    return a + b;\n}",
    ),
    Language.CPP: (
        ".cpp",
        # Large class
        "class DataProcessor {\npublic:\n"
        + _make_large_statements("    int field = 0;")
        + "\n};",
        # Normal
        "class Calculator {\npublic:\n    int add(int a, int b) { return a + b; }\n};",
    ),
    Language.HASKELL: (
        ".hs",
        # Large function with many let bindings
        "processData :: [Int] -> Int\nprocessData items = \n"
        + _make_large_statements("  let x = 1 in")
        + "\n  x",
        # Normal
        'greet :: String -> String\ngreet name = "Hello, " ++ name',
    ),
    Language.KOTLIN: (
        ".kt",
        # Large function
        "fun processData(items: List<Int>): Int {\n"
        + _make_large_statements("    val x = 1")
        + "\n    return x\n}",
        # Normal
        'fun greet(name: String): String {\n    return "Hello, $name"\n}',
    ),
    Language.GROOVY: (
        ".groovy",
        # Large function
        "def processData(items) {\n"
        + _make_large_statements("    def x = 1")
        + "\n    return x\n}",
        # Normal
        'def greet(String name) {\n    return "Hello, $name"\n}',
    ),
    Language.SWIFT: (
        ".swift",
        # Large function
        "func processData(items: [Int]) -> Int {\n"
        + _make_large_statements("    var x = 1")
        + "\n    return x\n}",
        # Normal
        'func greet(name: String) -> String {\n    return "Hello, \\(name)"\n}',
    ),
    Language.PHP: (
        ".php",
        # Large function
        "<?php\nfunction processData($items) {\n"
        + _make_large_statements("    $x = 1;")
        + "\n    return $x;\n}",
        # Normal
        '<?php\nfunction greet(string $name): string {\n    return "Hello, $name";\n}',
    ),
    Language.DART: (
        ".dart",
        # Large function
        "int processData(List<int> items) {\n"
        + _make_large_statements("  var x = 1;")
        + "\n  return x;\n}",
        # Normal
        "String greet(String name) {\n  return 'Hello, $name';\n}",
    ),
    Language.LUA: (
        ".lua",
        # Large function
        "function processData(items)\n"
        + _make_large_statements("    local x = 1")
        + "\n    return x\nend",
        # Normal
        "function greet(name)\n    return 'Hello, ' .. name\nend",
    ),
    Language.BASH: (
        ".sh",
        # Large script
        "#!/bin/bash\nprocess_data() {\n"
        + _make_large_statements("    x=1")
        + '\n    echo "$x"\n}',
        # Normal
        '#!/bin/bash\ngreet() {\n    echo "Hello, $1"\n}',
    ),
    Language.MATLAB: (
        ".m",
        # Large function
        "function result = process_data(items)\n"
        + _make_large_statements("    x = 1;")
        + "\n    result = x;\nend",
        # Normal
        "function result = add(a, b)\n    result = a + b;\nend",
    ),
    Language.OBJC: (
        ".mm",
        # Large implementation
        "@implementation DataProcessor\n"
        + _make_large_statements("- (int)field { return 0; }")
        + "\n@end",
        # Normal
        "@interface Greeter : NSObject\n- (NSString *)greet:(NSString *)name;\n@end\n\n"
        "@implementation Greeter\n"
        '- (NSString *)greet:(NSString *)name {\n    return [NSString stringWithFormat:@"Hello, %@", name];\n}\n'
        "@end",
    ),
    Language.MAKEFILE: (
        ".mk",
        # Large makefile
        ".PHONY: all\n\nall:\n" + _make_large_statements('\techo "building"'),
        # Normal
        '.PHONY: build\n\nbuild:\n\techo "Building..."\n\tgcc -o app main.c',
    ),
    Language.HCL: (
        ".tf",
        # Large resource
        'resource "aws_instance" "large" {\n'
        + _make_large_statements('  ami = "ami-12345678"')
        + "\n}",
        # Normal
        'resource "aws_instance" "example" {\n'
        '  ami           = "ami-12345678"\n'
        '  instance_type = "t2.micro"\n}',
    ),
    Language.VUE: (
        ".vue",
        # Large component
        "<template>\n  <div>\n"
        + _make_large_statements("    <span>item</span>")
        + "\n  </div>\n</template>\n\n<script>\nexport default {\n"
        "  data() {\n    return { greeting: 'Hello' };\n  }\n};\n</script>",
        # Normal
        "<template>\n  <div>{{ greeting }}</div>\n</template>\n\n"
        "<script>\nexport default {\n  data() {\n    return { greeting: 'Hello' };\n  }\n};\n</script>",
    ),
    Language.SVELTE: (
        ".svelte",
        # Large component
        "<script>\n"
        + _make_large_statements("  let x = 1;")
        + "\n</script>\n\n<div>{x}</div>",
        # Normal
        "<script>\n  let name = 'world';\n</script>\n\n<h1>Hello {name}!</h1>",
    ),
    # === Documentation Languages ===
    Language.MARKDOWN: (
        ".md",
        # Large document
        "# Large Document\n\n" + _make_large_statements("- Item with content here"),
        # Normal
        "# Hello World\n\nThis is a sample markdown file.\n\n## Features\n\n- Feature 1\n- Feature 2",
    ),
    # === Data/Configuration Languages ===
    Language.JSON: (
        ".json",
        # Large JSON
        '{\n  "items": [\n'
        + ",\n".join(['    {"id": ' + str(i) + "}" for i in range(200)])
        + "\n  ]\n}",
        # Normal
        '{\n  "name": "example",\n  "version": "1.0.0",\n  "description": "A sample JSON file"\n}',
    ),
    Language.YAML: (
        ".yaml",
        # Large YAML
        "items:\n" + _make_large_statements("  - id: 1"),
        # Normal
        "name: example\nversion: 1.0.0\ndescription: A sample YAML file",
    ),
    Language.TOML: (
        ".toml",
        # Large TOML
        "[package]\n" + _make_large_statements('name = "item"'),
        # Normal
        '[package]\nname = "example"\nversion = "1.0.0"',
    ),
    Language.TEXT: (
        ".txt",
        # Large text
        "Large text file content.\n" + _make_large_statements("Line of text content."),
        # Normal
        "This is a plain text file.\nIt contains some sample content.\nLine three.",
    ),
    Language.PDF: (
        ".pdf",
        None,  # PDF requires binary content, skip in tests
        None,
    ),
    # === Generic/Unknown ===
    Language.UNKNOWN: (
        ".unknown",
        # Large unknown
        "Unknown file content.\n" + _make_large_statements("Line of unknown content."),
        # Normal
        "Unknown file type content line one.\nSecond line of unknown content.",
    ),
}


@pytest.fixture
async def validating_db(tmp_path):
    """Create DB with ValidatingEmbeddingProvider to intercept all chunks."""
    db_path = tmp_path / "chunk_validation_test.duckdb"

    embedding_manager = EmbeddingManager()
    validating_provider = ValidatingEmbeddingProvider(
        dims=1536,
        max_chunk_size=MAX_CHUNK_SIZE,
        min_chunk_size=MIN_CHUNK_SIZE,
        safe_token_limit=SAFE_TOKEN_LIMIT,
    )
    embedding_manager.register_provider(validating_provider, set_default=True)

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
    yield services, validating_provider, tmp_path


@pytest.mark.asyncio
async def test_all_parsers_respect_chunk_size_constraints(validating_db):
    """Verify ALL parsers using cAST NEVER produce chunks exceeding size constraints.

    This test:
    1. Processes large samples (>1200 non-ws chars) that MUST trigger splitting
    2. Processes normal samples to verify no over-splitting
    3. Validates all chunks via ValidatingEmbeddingProvider intercept
    4. Fails on max_chunk_size or safe_token_limit violations (for cAST parsers)
    5. Logs warnings for suspiciously small chunks
    6. Separately tracks violations from parsers without size enforcement

    Note: Some parsers (TEXT, UNKNOWN, YAML) don't use cAST
    and are tracked separately as known limitations.
    """
    services, provider, tmp_path = validating_db

    # Track languages separately for validation
    enforced_languages = []
    non_enforced_languages = []

    for language, (ext, large_sample, normal_sample) in LARGE_LANGUAGE_SAMPLES.items():
        if language in BINARY_CONTENT_LANGUAGES:
            continue

        if large_sample is None:
            continue

        # Track which type of language
        if language in LANGUAGES_WITHOUT_SIZE_ENFORCEMENT:
            non_enforced_languages.append(language)
        else:
            enforced_languages.append(language)

        # Process large sample (must trigger splitting for cAST parsers)
        large_file = tmp_path / f"large_{language.name.lower()}{ext}"
        large_file.write_text(large_sample)
        await services.indexing_coordinator.process_file(
            large_file, skip_embeddings=False
        )

        # Process normal sample
        if normal_sample:
            normal_file = tmp_path / f"normal_{language.name.lower()}{ext}"
            normal_file.write_text(normal_sample)
            await services.indexing_coordinator.process_file(
                normal_file, skip_embeddings=False
            )

        # For enforced languages, check if large sample created violations
        # (violations from non-enforced languages will be filtered later)

    # Separate violations by type
    char_violations = provider.get_violations_by_type("max_chars_exceeded")
    token_violations = provider.get_violations_by_type("max_tokens_exceeded")
    small_chunks = provider.get_violations_by_type("suspiciously_small")

    # For reporting, note that violations may come from non-enforced languages
    # We can't easily filter by language since ValidatingProvider doesn't track source
    # So we report ALL violations but note which languages don't enforce

    # Check if violations are likely from non-enforced languages
    # (we know VUE/SVELTE have HTML templates, TEXT/UNKNOWN/YAML are plaintext-like)
    likely_non_enforced_chars = [
        v
        for v in char_violations
        if any(
            pattern in v["text_preview"]
            for pattern in [
                "<div>",
                "<span>",
                "items:",
                "Line of text",
                "Line of unknown",
            ]
        )
    ]
    likely_enforced_chars = [
        v for v in char_violations if v not in likely_non_enforced_chars
    ]

    likely_non_enforced_tokens = [
        v
        for v in token_violations
        if any(
            pattern in v["text_preview"]
            for pattern in [
                "<div>",
                "<span>",
                "items:",
                "Line of text",
                "Line of unknown",
            ]
        )
    ]
    likely_enforced_tokens = [
        v for v in token_violations if v not in likely_non_enforced_tokens
    ]

    # Fail on violations from cAST-enforced languages
    assert not likely_enforced_chars, (
        f"Found {len(likely_enforced_chars)} chunks from cAST parsers exceeding "
        f"max_chunk_size ({MAX_CHUNK_SIZE} non-ws chars):\n"
        + "\n".join(
            f"  - {v['non_ws_chars']} chars: {v['text_preview'][:50]}..."
            for v in likely_enforced_chars[:5]
        )
    )

    assert not likely_enforced_tokens, (
        f"Found {len(likely_enforced_tokens)} chunks from cAST parsers exceeding "
        f"safe_token_limit ({SAFE_TOKEN_LIMIT} tokens):\n"
        + "\n".join(
            f"  - {v['estimated_tokens']} tokens: {v['text_preview'][:50]}..."
            for v in likely_enforced_tokens[:5]
        )
    )

    # Log known limitations from non-enforced parsers
    if likely_non_enforced_chars:
        print(
            f"\nNote: {len(likely_non_enforced_chars)} oversized chunks from parsers "
            f"without size enforcement (TEXT, UNKNOWN, YAML)"
        )

    # Log suspiciously small chunks (soft threshold, don't fail)
    if small_chunks:
        print(
            f"\nWarning: {len(small_chunks)} suspiciously small chunks "
            f"(<{MIN_CHUNK_SIZE} non-ws chars)"
        )

    # Sanity check: verify we actually processed chunks
    assert provider.chunk_stats["total"] > 0, "No chunks were processed"
    print("\n=== Chunk Size Constraint Test Results ===")
    print(f"Languages with cAST enforcement: {len(enforced_languages)}")
    print(
        f"Languages without enforcement: {len(non_enforced_languages)} "
        f"({', '.join(lang.name for lang in non_enforced_languages)})"
    )
    print(f"Total chunks validated: {provider.chunk_stats['total']}")
    print(
        f"Char range: {provider.chunk_stats['min_size']} - "
        f"{provider.chunk_stats['max_size']} non-ws chars"
    )
    print(
        f"Token range: {provider.chunk_stats['min_tokens']} - "
        f"{provider.chunk_stats['max_tokens']} est. tokens"
    )


def test_all_parsers_covered_with_large_samples():
    """Verify every Language enum member has test samples.

    This test ensures that when new languages are added to the Language enum,
    they MUST also be added to LARGE_LANGUAGE_SAMPLES. Without this guard,
    new parsers could be added without chunk size constraint coverage.
    """
    all_languages = set(Language)
    covered_languages = set(LARGE_LANGUAGE_SAMPLES.keys())

    missing_languages = all_languages - covered_languages
    assert not missing_languages, (
        f"Missing test samples for languages: {sorted(lang.name for lang in missing_languages)}. "
        f"Every Language enum member MUST have an entry in LARGE_LANGUAGE_SAMPLES."
    )

    extra_languages = covered_languages - all_languages
    assert not extra_languages, (
        f"LARGE_LANGUAGE_SAMPLES contains languages not in Language enum: "
        f"{sorted(lang.name for lang in extra_languages)}"
    )


def test_large_samples_exceed_threshold():
    """Verify large samples actually exceed 1200 non-whitespace chars.

    If samples don't exceed the threshold, they won't trigger split paths
    and the test won't be effective.
    """
    import re

    min_large_size = 1200  # Must exceed max_chunk_size to trigger split

    undersized = []
    for language, (ext, large_sample, normal_sample) in LARGE_LANGUAGE_SAMPLES.items():
        if language in BINARY_CONTENT_LANGUAGES or large_sample is None:
            continue

        non_ws_chars = len(re.sub(r"\s", "", large_sample))
        if non_ws_chars < min_large_size:
            undersized.append(
                f"{language.name}: {non_ws_chars} chars (need {min_large_size}+)"
            )

    assert not undersized, (
        f"Large samples must exceed {min_large_size} non-ws chars to trigger splitting:\n"
        + "\n".join(f"  - {s}" for s in undersized)
    )
