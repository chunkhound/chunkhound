"""Indexing must send valid embedding requests for every supported text language.

Provider limits apply to complete inputs, including any metadata. Parser size,
chunk boundaries, and header formats are not external contracts.
"""

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.types.common import Language
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from tests.fixtures.fake_providers import ValidatingEmbeddingProvider

# PDF needs a binary fixture rather than generated source text.
BINARY_CONTENT_LANGUAGES = {Language.PDF}
# These enum values do not represent supported on-disk source languages.
UNSUPPORTED_FILE_LANGUAGES = {Language.UNKNOWN, Language.GIT_DIFF}


def _make_large_statements(statement: str, count: int = 500) -> str:
    """Build a large input without prescribing its chunk boundaries."""
    return "\n".join([statement] * count)


# Language samples: (extension, large_sample, normal_sample)
# Exercise both large and small inputs without requiring a particular split.
LARGE_LANGUAGE_SAMPLES: dict[Language, tuple[str, str, str]] = {
    # === Programming Languages ===
    Language.PYTHON: (
        ".py",
        # Large: ~1500+ non-ws chars - triggers line-based split
        "def process_large_data(items):\n"
        '    """Process items."""\n' + _make_large_statements("    x = 1"),
        # Normal
        "def greet(name: str) -> str:\n"
        '    """Say hello."""\n'
        '    return f"Hello, {name}"',
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
        + "\n".join(f"  private field{index}: number = 0;" for index in range(500))
        + "\n}",
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
    Language.METAL: (
        ".metal",
        # MSL is C++14; the C++ grammar chunks structs/classes
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
    Language.ELIXIR: (
        ".ex",
        # Large module with many functions - triggers line-based split
        "defmodule DataProcessor do\n"
        + _make_large_statements("  def process(x), do: x + 1")
        + "\nend",
        # Normal
        'defmodule Greeter do\n  def hello(name), do: "Hello, #{name}"\nend',
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
    Language.POWERSHELL: (
        ".ps1",
        # Large function
        "function Get-Data {\n"
        + _make_large_statements("    $x = 1")
        + "\n    return $x\n}",
        # Normal
        "function Get-Greeting {\n"
        "    param([string]$Name)\n"
        '    return "Hello, $Name"\n}',
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
        "@interface Greeter : NSObject\n"
        "- (NSString *)greet:(NSString *)name;\n@end\n\n"
        "@implementation Greeter\n"
        "- (NSString *)greet:(NSString *)name {\n"
        '    return [NSString stringWithFormat:@"Hello, %@", name];\n'
        "}\n@end",
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
        "<script>\nexport default {\n"
        "  data() {\n    return { greeting: 'Hello' };\n  }\n"
        "};\n</script>",
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
        "# Hello World\n\nThis is a sample markdown file.\n\n"
        "## Features\n\n- Feature 1\n- Feature 2",
    ),
    # === Data/Configuration Languages ===
    Language.JSON: (
        ".json",
        # Large JSON
        '{\n  "items": [\n'
        + ",\n".join(['    {"id": ' + str(i) + "}" for i in range(200)])
        + "\n  ]\n}",
        # Normal
        '{\n  "name": "example",\n  "version": "1.0.0",\n'
        '  "description": "A sample JSON file"\n}',
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
    # === Query Languages ===
    Language.SQL: (
        ".sql",
        # Large: many-column CREATE TABLE triggers splitting
        "CREATE TABLE large_table (\n"
        + ",\n".join(f"    col_{i} VARCHAR(255) NOT NULL DEFAULT ''" for i in range(80))
        + "\n);",
        # Normal: simple CREATE TABLE
        "CREATE TABLE users (\n"
        "    id INTEGER PRIMARY KEY,\n"
        "    name VARCHAR(255) NOT NULL,\n"
        "    email VARCHAR(255) UNIQUE\n"
        ");",
    ),
    # === Web Languages ===
    Language.HTML: (
        ".html",
        # Large HTML document
        "<!DOCTYPE html>\n<html>\n<body>\n"
        + _make_large_statements("  <p>Content paragraph here</p>")
        + "\n</body>\n</html>",
        # Normal
        "<!DOCTYPE html>\n<html>\n<body>\n  <h1>Hello World</h1>\n</body>\n</html>",
    ),
    Language.CSS: (
        ".css",
        # Large CSS with many rules
        ".container {\n" + _make_large_statements("  color: red;") + "\n}",
        # Normal
        ".container {\n  display: flex;\n  color: blue;\n}",
    ),
    Language.SCSS: (
        ".scss",
        # Large SCSS with nested rules
        "$primary: blue;\n.container {\n"
        + _make_large_statements("  color: $primary;")
        + "\n}",
        # Normal
        "$color: blue;\n.container {\n  color: $color;\n  &:hover { opacity: 0.8; }\n}",
    ),
    Language.JINJA: (
        ".jinja",
        # Large Jinja template
        "<!DOCTYPE html>\n<html>\n<body>\n"
        + _make_large_statements("  <p>{{ item }}</p>")
        + "\n</body>\n</html>",
        # Normal
        "<!DOCTYPE html>\n<html>\n<body>\n  <h1>{{ title }}</h1>\n</body>\n</html>",
    ),
    # === Generic/Unknown ===
    Language.UNKNOWN: (
        ".unknown",
        # Large unknown
        "Unknown file content.\n" + _make_large_statements("Line of unknown content."),
        # Normal
        "Unknown file type content line one.\nSecond line of unknown content.",
    ),
    Language.GIT_DIFF: (
        ".patch",
        # In-memory diff pseudo-language; not a supported on-disk file type.
        "diff --git a/foo.py b/foo.py\n"
        + _make_large_statements("+    line of diff content"),
        # Normal
        "diff --git a/foo.py b/foo.py\n+added line\n-removed line\n",
    ),
    Language.TWINCAT: (
        ".TcPOU",
        # Large FUNCTION_BLOCK - triggers line-based split
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<TcPlcObject Version="1.1.0.1">\n'
        '  <POU Name="FB_Large" Id="{00000000-0000-0000-0000-000000000001}" '
        'SpecialFunc="None">\n'
        "    <Declaration><![CDATA[FUNCTION_BLOCK FB_Large\n"
        "VAR\nEND_VAR\n]]></Declaration>\n"
        "    <Implementation>\n      <ST><![CDATA["
        + _make_large_statements("    nX := nX + 1;")
        + "]]></ST>\n    </Implementation>\n  </POU>\n</TcPlcObject>",
        # Normal - small FUNCTION_BLOCK, no over-splitting
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<TcPlcObject Version="1.1.0.1">\n'
        '  <POU Name="FB_Greet" Id="{00000000-0000-0000-0000-000000000002}" '
        'SpecialFunc="None">\n'
        "    <Declaration><![CDATA[FUNCTION_BLOCK FB_Greet\n"
        "VAR\nEND_VAR\n]]></Declaration>\n"
        "    <Implementation>\n      <ST><![CDATA[nX := 1;\n]]></ST>\n"
        "    </Implementation>\n  </POU>\n</TcPlcObject>",
    ),
}


def embedding_test_config(tmp_path, db_path):
    return Config(
        target_dir=tmp_path,
        database={"path": str(db_path), "provider": "duckdb"},
        embedding={
            "provider": "openai",
            "api_key": "fake-api-key-not-used",
            "model": "text-embedding-3-small",
        },
    )


@pytest.fixture
async def validating_db(tmp_path):
    """Use a real database and validate final embedding requests."""
    db_path = tmp_path / "chunk_validation_test.duckdb"
    embedding_manager = EmbeddingManager()
    validating_provider = ValidatingEmbeddingProvider(dims=1536)
    embedding_manager.register_provider(validating_provider, set_default=True)
    services = create_services(
        db_path, embedding_test_config(tmp_path, db_path), embedding_manager
    )
    try:
        yield services, validating_provider, tmp_path
    finally:
        services.provider.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language",
    [
        lang
        for lang in LARGE_LANGUAGE_SAMPLES
        if lang not in BINARY_CONTENT_LANGUAGES | UNSUPPORTED_FILE_LANGUAGES
    ],
    ids=lambda lang: lang.value,
)
async def test_indexing_sends_valid_embedding_requests(validating_db, language):
    """Every input must index successfully and reach the provider without rejection."""
    services, provider, tmp_path = validating_db
    extension, large, normal = LARGE_LANGUAGE_SAMPLES[language]
    for label, sample in (("large", large), ("normal", normal)):
        source = tmp_path / f"{label}_{language.name.lower()}{extension}"
        source.write_text(sample, encoding="utf-8")
        await assert_embedding_requests(services, provider, source)


async def assert_embedding_requests(services, provider, source):
    provider.all_texts.clear()
    result = await services.indexing_coordinator.process_file(source)
    assert result["status"] == "success", result
    assert result["embedding_error"] is None, result
    assert result["embeddings_generated"] > 0, result
    assert provider.all_texts, f"No embedding request for {source.name}"


@pytest.mark.asyncio
async def test_oversized_input_produces_valid_embedding_requests(validating_db):
    services, provider, tmp_path = validating_db
    directory = tmp_path / ("long_source_path_" + "x" * 120)
    directory.mkdir()
    source = directory / "oversized.py"
    # Exceed the provider budget but stay far under the 128KB parse-timeout
    # domain: ~34KB payload -> est. ~8200 tokens > max_tokens (8192).
    payload = "oversized_content " * ((provider.max_tokens * 4) // 17 + 10)
    assert provider.estimate_tokens(payload) > provider.max_tokens
    source.write_text(f'def oversized():\n    return "{payload}"\n', encoding="utf-8")
    await assert_embedding_requests(services, provider, source)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language", list(UNSUPPORTED_FILE_LANGUAGES), ids=lambda lang: lang.value
)
async def test_unsupported_files_do_not_request_embeddings(validating_db, language):
    services, provider, tmp_path = validating_db
    extension, sample, _ = LARGE_LANGUAGE_SAMPLES[language]
    source = tmp_path / f"unsupported{extension}"
    source.write_text(sample, encoding="utf-8")
    result = await services.indexing_coordinator.process_file(source)
    assert result["status"] == "skipped", result
    assert not provider.all_texts


def test_all_parsers_covered_with_large_samples():
    """New supported languages must receive embedding request coverage."""
    all_languages = set(Language)
    covered_languages = set(LARGE_LANGUAGE_SAMPLES.keys())

    missing_languages = all_languages - covered_languages
    assert not missing_languages, (
        f"Missing test samples for languages: "
        f"{sorted(lang.name for lang in missing_languages)}. "
        "Every Language enum member MUST have an entry in LARGE_LANGUAGE_SAMPLES."
    )

    extra_languages = covered_languages - all_languages
    assert not extra_languages, (
        f"LARGE_LANGUAGE_SAMPLES contains languages not in Language enum: "
        f"{sorted(lang.name for lang in extra_languages)}"
    )
