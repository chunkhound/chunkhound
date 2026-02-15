"""End-to-end integration test for Makefile parsing with cAST algorithm verification."""

import re

import pytest

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.search_service import SearchService


@pytest.fixture
def makefile_workflow(tmp_path):
    """Setup real components for Makefile integration testing."""
    # Real in-memory database
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    # Real parser and coordinator
    parser = create_parser_for_language(Language.MAKEFILE)
    coordinator = IndexingCoordinator(db, tmp_path, None, {Language.MAKEFILE: parser})
    search_service = SearchService(db)

    return {
        "db": db,
        "coordinator": coordinator,
        "search": search_service,
        "parser": parser,
    }


async def test_makefile_cast_algorithm_chunking(makefile_workflow, tmp_path):
    """Test that cAST produces optimal Makefile chunks."""

    # Create Makefile with cAST edge cases
    large_install_commands = "\n".join(
        [f"\tcp file{i}.dat $(DESTDIR)/share/data/" for i in range(30)]
    )

    makefile_content = f"""# Small variables that should merge
CC = gcc
CXX = g++
CFLAGS = -Wall
CXXFLAGS = -Wall -std=c++17

# Medium target that should stay intact
all: $(OBJECTS)
\t@echo "Building project..."
\t$(CC) -o myapp $(OBJECTS) $(LIBS)

# Large installation target that should split intelligently
install: all
\t@echo "Installing myapp..."
\tmkdir -p $(DESTDIR)/usr/local/bin
\tmkdir -p $(DESTDIR)/usr/local/lib/myapp
\tmkdir -p $(DESTDIR)/usr/local/share/myapp
\tmkdir -p $(DESTDIR)/usr/local/share/man/man1
\tmkdir -p $(DESTDIR)/etc/myapp
\tcp myapp $(DESTDIR)/usr/local/bin/
\tcp lib/*.so $(DESTDIR)/usr/local/lib/myapp/
\tcp share/*.dat $(DESTDIR)/usr/local/share/myapp/
\tcp man/myapp.1 $(DESTDIR)/usr/local/share/man/man1/
\tcp etc/config.conf $(DESTDIR)/etc/myapp/
{large_install_commands}
\tchmod +x $(DESTDIR)/usr/local/bin/myapp
\tchmod 644 $(DESTDIR)/etc/myapp/config.conf
\tldconfig $(DESTDIR)/usr/local/lib/myapp
\tupdate-desktop-database
\tgtk-update-icon-cache

# Tiny related targets that should merge
.PHONY: clean
clean:
\trm -f *.o

.PHONY: distclean
distclean: clean
\trm -f myapp
"""

    # Create file within the base directory
    makefile_path = tmp_path / "Makefile"
    makefile_path.write_text(makefile_content)

    # Index with cAST processing
    result = await makefile_workflow["coordinator"].process_file(makefile_path)
    assert result["status"] == "success"

    chunks = makefile_workflow["db"].get_chunks_by_file_id(
        result["file_id"], as_model=True
    )
    assert len(chunks) > 0

    # VERIFY: cAST merge behavior for small variables
    var_names = ["CC", "CXX", "CFLAGS", "CXXFLAGS"]
    variable_chunks = [c for c in chunks if c.symbol and c.symbol in var_names]
    # Should be merged, not 4 separate tiny chunks
    assert len(variable_chunks) <= 2, (
        f"Variables not merged efficiently: {len(variable_chunks)} chunks"
    )

    # VERIFY: Medium chunks stay intact
    all_chunks = [c for c in chunks if c.symbol == "all"]
    assert len(all_chunks) == 1, (
        f"Simple 'all' target should stay intact: {len(all_chunks)}"
    )

    # VERIFY: Chunk size constraints respected
    # (non-whitespace characters per cAST algorithm)
    cast_config = makefile_workflow["parser"].cast_config
    max_chunk_size = cast_config.max_chunk_size

    for chunk in chunks:
        non_ws_chars = len(re.sub(r"\s", "", chunk.code))
        assert non_ws_chars <= max_chunk_size, (
            f"Chunk exceeds size limit: {non_ws_chars} > {max_chunk_size}"
        )

    # VERIFY: Semantic coherence maintained
    for chunk in chunks:
        code = chunk.code.strip()
        if code:
            lines = code.split("\n")
            has_recipe_lines = any(line.startswith("\t") for line in lines)
            if has_recipe_lines:
                # Should have a target line (contains ':')
                has_target = any(
                    ":" in line and not line.strip().startswith("#") for line in lines
                )
                assert has_target, f"Recipe without target in chunk: {chunk.symbol}"

    # VERIFY: Search functionality works
    search = makefile_workflow["search"]

    # Search for variable assignment
    cc_results, _ = search.search_regex("CC.*gcc")
    assert len(cc_results) > 0, "Variable assignment not searchable"

    # Search for install commands
    install_results, _ = search.search_regex("mkdir.*DESTDIR")
    assert len(install_results) > 0, "Install commands not searchable"

    print("✅ cAST Algorithm Results:")
    print(f"   📦 Total chunks: {len(chunks)}")
    avg = sum(len(c.code) for c in chunks) / len(chunks)
    print(f"   📏 Avg chunk size: {avg:.0f} chars")
    print(f"   🔗 Variable chunks: {len(variable_chunks)} (merged efficiently)")


async def test_makefile_end_to_end_indexing(makefile_workflow, tmp_path):
    """Test complete Makefile indexing workflow with enhanced metadata."""

    makefile_content = """# Project build configuration
PROJECT = myapp
CC := gcc
CFLAGS += -Wall -Werror -O2

# Source files
SOURCES = main.c utils.c parser.c
OBJECTS = $(SOURCES:.c=.o)

# Pattern rule with automatic variables
%.o: %.c include/%.h
\t@echo "Compiling $<..."
\t$(CC) $(CFLAGS) -c $< -o $@

# Main target with dependencies
$(PROJECT): $(OBJECTS)
\t@echo "Linking $(PROJECT)..."
\t$(CC) $(OBJECTS) $(LDFLAGS) -o $@

# Conditional compilation
ifdef DEBUG
    CFLAGS += -g -DDEBUG
    debug: $(PROJECT)
\t@echo "Debug build complete"
endif

# Include dependencies
-include $(OBJECTS:.o=.d)

# Phony targets
.PHONY: all clean install test

all: $(PROJECT)

clean:
\t@echo "Cleaning build files..."
\trm -f $(OBJECTS) $(PROJECT) *.d

install: $(PROJECT)
\tcp $(PROJECT) /usr/local/bin/

test: $(PROJECT)
\t@echo "Running tests..."
\t./$(PROJECT) --test
"""

    # Create file within the base directory
    makefile_path = tmp_path / "Makefile"
    makefile_path.write_text(makefile_content)

    # INDEX: Process the file through the full pipeline
    result = await makefile_workflow["coordinator"].process_file(makefile_path)

    # Debug the result
    print(f"Debug - Result: {result}")

    # VERIFY: Indexing succeeded
    assert result["status"] == "success", (
        f"Indexing failed: {result.get('error')}, full result: {result}"
    )
    assert result["chunks"] > 0, "No chunks were created"
    file_id = result["file_id"]

    # VERIFY: Chunks stored in database
    chunks = makefile_workflow["db"].get_chunks_by_file_id(file_id, as_model=True)
    assert len(chunks) > 0, "No chunks found in database"

    # VERIFY: Language detection
    for chunk in chunks:
        assert chunk.language == Language.MAKEFILE, f"Wrong language: {chunk.language}"

    # VERIFY: cAST algorithm is working (chunks are created and may be merged)
    chunk_symbols = [chunk.symbol for chunk in chunks if chunk.symbol]
    assert len(chunk_symbols) >= 1, f"No symbols extracted: {chunk_symbols}"

    # VERIFY: Enhanced Makefile constructs are captured in content
    all_content = " ".join(chunk.code for chunk in chunks)

    # Check for various Makefile constructs in the content
    makefile_features = [
        "PROJECT = myapp",  # Variable assignment
        "CC := gcc",  # Variable assignment with :=
        "CFLAGS +=",  # Append assignment
        "%.o: %.c",  # Pattern rule
        "$(CC)",  # Variable reference
        "@echo",  # Silent command
        ".PHONY:",  # Special target
        "ifdef DEBUG",  # Conditional
        "-include",  # Include directive
    ]

    found_features = []
    for feature in makefile_features:
        if feature in all_content:
            found_features.append(feature)

    assert len(found_features) >= 5, (
        f"Too few Makefile features found: {found_features}"
    )

    # VERIFY: Search functionality works
    search = makefile_workflow["search"]

    # Search for pattern rule
    pattern_results, _ = search.search_regex(r"%.o.*%.c")
    assert len(pattern_results) > 0, "Pattern rule not searchable"

    # Search for variable assignment
    cc_results, _ = search.search_regex("CC.*gcc")
    assert len(cc_results) > 0, "Variable assignment not searchable"

    # Search for phony target
    phony_results, _ = search.search_regex(r"\.PHONY.*clean")
    assert len(phony_results) > 0, "Phony targets not searchable"

    print("✅ Successfully indexed Makefile:")
    print(f"   📁 File ID: {file_id}")
    print(f"   📦 Chunks: {len(chunks)}")
    print(f"   🏷️  Symbols: {len(chunk_symbols)}")
    print(f"   🔍 Features found: {len(found_features)}")
    print(f"   📝 Content length: {len(all_content)} chars")


async def test_makefile_semantic_boundaries(makefile_workflow, tmp_path):
    """Test that cAST preserves Makefile semantic boundaries."""

    long_var_name = "VERY_LONG_VARIABLE_NAME_THAT_MIGHT_CAUSE_SPLITTING"
    long_var_value = (
        "this is a very long value that goes on and on and might trigger"
        " the chunking algorithm to consider splitting but it should not"
        " split a variable assignment in the middle"
    )
    deps = " ".join(f"dep{i}.o" for i in range(1, 11))
    long_target = f"long-target-name-with-many-dependencies: {deps}"

    makefile_content = f"""# Variables should not be split from their values
{long_var_name} = {long_var_value}

# Target with recipe should stay together
{long_target}
\t@echo "This recipe belongs with its target"
\t$(CC) -o $@ $^
\t@echo "All recipe lines should stay with the target"

# Pattern rule should not be split
%.very-long-extension-name: %.c
\t@echo "Pattern rules are atomic semantic units"
\t$(CC) -c $< -o $@
"""

    # Create file within the base directory
    makefile_path = tmp_path / "Makefile"
    makefile_path.write_text(makefile_content)

    result = await makefile_workflow["coordinator"].process_file(makefile_path)
    assert result["status"] == "success"

    chunks = makefile_workflow["db"].get_chunks_by_file_id(
        result["file_id"], as_model=True
    )

    # VERIFY: Variable assignments are not split
    var_chunks = [c for c in chunks if long_var_name in c.code]
    if var_chunks:
        for chunk in var_chunks:
            # Should contain both name and value
            assert long_var_name in chunk.code
            assert "very long value" in chunk.code

    # VERIFY: Target and recipe stay together
    target_chunks = [c for c in chunks if "long-target-name" in c.code]
    if target_chunks:
        for chunk in target_chunks:
            # Should contain target line and at least some recipe
            assert ":" in chunk.code  # Target declaration
            assert "\t@echo" in chunk.code or "\t$(CC)" in chunk.code  # Recipe commands

    # VERIFY: Pattern rules are atomic
    pattern_chunks = [c for c in chunks if "%.very-long-extension-name" in c.code]
    if pattern_chunks:
        for chunk in pattern_chunks:
            # Should contain pattern and its recipe
            assert "%.very-long-extension-name: %.c" in chunk.code
            assert '\t@echo "Pattern rules are atomic' in chunk.code
