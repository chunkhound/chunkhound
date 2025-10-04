"""Tests for config file filtering (JSON/YAML/TOML) based on size and exclude patterns."""

import json
import pytest
from pathlib import Path

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language


@pytest.fixture
def real_components(tmp_path):
    """Real system components for testing."""
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    # Create parsers for config languages and Python
    json_parser = create_parser_for_language(Language.JSON)
    yaml_parser = create_parser_for_language(Language.YAML)
    toml_parser = create_parser_for_language(Language.TOML)
    python_parser = create_parser_for_language(Language.PYTHON)

    coordinator = IndexingCoordinator(
        db,
        tmp_path,
        None,
        {
            Language.JSON: json_parser,
            Language.YAML: yaml_parser,
            Language.TOML: toml_parser,
            Language.PYTHON: python_parser,
        }
    )

    return {"db": db, "coordinator": coordinator}


@pytest.mark.asyncio
async def test_json_size_threshold_filtering(tmp_path, real_components):
    """Test that JSON files > 20KB are skipped, <= 20KB are processed."""
    db = real_components["db"]
    coordinator = real_components["coordinator"]
    
    # Create small JSON file (< 20KB)
    small_json_path = tmp_path / "config.json"
    small_config = {
        "name": "test-project",
        "version": "1.0.0",
        "description": "Test configuration file",
        "dependencies": {f"package-{i}": f"{i}.0.0" for i in range(10)}
    }
    small_json_path.write_text(json.dumps(small_config, indent=2))
    
    # Create large JSON file (> 20KB)
    large_json_path = tmp_path / "data.json"
    # Create ~25KB of data
    large_data = {
        "data": "A" * 25000,
        "items": [{"id": i, "value": f"item-{i}"} for i in range(100)]
    }
    large_json_path.write_text(json.dumps(large_data))
    
    # Process small JSON file
    small_result = await coordinator.process_file(small_json_path)
    assert small_result["status"] in ["complete", "success"], f"Small JSON should be processed: {small_result}"
    assert small_result.get("chunks", 0) > 0, "Small JSON should produce chunks"
    
    # Process large JSON file
    large_result = await coordinator.process_file(large_json_path)
    assert large_result["status"] == "skipped", f"Large JSON should be skipped: {large_result}"
    assert large_result["reason"] == "large_config_file", f"Should skip for size reason: {large_result}"
    assert large_result.get("chunks", 0) == 0, "Large JSON should not produce chunks"
    
    # Verify in database by checking if file exists
    small_file = db.get_file_by_path(str(small_json_path))
    large_file = db.get_file_by_path(str(large_json_path))
    
    # Small file should be in DB
    assert small_file is not None, "Small JSON should be indexed"
    
    # Large file should NOT be in DB
    assert large_file is None, "Large JSON should not be indexed"


@pytest.mark.asyncio
async def test_json_exactly_at_threshold(tmp_path, real_components):
    """Test JSON file exactly at 20KB threshold is processed."""
    coordinator = real_components["coordinator"]
    
    # Create JSON exactly 20KB (20 * 1024 = 20480 bytes)
    edge_json_path = tmp_path / "edge.json"
    # Account for JSON overhead (quotes, brackets, etc.)
    target_size = 20 * 1024
    padding_size = target_size - 50  # Leave room for JSON structure
    edge_data = {"data": "X" * padding_size}
    edge_json_path.write_text(json.dumps(edge_data))
    
    # Verify it's exactly 20KB or just under
    actual_size = edge_json_path.stat().st_size
    assert actual_size <= target_size, f"File size {actual_size} should be <= {target_size}"
    
    # Process edge case file
    result = await coordinator.process_file(edge_json_path)
    assert result["status"] in ["complete", "success"], "20KB file should be processed"
    assert result.get("chunks", 0) > 0, "20KB file should produce chunks"


@pytest.mark.asyncio
async def test_yaml_size_threshold_filtering(tmp_path, real_components):
    """Test that YAML files > 20KB are skipped, <= 20KB are processed."""
    db = real_components["db"]
    coordinator = real_components["coordinator"]

    # Create small YAML file (< 20KB)
    small_yaml_path = tmp_path / "config.yaml"
    small_config = """
name: test-project
version: 1.0.0
description: Test configuration file
dependencies:
  package-1: 1.0.0
  package-2: 2.0.0
  package-3: 3.0.0
"""
    small_yaml_path.write_text(small_config)

    # Create large YAML file (> 20KB)
    large_yaml_path = tmp_path / "data.yaml"
    # Create ~25KB of YAML data
    large_data = "data: " + "A" * 25000 + "\n"
    large_data += "items:\n"
    for i in range(100):
        large_data += f"  - id: {i}\n    value: item-{i}\n"
    large_yaml_path.write_text(large_data)

    # Process small YAML file
    small_result = await coordinator.process_file(small_yaml_path)
    assert small_result["status"] in ["complete", "success"], f"Small YAML should be processed: {small_result}"
    assert small_result.get("chunks", 0) > 0, "Small YAML should produce chunks"

    # Process large YAML file
    large_result = await coordinator.process_file(large_yaml_path)
    assert large_result["status"] == "skipped", f"Large YAML should be skipped: {large_result}"
    assert large_result["reason"] == "large_config_file", f"Should skip for size reason: {large_result}"
    assert large_result.get("chunks", 0) == 0, "Large YAML should not produce chunks"

    # Verify in database
    small_file = db.get_file_by_path(str(small_yaml_path))
    large_file = db.get_file_by_path(str(large_yaml_path))

    assert small_file is not None, "Small YAML should be indexed"
    assert large_file is None, "Large YAML should not be indexed"


@pytest.mark.asyncio
async def test_toml_size_threshold_filtering(tmp_path, real_components):
    """Test that TOML files > 20KB are skipped, <= 20KB are processed."""
    db = real_components["db"]
    coordinator = real_components["coordinator"]

    # Create small TOML file (< 20KB)
    small_toml_path = tmp_path / "config.toml"
    small_config = """
[package]
name = "test-project"
version = "1.0.0"
description = "Test configuration file"

[dependencies]
package-1 = "1.0.0"
package-2 = "2.0.0"
package-3 = "3.0.0"
"""
    small_toml_path.write_text(small_config)

    # Create large TOML file (> 20KB)
    large_toml_path = tmp_path / "data.toml"
    # Create ~25KB of TOML data
    large_data = 'data = "' + "A" * 25000 + '"\n'
    large_data += "\n[[items]]\n"
    for i in range(100):
        large_data += f'id = {i}\nvalue = "item-{i}"\n\n[[items]]\n'
    large_toml_path.write_text(large_data)

    # Process small TOML file
    small_result = await coordinator.process_file(small_toml_path)
    assert small_result["status"] in ["complete", "success"], f"Small TOML should be processed: {small_result}"
    assert small_result.get("chunks", 0) > 0, "Small TOML should produce chunks"

    # Process large TOML file
    large_result = await coordinator.process_file(large_toml_path)
    assert large_result["status"] == "skipped", f"Large TOML should be skipped: {large_result}"
    assert large_result["reason"] == "large_config_file", f"Should skip for size reason: {large_result}"
    assert large_result.get("chunks", 0) == 0, "Large TOML should not produce chunks"

    # Verify in database
    small_file = db.get_file_by_path(str(small_toml_path))
    large_file = db.get_file_by_path(str(large_toml_path))

    assert small_file is not None, "Small TOML should be indexed"
    assert large_file is None, "Large TOML should not be indexed"


@pytest.mark.asyncio
async def test_json_data_file_excludes(tmp_path):
    """Test that common JSON data files are excluded by default config."""
    from chunkhound.core.config.indexing_config import IndexingConfig
    
    # Get default exclude patterns
    config = IndexingConfig()
    
    # Check each file against exclude patterns
    from fnmatch import fnmatch
    
    test_files = [
        "package-lock.json",  # Should be excluded
        "yarn.lock",  # Should be excluded
        "composer.lock",  # Should be excluded  
        "assets.json",  # Should be excluded
        "bundle.map.json",  # Should be excluded
        "data.min.json",  # Should be excluded
        "config.json",  # Should NOT be excluded
    ]
    
    for filename in test_files:
        is_excluded = False
        for pattern in config.exclude:
            # Simple pattern matching - remove **/ prefix for filename matching
            simple_pattern = pattern.replace("**/", "")
            if fnmatch(filename, simple_pattern):
                is_excluded = True
                break
        
        if filename == "config.json":
            assert not is_excluded, "config.json should NOT be excluded"
        else:
            assert is_excluded, f"{filename} should be excluded"


@pytest.mark.asyncio
async def test_json_filtering_in_directory_processing(tmp_path, real_components):
    """Test that JSON filtering works during bulk directory processing."""
    coordinator = real_components["coordinator"]
    db = real_components["db"]
    
    # Create mixed directory structure
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # Small config files (should be indexed)
    (project_dir / "tsconfig.json").write_text(json.dumps({
        "compilerOptions": {"target": "es2020", "module": "commonjs"}
    }))
    
    (project_dir / ".eslintrc.json").write_text(json.dumps({
        "extends": "standard", "rules": {}
    }))
    
    # Large data file (should be skipped)
    large_data = {"data": "B" * 30000, "entries": list(range(1000))}
    (project_dir / "cache.json").write_text(json.dumps(large_data))
    
    # Excluded by pattern
    (project_dir / "package-lock.json").write_text(json.dumps({
        "lockfileVersion": 2, "packages": {}
    }))
    
    # Python file for comparison
    (project_dir / "main.py").write_text("def main():\n    pass\n")
    
    # Process directory with patterns
    from chunkhound.core.config.indexing_config import IndexingConfig
    config = IndexingConfig()
    patterns = ["**/*.json", "**/*.py"]
    
    # Filter out patterns that exclude CI temp directories
    exclude_patterns = [p for p in config.exclude if p not in ["**/tmp/**", "**/temp/**"]]
    
    result = await coordinator.process_directory(
        project_dir, 
        patterns=patterns,
        exclude_patterns=exclude_patterns
    )
    
    # Debug info for CI troubleshooting
    import os
    if os.environ.get("CI") == "true":
        print(f"\n=== DEBUG INFO ===")
        print(f"project_dir: {project_dir}")
        print(f"result: {result}")
        # Get all files in database for debugging
        try:
            # Query database directly through provider
            search_results, _ = db.search_regex(pattern=".*", page_size=100)
            files_in_db = [result.get('file_path', result.get('path', '')) for result in search_results]
            print(f"Files in database: {files_in_db}")
        except Exception as e:
            print(f"Error querying database: {e}")
        print(f"==================")

    # Check what was indexed using file paths (matching pattern from working tests)
    tsconfig_file = db.get_file_by_path(str(project_dir / "tsconfig.json"))
    eslint_file = db.get_file_by_path(str(project_dir / ".eslintrc.json"))
    python_file = db.get_file_by_path(str(project_dir / "main.py"))
    cache_file = db.get_file_by_path(str(project_dir / "cache.json"))
    lock_file = db.get_file_by_path(str(project_dir / "package-lock.json"))
    
    # Should be indexed
    assert tsconfig_file is not None, "Small config should be indexed"
    assert eslint_file is not None, "ESLint config should be indexed"
    assert python_file is not None, "Python file should be indexed"
    
    # Should NOT be indexed
    assert cache_file is None, "Large JSON should not be indexed (>20KB)"
    assert lock_file is None, "Lock file should not be indexed (excluded pattern)"