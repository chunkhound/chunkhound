"""Verify a sidecar-free portable test DB works on the current OS.

Intended for cross-OS CI validation: the DB is created on Ubuntu without
chunks.db.root.json, downloaded as an artifact, and verified on all target OSs.
"""

import sys
from pathlib import Path

from chunkhound.providers.database.duckdb_provider import DuckDBProvider


def main() -> None:
    db_path = Path(".chunkhound/db/chunks.db")
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    sidecar = db_path.with_name(db_path.name + ".root.json")
    assert not sidecar.exists(), f"Expected no sidecar, found {sidecar}"

    provider = DuckDBProvider(db_path, base_directory=Path("."))
    provider.connect()

    try:
        files = provider.execute_query("SELECT path FROM files ORDER BY path")
        stored_paths = [row["path"] for row in files]
        assert stored_paths == ["lib/core/engine.py", "src/utils/helpers.py"], (
            stored_paths
        )

        for path in stored_paths:
            assert "\\" not in path, f"Backslash in path: {path}"
            assert not Path(path).is_absolute(), f"Absolute path: {path}"

        regex_results, _ = provider.search_regex(
            pattern="def|class", page_size=10, offset=0
        )
        assert len(regex_results) >= 2, (
            f"Expected regex results from both files, got {len(regex_results)}"
        )
        assert sorted({row["file_path"] for row in regex_results}) == stored_paths

        value = 1.0 / (8**0.5)
        query_vec = [value] * 8
        semantic_results, _ = provider.search_semantic(
            query_embedding=query_vec,
            provider="fake",
            model="test-model",
            page_size=10,
            offset=0,
        )
        assert len(semantic_results) >= 2, (
            f"Expected semantic results from both files, got {len(semantic_results)}"
        )
        assert sorted({row["file_path"] for row in semantic_results}) == stored_paths
    finally:
        provider.disconnect()

    print(f"Portable sidecar-free DB verified on {sys.platform}")


if __name__ == "__main__":
    main()
