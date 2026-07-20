#!/usr/bin/env python3
"""Convert a ChunkHound LanceDB index into DuckDB (usable as a native Duck index).

Prefer the shell wrappers (same args forwarded):

  ./scripts/activate_duckdb_from_lancedb.sh --project . --activate --overwrite
  .\\scripts\\activate_duckdb_from_lancedb.ps1 --project . --activate --overwrite

Direct:

  uv run python scripts/convert_lancedb_to_duckdb.py --project . --activate --overwrite

Flags:
  --activate       After convert, rewrite .chunkhound.json to provider=duckdb
                   (path defaults to .chunkhound so chunks.db is used like a
                   normal DuckDB index). Does not remove Lance data.
  --activate-only  Only rewrite config (no conversion).
  --overwrite      Replace existing destination chunks.db.
  --batch-size N   Stream/insert batch size (default 2000; keep modest for 4M+ rows).
  --compact MODE   auto|always|never (default auto = product fragmentation check).
  --source / --dest  Override Lance source / Duck dest paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LanceDB ChunkHound index to DuckDB so search/MCP can use "
            "provider=duckdb as if the corpus was indexed into DuckDB."
        )
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Project root (default: CWD). Used for --activate and default paths.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Lance source: .lancedb dir, database config dir, or project root. "
            "Default: <project>/.chunkhound"
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=(
            "DuckDB destination file or directory (chunks.db). "
            "Default: <project>/.chunkhound"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination DuckDB file",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="After convert, set .chunkhound.json database.provider=duckdb",
    )
    parser.add_argument(
        "--activate-only",
        action="store_true",
        help="Only update config to duckdb (no conversion)",
    )
    parser.add_argument(
        "--database-path",
        type=str,
        default=None,
        help=(
            "Config database.path when activating. "
            "Default: derived from --dest (or .chunkhound for --activate-only)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print ConversionStats as JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help=(
            "Lance scan + Duck INSERT batch size in rows (default 2000). "
            "Lower if convert OOMs; higher can speed small indexes."
        ),
    )
    parser.add_argument(
        "--compact",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "After convert: auto = compact when fragmentation exceeds product "
            "threshold; always = force compact_database; never = skip"
        ),
    )
    parser.add_argument(
        "--allow-full-scan",
        action="store_true",
        help=(
            "If Lance streaming fails, allow full-table load (unsafe for "
            "multi-million-row indexes; default is refuse)"
        ),
    )
    args = parser.parse_args()

    from chunkhound.utils.lance_to_duckdb import (
        activate_duckdb_in_config,
        convert_and_activate,
        convert_lancedb_to_duckdb,
    )

    project = (args.project or Path.cwd()).expanduser().resolve()

    if args.activate_only:
        db_path = args.database_path or ".chunkhound"
        cfg = activate_duckdb_in_config(
            project,
            database_path=db_path,
            require_db_exists=True,
        )
        print(f"Activated DuckDB in config: {cfg}")
        print("  database.provider=duckdb")
        print(f"  database.path={db_path}")
        return 0

    source = args.source or (project / ".chunkhound")
    dest = args.dest or (project / ".chunkhound")

    from chunkhound.utils.lance_to_duckdb import config_path_for_dest

    if args.batch_size < 1:
        print("error: --batch-size must be >= 1", file=sys.stderr)
        return 2

    # Progress on stderr by default; silence when --json so stdout is JSON-only.
    common_kwargs: dict = {
        "overwrite": args.overwrite,
        "batch_size": args.batch_size,
        "compact": args.compact,
        "allow_full_scan": args.allow_full_scan,
    }
    if args.json:
        common_kwargs["progress"] = None

    if args.activate:
        stats = convert_and_activate(
            project,
            source=source,
            dest=dest,
            activate=True,
            database_path=args.database_path,
            **common_kwargs,
        )
        activated_path = args.database_path or config_path_for_dest(
            project, Path(stats.dest_duckdb)
        )
    else:
        stats = convert_lancedb_to_duckdb(
            source,
            dest,
            base_directory=project,
            **common_kwargs,
        )
        activated_path = None

    if args.json:
        print(json.dumps(stats.as_dict(), indent=2))
    else:
        print("=== LanceDB → DuckDB conversion ===")
        print(f"  source:  {stats.source_lance}")
        print(f"  dest:    {stats.dest_duckdb}")
        print(f"  files:   {stats.files}")
        print(f"  chunks:  {stats.chunks}")
        print(f"  embeds:  {stats.embeddings}  dims={stats.embedding_dims}")
        print(f"  batch:   {stats.stream_batch_size}")
        print(f"  compact: {stats.compacted}")
        if stats.skipped_invalid_embeddings:
            print(
                f"  skipped invalid embeddings: {stats.skipped_invalid_embeddings}"
            )
        if args.activate and activated_path is not None:
            print(
                f"  activated: {project / '.chunkhound.json'} "
                f"(provider=duckdb, path={activated_path})"
            )
        db_arg = Path(stats.dest_duckdb)
        print(f'\nUse with: chunkhound search "query" --db {db_arg}')
        print("  (or open MCP with database.provider=duckdb)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)
