"""Profile RapidYAML sanitizer and parser on a directory of YAML files.

Usage:
  uv run python -m chunkhound.scripts.profile_ryml_yaml /path/to/repo

This script intentionally bypasses the DB/service layers to focus on parser
hotspots. It will emit a one-line RapidYAML summary at the end via stdlib
logging from RapidYamlParser.cleanup().
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from chunkhound.core.types.common import FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory


def iter_yaml_files(root: Path):
    exts = {".yaml", ".yml"}
    for base, _, files in os.walk(root):
        for name in files:
            if Path(name).suffix.lower() in exts:
                yield Path(base) / name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="Root directory to scan for YAML files")
    ap.add_argument(
        "--limit", type=int, default=0, help="Optional limit of files to parse"
    )
    args = ap.parse_args()

    if not args.path.exists() or not args.path.is_dir():
        print(f"Not a directory: {args.path}", file=sys.stderr)
        return 2

    # Configure stdlib logging so RapidYamlParser INFO logs are visible
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    factory = ParserFactory()
    parser = factory.create_parser(Language.YAML)

    files = list(iter_yaml_files(args.path))
    if args.limit:
        files = files[: args.limit]

    t0 = time.perf_counter()
    total_chunks = 0
    errors = 0
    for idx, fp in enumerate(files, 1):
        try:
            chunks = parser.parse_file(fp, FileId(0))
            total_chunks += len(chunks)
            if idx % 500 == 0:
                print(f"… {idx}/{len(files)} files parsed; chunks={total_chunks}")
        except Exception as exc:  # pragma: no cover - diagnostic harness
            errors += 1
            print(f"ERROR parsing {fp}: {exc}", file=sys.stderr)

    # Force emission of RapidYAML summary/logs
    try:
        parser.cleanup()
    except Exception:
        pass

    dur = time.perf_counter() - t0
    print(
        f"Parsed {len(files)} files, chunks={total_chunks}, errors={errors}, time={dur:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
