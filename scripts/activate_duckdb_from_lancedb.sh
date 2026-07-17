#!/usr/bin/env bash
# Activate DuckDB from a LanceDB ChunkHound index.
#
# Forwards all args to convert_lancedb_to_duckdb.py via uv, without changing
# the caller's working directory (so --project . means *your* project).
#
# What --activate does:
#   Converts Lance → DuckDB (unless --activate-only), then updates
#   <project>/.chunkhound.json:
#     database.provider = "duckdb"
#     database.path     = .chunkhound   (or --database-path / explicit dest)
#   Search/MCP then open chunks.db as a native DuckDB index. Lance is kept.
#
# Examples (run from the project you indexed, or pass absolute --project):
#   /path/to/chunkhound/scripts/activate_duckdb_from_lancedb.sh --project . --activate --overwrite
#   ./scripts/activate_duckdb_from_lancedb.sh --project /data/myrepo --activate --overwrite
#   ./scripts/activate_duckdb_from_lancedb.sh --project . --activate-only
#   ./scripts/activate_duckdb_from_lancedb.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CALLER_CWD="$(pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found on PATH (required to run the converter)" >&2
  exit 1
fi

# Absolutize relative path flags against the caller's CWD (not the repo root).
# uv --directory may set project root; we must not rewrite caller's ".".
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project|--source|--dest)
      flag="$1"
      shift
      if [[ $# -eq 0 ]]; then
        echo "error: ${flag} requires a value" >&2
        exit 2
      fi
      val="$1"
      shift
      if [[ "$val" != /* && "$val" != [A-Za-z]:* ]]; then
        val="${CALLER_CWD}/${val}"
      fi
      args+=("${flag}" "${val}")
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

# Run Python with this repo as the uv project, keeping process CWD = caller.
# --directory is used only for project discovery; paths above are absolute.
exec uv run --directory "${REPO_ROOT}" python \
  "${SCRIPT_DIR}/convert_lancedb_to_duckdb.py" "${args[@]}"
