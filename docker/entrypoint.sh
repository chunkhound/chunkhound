#!/usr/bin/env bash
set -euo pipefail

DB_DIR="${CHUNKHOUND_DATABASE__PATH:-/data/db}"
CONFIG_FILE="${CHUNKHOUND_CONFIG_FILE:-}"

if [ ! -f "${DB_DIR}/chunks.db" ]; then
    echo "[entrypoint] WARNING: ${DB_DIR}/chunks.db not found." >&2
    echo "[entrypoint] Mount a pre-built database volume at ${DB_DIR}." >&2
    echo "[entrypoint] Regex search will work once the DB is present; semantic search also requires embedding config." >&2
fi

# Pre-warm the OS page cache so chunkhound mcp doesn't cold-start on the first
# connection after a container restart. Runs at idle I/O priority so it doesn't
# compete with DuckDB when the first MCP session opens the database.
if [ -f "${DB_DIR}/chunks.db" ]; then
    ionice -c 3 dd if="${DB_DIR}/chunks.db" of=/dev/null bs=4M 2>/dev/null &
fi

echo "[entrypoint] Starting chunkhound HTTP MCP server" >&2
echo "[entrypoint] Database: ${DB_DIR}" >&2
echo "[entrypoint] Config:   ${CONFIG_FILE:-<none>}" >&2
echo "[entrypoint] Port: ${PORT:-8080}" >&2

CONFIG_ARG=""
if [ -n "${CONFIG_FILE}" ]; then
    CONFIG_ARG="--config ${CONFIG_FILE}"
fi

exec chunkhound mcp \
    --transport http \
    --port "${PORT:-8080}" \
    --host 0.0.0.0 \
    --db "${DB_DIR}" \
    ${CONFIG_ARG}
