#!/usr/bin/env bash
set -euo pipefail

DB_DIR="${CHUNKHOUND_DATABASE__PATH:-/data/db}"
CONFIG_FILE="${CHUNKHOUND_CONFIG_FILE:-}"

if [ ! -f "${DB_DIR}/chunks.db" ]; then
    echo "[entrypoint] WARNING: ${DB_DIR}/chunks.db not found." >&2
    echo "[entrypoint] Mount a pre-built database volume at ${DB_DIR}." >&2
    echo "[entrypoint] Regex search will work once the DB is present; semantic search also requires embedding config." >&2
fi

echo "[entrypoint] Starting supergateway wrapping chunkhound mcp" >&2
echo "[entrypoint] Database: ${DB_DIR}" >&2
echo "[entrypoint] Config:   ${CONFIG_FILE:-<none>}" >&2
echo "[entrypoint] Port: ${PORT:-8080}" >&2

CONFIG_ARG=""
if [ -n "${CONFIG_FILE}" ]; then
    CONFIG_ARG="--config ${CONFIG_FILE}"
fi

exec npx -y supergateway \
    --stdio "chunkhound mcp --db ${DB_DIR} ${CONFIG_ARG} --no-daemon" \
    --port "${PORT:-8080}" \
    --cors \
    --outputTransport streamableHttp \
    --stateful \
    --sessionTimeout 120000 \
    --protocolVersion 2025-11-25
