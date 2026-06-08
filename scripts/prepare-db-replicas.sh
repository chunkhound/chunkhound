#!/usr/bin/env bash
# Prepare per-replica database directories for the Docker MCP deployment.
#
# Run this once before `docker compose up`, and again whenever the indexed
# database is rebuilt (re-indexing the source codebase).
#
# Usage:
#   ./scripts/prepare-db-replicas.sh /path/to/.chunkhound/db
#
# The source directory must contain chunks.db (the indexed DuckDB file).
# Replica copies are written to /opt/chunkhound/db/replica{1..N}/.
#
# Why copies instead of a shared mount: DuckDB takes an exclusive write-lock
# on chunks.db at open time. Multiple containers cannot share one file.

set -euo pipefail

SOURCE_DIR="${1:?Usage: $0 <source-db-directory containing chunks.db>}"
REPLICAS="${REPLICAS:-3}"
DEST_BASE="/opt/chunkhound/db"

if [ ! -f "${SOURCE_DIR}/chunks.db" ]; then
    echo "ERROR: ${SOURCE_DIR}/chunks.db not found." >&2
    echo "Index your codebase first: uv run chunkhound index /path/to/project" >&2
    exit 1
fi

SOURCE_SIZE=$(du -sh "${SOURCE_DIR}/chunks.db" | cut -f1)
echo "Source: ${SOURCE_DIR}/chunks.db (${SOURCE_SIZE})"
echo "Preparing ${REPLICAS} replica(s) under ${DEST_BASE}..."

for i in $(seq 1 "$REPLICAS"); do
    DEST="${DEST_BASE}/replica${i}"
    mkdir -p "$DEST"
    cp "${SOURCE_DIR}/chunks.db" "${DEST}/chunks.db"
    echo "  Replica ${i}: ${DEST}/chunks.db"
done

echo ""
echo "Done. Start the stack with:"
echo "  docker compose up -d --build"
