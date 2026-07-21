#!/usr/bin/env bash
# Prepare per-replica database directories for the Docker MCP deployment.
#
# Run this once before `docker compose up`, and again whenever the indexed
# database is rebuilt (re-indexing the source codebase).
#
# Usage:
#   ./scripts/prepare-db-replicas.sh [--replicas N] <source-db-directory>
#
# The source directory must contain chunks.db (the indexed DuckDB file).
# Replica copies are written to /opt/chunkhound/db/replica{1..N}/.
#
# Why copies instead of a shared mount: DuckDB takes an exclusive write-lock
# on chunks.db at open time. Multiple containers cannot share one file.

set -euo pipefail

REPLICAS="${REPLICAS:-2}"
SOURCE_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --replicas|-r)
            REPLICAS="$2"
            shift 2
            ;;
        *)
            SOURCE_DIR="$1"
            shift
            ;;
    esac
done

: "${SOURCE_DIR:?Usage: $0 [--replicas N] <source-db-directory containing chunks.db>}"

if ! [[ "$REPLICAS" =~ ^[0-9]+$ ]] || [ "$REPLICAS" -lt 1 ]; then
    echo "ERROR: --replicas must be a positive integer, got: $REPLICAS" >&2
    exit 1
fi

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

# Resolve the UID/GID the container actually runs as, so ownership is correct
# on any machine regardless of local user mapping.
if [ -z "${HOST_UID:-}" ] || [ -z "${HOST_GID:-}" ]; then
    IMAGE=$(docker inspect chunkhound-mcp-1 --format '{{.Config.Image}}' 2>/dev/null || echo "chunkhound-mcp:latest")
    HOST_UID=$(docker run --rm "$IMAGE" id -u)
    HOST_GID=$(docker run --rm "$IMAGE" id -g)
fi
echo "Setting ownership to ${HOST_UID}:${HOST_GID} (from container image)..."
sudo chown -R "${HOST_UID}:${HOST_GID}" "$DEST_BASE"

echo ""
echo "Done. Start the stack with:"
echo "  docker compose --env-file .env.docker up -d --build"
