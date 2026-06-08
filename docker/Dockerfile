FROM python:3.14-slim-bookworm

# System deps: Node 20 (for npx/supergateway) + curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install chunkhound from PyPI (no source code in image).
# Python 3.14 is within chunkhound's supported range (>=3.10,<3.15).
# If native dependency wheels (duckdb, tree-sitter) are not yet published for 3.14,
# add build deps before this step: apt-get install -y build-essential cmake
ARG CHUNKHOUND_VERSION=latest
RUN if [ "$CHUNKHOUND_VERSION" = "latest" ]; then \
        pip install --no-cache-dir chunkhound; \
    else \
        pip install --no-cache-dir "chunkhound==${CHUNKHOUND_VERSION}"; \
    fi

# Pre-warm supergateway in the npx cache (avoids ~5-10s download on first container start).
# If the npm registry is unreachable at build time, this line is safe to remove;
# the container will download supergateway on first startup instead.
RUN npx -y supergateway --version 2>/dev/null || true

RUN useradd --create-home --shell /bin/bash chunkhound
USER chunkhound
WORKDIR /home/chunkhound

# Operator mounts a pre-built DuckDB directory here.
VOLUME ["/data/db"]

COPY --chown=chunkhound:chunkhound docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
