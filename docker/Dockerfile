FROM python:3.14-slim-bookworm

# System deps: Node 20 (for npx/supergateway) + curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install chunkhound from a pre-built wheel (built on the host with `uv build`).
# The wheel is built outside Docker so hatch-vcs can read the git tags.
# Run `uv build --wheel` before `docker compose build` to refresh the wheel.
COPY dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Pre-warm supergateway in the npx cache (avoids ~5-10s download on first container start).
RUN npx -y supergateway --version 2>/dev/null || true

# Claude Code CLI is mounted from the host at runtime (see docker-compose volumes).
# This wrapper calls node on the mounted module so no npm install is needed.
RUN mkdir -p /usr/local/lib/node_modules/@anthropic-ai/claude-code && \
    printf '#!/bin/sh\nexec node /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/run.js "$@"\n' \
        > /usr/local/bin/claude && \
    chmod +x /usr/local/bin/claude

RUN useradd --create-home --shell /bin/bash chunkhound
USER chunkhound
WORKDIR /home/chunkhound

# Operator mounts a pre-built DuckDB directory here.
VOLUME ["/data/db"]

COPY --chown=chunkhound:chunkhound docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
