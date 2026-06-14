FROM python:3.14-slim-bookworm

# System deps: curl + procps (for pgrep in chunkhound-mcp-wrapper)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates procps \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install chunkhound from a pre-built wheel (built on the host with `uv build`).
# The wheel is built outside Docker so hatch-vcs can read the git tags.
# Run `uv build --wheel` before `docker compose build` to refresh the wheel.
COPY dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Claude Code CLI is mounted from the host at runtime (see docker-compose volumes).
# Node.js is still required for the claude CLI wrapper.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/local/lib/node_modules/@anthropic-ai/claude-code \
    && printf '#!/bin/sh\nexec node /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/run.js "$@"\n' \
        > /usr/local/bin/claude \
    && chmod +x /usr/local/bin/claude


COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd --gid ${HOST_GID} chunkhound 2>/dev/null || true && \
    useradd --uid ${HOST_UID} --gid ${HOST_GID} --create-home --shell /bin/bash chunkhound
USER chunkhound
WORKDIR /home/chunkhound

# Operator mounts a pre-built DuckDB directory here.
VOLUME ["/data/db"]

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
