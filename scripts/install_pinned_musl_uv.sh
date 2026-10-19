#!/usr/bin/env bash
# Install the static uv bootstrap used by the oldest-supported Linux build/smoke
# environments. Keep this pin shared so their libc compatibility cannot drift.
set -euo pipefail

UV_VERSION=0.13.1

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <destination>" >&2
  exit 2
fi

case "$(uname -m)" in
  x86_64 | aarch64) uv_arch="$(uname -m)" ;;
  *) echo "Unsupported Linux arch for uv: $(uname -m)" >&2; exit 1 ;;
esac

mkdir -p "$1"
curl -LsSf "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-${uv_arch}-unknown-linux-musl.tar.gz" \
  | tar -xz --strip-components=1 -C "$1"
