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
  x86_64)
    uv_arch=x86_64
    uv_sha256=6c86040a8f045ab9adc9e3ec4cdafa03fd00e9a988ed2bf28971c7fc2d0debd0
    ;;
  aarch64)
    uv_arch=aarch64
    uv_sha256=345a8308e4d1835cdfaadf692836fd3137a6d45864773f05ad7bf59442b1ba16
    ;;
  *) echo "Unsupported Linux arch for uv: $(uname -m)" >&2; exit 1 ;;
esac

archive="$(mktemp)"
trap 'rm -f "$archive"' EXIT
curl --fail --location --silent --show-error \
  "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-${uv_arch}-unknown-linux-musl.tar.gz" \
  --output "$archive"
printf '%s  %s\n' "$uv_sha256" "$archive" | sha256sum --check --status

mkdir -p "$1"
tar -xzf "$archive" --strip-components=1 -C "$1"
