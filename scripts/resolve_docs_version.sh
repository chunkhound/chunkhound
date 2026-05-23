#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${GITHUB_ENV:-}" ]]; then
  echo "GITHUB_ENV must be set for docs version export" >&2
  exit 1
fi

if ! tag="$(git describe --tags --abbrev=0 2>/dev/null)"; then
  echo "Unable to resolve docs version: git describe --tags --abbrev=0 failed" >&2
  exit 1
fi

version="${tag#v}"

if [[ -z "$version" ]]; then
  echo "Unable to resolve docs version: derived version is empty" >&2
  exit 1
fi

printf 'CHUNKHOUND_DOCS_VERSION=%s\n' "$version" >> "$GITHUB_ENV"
