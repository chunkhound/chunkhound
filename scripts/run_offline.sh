#!/usr/bin/env bash
# Offline test runner: private network and PID namespaces isolate sockets
# and descendant teardown.
#
# WHY: Python socket guards stop in-process traffic only. Native code and
# child processes need OS isolation. This script provides it on Linux.
#
# PREREQS: Linux + passwordless `sudo -n` (CI runners have it; local dev may
# need a sudoers entry). This script refuses to run off Linux; CI leaves the
# runner empty on macOS/Windows and relies on the Python socket guard there.
# Run `uv run python scripts/prepare_offline_tests.py` first so DuckDB VSS
# and the Watchman binary exist before the network goes away.
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Offline network namespaces require Linux.' >&2
  exit 1
fi
if [[ $# -eq 0 ]]; then
  echo 'Usage: run_offline.sh COMMAND [ARG ...]' >&2
  exit 2
fi
exec sudo -n --preserve-env bash -euo pipefail -c '
  namespace_pid=
  stop_namespace() {
    trap "" HUP INT TERM
    if [[ -n $namespace_pid ]]; then
      kill -KILL "$namespace_pid" 2>/dev/null || true
      wait "$namespace_pid" 2>/dev/null || true
    fi
    exit "$1"
  }
  trap "stop_namespace 129" HUP
  trap "stop_namespace 130" INT
  trap "stop_namespace 143" TERM
  # unshare ignores INT/TERM while waiting. Kill it explicitly so --kill-child
  # kills namespace init; the kernel then kills only its namespace descendants.
  # 'offline' becomes the inner bash's $0; $1..$4 are uid/gid/HOME/PATH; the
  # rest is the command. setpriv drops root back to the invoking user.
  unshare --net --pid --fork --kill-child --mount-proc -- \
    bash -euo pipefail -c '\''
      ip link set lo up
      uid=$1 gid=$2; export HOME=$3 PATH=$4; shift 4
      exec setpriv --reuid="$uid" --regid="$gid" --init-groups -- "$@"
    '\'' offline "$@" <&0 &
  namespace_pid=$!
  status=0
  wait "$namespace_pid" || status=$?
  exit "$status"
' offline "$(id -u)" "$(id -g)" "$HOME" "$PATH" "$@"
