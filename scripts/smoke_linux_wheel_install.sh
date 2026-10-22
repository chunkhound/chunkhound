#!/usr/bin/env bash
# Install the branch main + native wheels into a clean Linux container and
# smoke-test them as a user would. Run inside the container by ci.yml's
# old-linux-wheel-smoke job with the repo mounted at /work and wheels at /dist.
#
# Usage: smoke_linux_wheel_install.sh <uv-python-request> <expected-default-backend>
#
# CI native wheels keep Cargo.toml's placeholder version, which cannot satisfy
# the main wheel's chunkhound-native>=X requirement. Install the main wheel
# without dependencies, resolve its other declared dependencies, then install
# the branch native wheel in place of the excluded one.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <uv-python-request> <expected-default-backend>" >&2
  exit 2
fi
python_request="$1"
expected_backend="$2"

# The static musl uv has no glibc requirement and shares the build's pin.
bash /work/scripts/install_pinned_musl_uv.sh /tmp/uv
uv=/tmp/uv/uv
"$uv" venv --python "$python_request" /tmp/venv
py=/tmp/venv/bin/python
"$uv" pip install --python "$py" --no-cache-dir --no-deps /dist/chunkhound-*.whl
"$py" - <<'PY' > /tmp/chunkhound-requirements.txt
from importlib.metadata import requires

for requirement in requires("chunkhound") or []:
    if not requirement.lower().startswith("chunkhound-native"):
        print(requirement)
PY
"$uv" pip install --python "$py" --no-cache-dir -r /tmp/chunkhound-requirements.txt
"$uv" pip install --python "$py" --no-cache-dir --reinstall --no-deps \
  /dist/chunkhound_native-*.whl

"$py" /work/scripts/smoke_test_native_wheel.py
/tmp/venv/bin/chunkhound --version
# The packaged Watchman payload declares its own glibc floor; hosts below it
# must default to watchdog instead of starting an incompatible binary.
"$py" - "$expected_backend" <<'PY'
import sys

from chunkhound.watchman_runtime import default_realtime_backend_for_current_install

actual = default_realtime_backend_for_current_install()
if actual != sys.argv[1]:
    raise SystemExit(f"default realtime backend {actual!r} != {sys.argv[1]!r}")
print(f"default realtime backend: {actual}")
PY
