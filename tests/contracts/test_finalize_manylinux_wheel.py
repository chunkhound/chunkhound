"""Contract test — a Linux native wheel must not exceed its declared glibc floor.

scripts/finalize_manylinux_wheel.py is the gate that keeps the wheel's
manylinux tag honest: a wheel that requires a newer GLIBC symbol version than
--max-glibc must be rejected before it is published.
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "finalize_manylinux_wheel.py"

requires_toolchain = pytest.mark.skipif(
    sys.platform != "linux"
    or not all(
        shutil.which(tool) for tool in ("gcc", "readelf", "objdump", "patchelf")
    ),
    reason="needs Linux with gcc, readelf, objdump and patchelf",
)

# dlopen is versioned GLIBC_2.34 in any glibc >= 2.34 build host, so an
# extension that references it requires glibc 2.34 at runtime.
_HIGH_GLIBC_SOURCE = (
    "#include <dlfcn.h>\nvoid *ext_entry(void) { return (void *)&dlopen; }\n"
)
_LOW_GLIBC_SOURCE = "int ext_entry(void) { return 0; }\n"


def _build_wheel(tmp_path: Path, source: str) -> Path:
    src = tmp_path / "ext.c"
    src.write_text(source)
    so = tmp_path / "chunkhound_native.abi3.so"
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-o", str(so), str(src), "-ldl"],
        check=True,
        capture_output=True,
    )
    wheel = tmp_path / "chunkhound_native-0.0.0-cp312-abi3-linux_x86_64.whl"
    dist = "chunkhound_native-0.0.0.dist-info"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.write(so, so.name)
        zf.writestr(
            f"{dist}/METADATA",
            "Metadata-Version: 2.1\nName: chunkhound-native\nVersion: 0.0.0\n",
        )
        zf.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n")
        zf.writestr(f"{dist}/RECORD", "")
    return wheel


def _run_finalizer(wheel: Path, max_glibc: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(wheel), "--max-glibc", max_glibc],
        capture_output=True,
        text=True,
    )


@requires_toolchain
def test_wheel_requiring_glibc_above_floor_is_rejected(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path, _HIGH_GLIBC_SOURCE)
    result = _run_finalizer(wheel, "2.31")
    assert result.returncode != 0
    assert "exceeds glibc 2.31 floor" in result.stderr


@requires_toolchain
def test_wheel_within_glibc_floor_is_accepted(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path, _LOW_GLIBC_SOURCE)
    result = _run_finalizer(wheel, "2.31")
    assert result.returncode == 0, result.stderr
