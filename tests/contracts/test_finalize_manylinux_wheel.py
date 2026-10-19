"""Contract test — a Linux native wheel must not exceed its declared glibc floor.

scripts/finalize_manylinux_wheel.py is the gate that keeps the wheel's
manylinux tag honest: a wheel that requires a newer GLIBC symbol version than
--max-glibc must be rejected before it is published.
"""

import base64
import csv
import hashlib
import io
import shutil
import struct
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

# The high-floor fixture links against a synthetic provider which exports a
# GLIBC_2.32 version. That is deterministic across build hosts, unlike libc
# symbols whose versions depend on the host's glibc.
_HIGH_GLIBC_SOURCE = (
    "extern void high_versioned(void);\nvoid ext_entry(void) { high_versioned(); }\n"
)
_HIGH_GLIBC_PROVIDER_SOURCE = "void high_versioned(void) {}\n"
_HIGH_GLIBC_VERSION_SCRIPT = "GLIBC_2.32 { global: high_versioned; };\n"
_LOW_GLIBC_SOURCE = "int ext_entry(void) { return 0; }\n"
# getrandom was introduced in glibc 2.25.  The ELF symbol is made weak below
# so this mirrors Rust's optional syscall-wrapper reference.
_WEAK_GETRANDOM_SOURCE = (
    "#define _GNU_SOURCE\n#include <sys/random.h>\n"
    "void *ext_entry(void) { return (void *)&getrandom; }\n"
)


def _make_undefined_symbol_weak(elf: Path, symbol: str) -> None:
    """Change one ELF64 dynamic undefined symbol from GLOBAL to WEAK."""
    data = bytearray(elf.read_bytes())
    assert data[:5] == b"\x7fELF\x02" and data[5] == 1  # 64-bit little-endian ELF
    (section_offset,) = struct.unpack_from("<Q", data, 40)
    section_size, section_count = struct.unpack_from("<HH", data, 58)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, section_offset + i * section_size)
        for i in range(section_count)
    ]
    for section in sections:
        # SHT_DYNSYM; sh_link points at its associated string table.
        if section[1] != 11:
            continue
        strings = sections[section[6]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            (name_offset,) = struct.unpack_from("<I", data, offset)
            name_end = data.index(0, strings[4] + name_offset)
            if data[strings[4] + name_offset : name_end].decode() == symbol:
                data[offset + 4] = (data[offset + 4] & 0x0F) | 0x20  # STB_WEAK
                elf.write_bytes(data)
                return
    raise AssertionError(f"dynamic symbol {symbol!r} not found in {elf}")


def _record_hash(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={encoded.decode('ascii')}"


def _build_wheel(
    tmp_path: Path,
    source: str,
    *,
    make_weak_symbol: str | None = None,
    requires_high_glibc: bool = False,
) -> Path:
    src = tmp_path / "ext.c"
    src.write_text(source)
    so = tmp_path / "chunkhound_native.abi3.so"
    command = ["gcc", "-shared", "-fPIC", "-o", str(so), str(src)]
    if requires_high_glibc:
        provider = tmp_path / "libversioned-provider.so"
        provider_src = tmp_path / "provider.c"
        provider_map = tmp_path / "provider.map"
        provider_src.write_text(_HIGH_GLIBC_PROVIDER_SOURCE)
        provider_map.write_text(_HIGH_GLIBC_VERSION_SCRIPT)
        subprocess.run(
            [
                "gcc",
                "-shared",
                "-fPIC",
                "-Wl,--version-script",
                f"-Wl,{provider_map}",
                "-Wl,-soname,libversioned-provider.so",
                "-o",
                str(provider),
                str(provider_src),
            ],
            check=True,
            capture_output=True,
        )
        command.extend([f"-L{tmp_path}", "-lversioned-provider"])
    else:
        command.append("-ldl")
    subprocess.run(
        command,
        check=True,
        capture_output=True,
    )
    if make_weak_symbol:
        _make_undefined_symbol_weak(so, make_weak_symbol)
    wheel = tmp_path / "chunkhound_native-0.0.0-cp312-abi3-linux_x86_64.whl"
    dist = "chunkhound_native-0.0.0.dist-info"
    contents = {
        so.name: so.read_bytes(),
        f"{dist}/METADATA": (
            b"Metadata-Version: 2.1\nName: chunkhound-native\nVersion: 0.0.0\n"
        ),
        f"{dist}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\n",
    }
    record_name = f"{dist}/RECORD"
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for name, data in contents.items():
        writer.writerow((name, _record_hash(data), len(data)))
    writer.writerow((record_name, "", ""))
    with zipfile.ZipFile(wheel, "w") as zf:
        for name, data in contents.items():
            zf.writestr(name, data)
        zf.writestr(record_name, record.getvalue())
    return wheel


def _run_finalizer(wheel: Path, max_glibc: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(wheel), "--max-glibc", max_glibc],
        capture_output=True,
        text=True,
    )


@requires_toolchain
def test_wheel_requiring_glibc_above_floor_is_rejected(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path, _HIGH_GLIBC_SOURCE, requires_high_glibc=True)
    result = _run_finalizer(wheel, "2.31")
    assert result.returncode != 0
    assert "exceeds glibc 2.31 floor" in result.stderr


@requires_toolchain
def test_wheel_within_glibc_floor_is_accepted(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path, _LOW_GLIBC_SOURCE)
    result = _run_finalizer(wheel, "2.31")
    assert result.returncode == 0, result.stderr


@requires_toolchain
def test_weak_versioned_getrandom_is_unversioned_and_record_is_rewritten(
    tmp_path: Path,
) -> None:
    wheel = _build_wheel(tmp_path, _WEAK_GETRANDOM_SOURCE, make_weak_symbol="getrandom")
    with zipfile.ZipFile(wheel) as zf:
        extension = tmp_path / "before.abi3.so"
        extension.write_bytes(zf.read("chunkhound_native.abi3.so"))
    before = subprocess.run(
        ["readelf", "--dyn-syms", "-W", str(extension)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "WEAK" in next(line for line in before.splitlines() if "getrandom@" in line)

    result = _run_finalizer(wheel, "2.31")
    assert result.returncode == 0, result.stderr
    assert "cleared glibc version on weak symbol 'getrandom'" in result.stdout

    with zipfile.ZipFile(wheel) as zf:
        extension_data = zf.read("chunkhound_native.abi3.so")
        extension.write_bytes(extension_data)
        rows = list(
            csv.reader(
                io.StringIO(
                    zf.read("chunkhound_native-0.0.0.dist-info/RECORD").decode()
                )
            )
        )
    after = subprocess.run(
        ["readelf", "--dyn-syms", "-W", str(extension)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "getrandom@GLIBC_" not in after
    record = {row[0]: row[1:] for row in rows}
    assert record["chunkhound_native.abi3.so"] == [
        _record_hash(extension_data),
        str(len(extension_data)),
    ]


@requires_toolchain
def test_strong_versioned_getrandom_is_not_unversioned(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path, _WEAK_GETRANDOM_SOURCE)

    result = _run_finalizer(wheel, "2.24")

    assert result.returncode != 0
    assert "cleared glibc version" not in result.stdout
    assert "exceeds glibc 2.24 floor" in result.stderr


@requires_toolchain
def test_unversioned_weak_symbol_does_not_hide_above_floor_version_need(
    tmp_path: Path,
) -> None:
    wheel = _build_wheel(tmp_path, _WEAK_GETRANDOM_SOURCE, make_weak_symbol="getrandom")

    result = _run_finalizer(wheel, "2.24")

    assert result.returncode != 0
    assert "exceeds glibc 2.24 floor" in result.stderr
