from __future__ import annotations

import pytest
from packaging.tags import Tag

import hatch_build


def test_load_supported_watchman_platforms_matches_declared_slots() -> None:
    assert hatch_build._load_supported_watchman_platforms() == {
        "linux-x86_64",
        "windows-x86_64",
    }


@pytest.mark.parametrize(
    ("system_name", "machine_name", "expected_platform"),
    [
        ("Linux", "amd64", "linux-x86_64"),
        ("Windows", "AMD64", "windows-x86_64"),
    ],
)
def test_require_supported_build_host_accepts_declared_slots(
    system_name: str,
    machine_name: str,
    expected_platform: str,
) -> None:
    supported_platforms = hatch_build._load_supported_watchman_platforms()

    assert (
        hatch_build._require_supported_build_host(
            supported_platforms,
            system_name=system_name,
            machine_name=machine_name,
        )
        == expected_platform
    )


def test_require_supported_build_host_rejects_unsupported_host() -> None:
    supported_platforms = hatch_build._load_supported_watchman_platforms()

    with pytest.raises(RuntimeError, match="linux-arm64"):
        hatch_build._require_supported_build_host(
            supported_platforms,
            system_name="Linux",
            machine_name="aarch64",
        )


def test_custom_build_hook_hydrates_runtime_for_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setenv(hatch_build._PACKAGED_RUNTIME_BUILD_ENV, "1")
    monkeypatch.setattr(
        hatch_build,
        "_manifest_force_include_entries",
        lambda: pytest.fail(
            "release wheel builds should rely on hydrated host payload entries only"
        ),
    )
    monkeypatch.setattr(
        hatch_build,
        "_host_watchman_platform",
        lambda **_: "linux-x86_64",
    )
    monkeypatch.setattr(
        hatch_build,
        "_hydrate_runtime_for_build",
        lambda: (calls.append("hydrated") or {"src": "dst"}),
    )
    monkeypatch.setattr(
        hatch_build,
        "_allowed_wheel_platform_tags_for_build_host",
        lambda **_: {"manylinux_2_31_x86_64"},
    )
    monkeypatch.setattr(
        hatch_build,
        "_platform_only_tag",
        lambda allowed_platform_tags=None: "py3-none-manylinux_2_31_x86_64",
    )

    build_data: dict[str, object] = {}
    hook = object.__new__(hatch_build.CustomBuildHook)
    hook.initialize("standard", build_data)

    assert calls == ["hydrated"]
    assert build_data["force_include"] == {"src": "dst"}
    assert build_data["pure_python"] is False
    assert isinstance(build_data["tag"], str)


def test_custom_build_hook_skips_native_runtime_for_supported_editable_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hatch_build._PACKAGED_RUNTIME_BUILD_ENV, "1")
    monkeypatch.setattr(
        hatch_build,
        "_manifest_force_include_entries",
        lambda: pytest.fail(
            "editable builds should not add manifest force-include entries"
        ),
    )
    monkeypatch.setattr(
        hatch_build,
        "_host_watchman_platform",
        lambda **_: "linux-x86_64",
    )
    monkeypatch.setattr(
        hatch_build,
        "_hydrate_runtime_for_build",
        lambda: pytest.fail(
            "editable builds should skip native runtime hydration on supported hosts"
        ),
    )

    build_data: dict[str, object] = {"force_include": {"existing": "entry"}}
    hook = object.__new__(hatch_build.CustomBuildHook)
    hook.initialize("editable", build_data)

    assert build_data == {"force_include": {"existing": "entry"}}


def test_custom_build_hook_skips_native_runtime_for_unsupported_editable_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hatch_build._PACKAGED_RUNTIME_BUILD_ENV, "1")
    monkeypatch.setattr(
        hatch_build,
        "_manifest_force_include_entries",
        lambda: pytest.fail(
            "editable builds should not add manifest force-include entries"
        ),
    )
    monkeypatch.setattr(
        hatch_build,
        "_host_watchman_platform",
        lambda **_: "macos-arm64",
    )
    monkeypatch.setattr(
        hatch_build,
        "_hydrate_runtime_for_build",
        lambda: pytest.fail(
            "unsupported editable builds should skip native runtime hydration"
        ),
    )

    build_data: dict[str, object] = {"force_include": {"existing": "entry"}}
    hook = object.__new__(hatch_build.CustomBuildHook)
    hook.initialize("editable", build_data)

    assert build_data == {"force_include": {"existing": "entry"}}


def test_platform_only_tag_prefers_declared_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hatch_build.tags,
        "sys_tags",
        lambda: iter(
            [
                Tag("py3", "none", "manylinux_2_39_x86_64"),
                Tag("py3", "none", "manylinux_2_31_x86_64"),
            ]
        ),
    )

    assert (
        hatch_build._platform_only_tag({"manylinux_2_31_x86_64"})
        == "py3-none-manylinux_2_31_x86_64"
    )


def test_platform_only_tag_rejects_when_declared_runtime_tag_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hatch_build.tags,
        "sys_tags",
        lambda: iter([Tag("py3", "none", "manylinux_2_39_x86_64")]),
    )

    with pytest.raises(RuntimeError, match="manylinux_2_31_x86_64"):
        hatch_build._platform_only_tag({"manylinux_2_31_x86_64"})


def test_custom_build_hook_skips_packaged_runtime_when_release_env_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(hatch_build._PACKAGED_RUNTIME_BUILD_ENV, raising=False)
    monkeypatch.setattr(
        hatch_build,
        "_manifest_force_include_entries",
        lambda: {"manifest-src": "manifest-dst"},
    )
    monkeypatch.setattr(
        hatch_build,
        "_host_watchman_platform",
        lambda **_: "linux-x86_64",
    )
    monkeypatch.setattr(
        hatch_build,
        "_hydrate_runtime_for_build",
        lambda: pytest.fail(
            "standard builds without the release env should skip hydration"
        ),
    )

    build_data: dict[str, object] = {"force_include": {"existing": "entry"}}
    hook = object.__new__(hatch_build.CustomBuildHook)
    hook.initialize("standard", build_data)

    assert build_data == {
        "force_include": {
            "existing": "entry",
            "manifest-src": "manifest-dst",
        }
    }


def test_custom_build_hook_rejects_unsupported_release_wheel_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hatch_build._PACKAGED_RUNTIME_BUILD_ENV, "1")
    monkeypatch.setattr(
        hatch_build,
        "_manifest_force_include_entries",
        lambda: pytest.fail(
            "unsupported release wheel hosts should fail before adding "
            "fallback manifests"
        ),
    )
    monkeypatch.setattr(
        hatch_build,
        "_host_watchman_platform",
        lambda **_: "macos-arm64",
    )
    monkeypatch.setattr(
        hatch_build,
        "_hydrate_runtime_for_build",
        lambda: pytest.fail("unsupported wheel builds should fail before hydration"),
    )

    build_data: dict[str, object] = {"force_include": {"existing": "entry"}}
    hook = object.__new__(hatch_build.CustomBuildHook)
    with pytest.raises(RuntimeError, match="macos-arm64"):
        hook.initialize("standard", build_data)

    assert build_data == {"force_include": {"existing": "entry"}}


def test_manifest_force_include_entries_cover_supported_platforms() -> None:
    entries = hatch_build._manifest_force_include_entries()

    linux_manifest = (
        hatch_build._REPO_ROOT
        / "chunkhound"
        / "watchman_runtime"
        / "platforms"
        / "linux-x86_64"
        / "manifest.json"
    )
    windows_manifest = (
        hatch_build._REPO_ROOT
        / "chunkhound"
        / "watchman_runtime"
        / "platforms"
        / "windows-x86_64"
        / "manifest.json"
    )

    assert entries[str(linux_manifest)] == (
        "chunkhound/watchman_runtime/platforms/linux-x86_64/manifest.json"
    )
    assert entries[str(windows_manifest)] == (
        "chunkhound/watchman_runtime/platforms/windows-x86_64/manifest.json"
    )


def test_metadata_hook_links_package_to_built_commit() -> None:
    metadata: dict = {"urls": {"Homepage": "https://example.test"}}

    hatch_build._add_source_commit_url(metadata, hatch_build._REPO_ROOT)

    sha = hatch_build._source_commit(hatch_build._REPO_ROOT)
    assert sha is not None
    assert metadata["urls"]["Source commit"] == (
        f"https://github.com/chunkhound/chunkhound/commit/{sha}"
    )
    assert metadata["urls"]["Homepage"] == "https://example.test"


def test_metadata_hook_adds_no_link_outside_git(tmp_path) -> None:
    metadata: dict = {}

    hatch_build._add_source_commit_url(metadata, tmp_path)

    assert "urls" not in metadata or "Source commit" not in metadata["urls"]


def test_native_wheel_gets_source_commit_link(tmp_path) -> None:
    import base64
    import hashlib
    import zipfile

    from scripts import add_source_commit_link

    wheel = tmp_path / "chunkhound_native-0.0.0-py3-none-any.whl"
    dist = "chunkhound_native-0.0.0.dist-info"
    metadata_text = "Metadata-Version: 2.4\nName: chunkhound-native\n\nbody\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("chunkhound_native/__init__.py", "")
        archive.writestr(f"{dist}/METADATA", metadata_text)
        archive.writestr(f"{dist}/RECORD", "")

    add_source_commit_link.add_source_commit_link(wheel, hatch_build._REPO_ROOT)

    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(f"{dist}/METADATA").decode()
        record = archive.read(f"{dist}/RECORD").decode()
        for name in archive.namelist():
            if name == f"{dist}/RECORD":
                continue
            digest = base64.urlsafe_b64encode(
                hashlib.sha256(archive.read(name)).digest()
            ).rstrip(b"=")
            assert f"{name},sha256={digest.decode()}," in record

    sha = hatch_build._source_commit(hatch_build._REPO_ROOT)
    assert metadata.count("Project-URL: Source commit, ") == 1
    assert f"/commit/{sha}" in metadata
