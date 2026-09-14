"""Keep logical symlinks without treating lexical parent escapes as lookup keys."""

from pathlib import Path

import pytest

from chunkhound.core.utils.path_utils import (
    get_relative_path_safe,
    normalize_path_for_lookup,
    normalize_realtime_path,
)


@pytest.mark.parametrize(
    "suffix", ["../outside.py", "sub/../../outside.py", "../sibling/file.py"]
)
@pytest.mark.parametrize("operation", ["relative", "lookup"])
def test_absolute_parent_escape_is_rejected(tmp_path, suffix, operation):
    base = tmp_path / "root"
    (base / "sub").mkdir(parents=True)
    (tmp_path / "outside.py").write_text("outside", encoding="utf-8")
    (tmp_path / "sibling").mkdir()
    (tmp_path / "sibling/file.py").write_text("outside", encoding="utf-8")
    path = base / suffix
    with pytest.raises(ValueError):
        if operation == "relative":
            get_relative_path_safe(path, base)
        else:
            normalize_path_for_lookup(path, base)


def test_absolute_missing_parent_escape_is_rejected(tmp_path):
    base = tmp_path / "root"
    base.mkdir()
    with pytest.raises(ValueError):
        normalize_path_for_lookup(base / "../does-not-exist.py", base)


def test_internal_parent_components_resolve_normally(tmp_path):
    base = tmp_path / "root"
    (base / "sub").mkdir(parents=True)
    (base / "file.py").write_text("inside", encoding="utf-8")
    assert normalize_path_for_lookup(base / "sub/../file.py", base) == "file.py"


@pytest.mark.parametrize("target_outside", [False, True])
def test_file_symlink_preserves_logical_name(tmp_path, target_outside):
    base = tmp_path / "root"
    base.mkdir()
    target = (tmp_path if target_outside else base) / "target.py"
    target.write_text("shared", encoding="utf-8")
    link = base / "alias.py"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    assert get_relative_path_safe(link, base) == Path("alias.py")
    assert normalize_path_for_lookup(link, base) == "alias.py"


def test_directory_symlink_preserves_logical_child(tmp_path):
    base = tmp_path / "root"
    target = tmp_path / "shared"
    base.mkdir()
    target.mkdir()
    (target / "file.py").write_text("shared", encoding="utf-8")
    link = base / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    assert normalize_path_for_lookup(link / "file.py", base) == "linked/file.py"
    assert normalize_realtime_path(link / "file.py", base) == link / "file.py"


def test_symlink_parent_escape_is_not_reinterpreted_lexically(tmp_path):
    base = tmp_path / "root"
    target = tmp_path / "shared/sub"
    base.mkdir()
    target.mkdir(parents=True)
    (target.parent / "outside.py").write_text("outside", encoding="utf-8")
    link = base / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    with pytest.raises(ValueError):
        normalize_path_for_lookup(link / "../outside.py", base)


def test_broken_symlink_preserves_logical_name(tmp_path):
    base = tmp_path / "root"
    base.mkdir()
    link = base / "alias.py"
    try:
        link.symlink_to(tmp_path / "missing.py")
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    assert normalize_path_for_lookup(link, base) == "alias.py"


def test_realtime_outside_path_uses_resolved_fallback(tmp_path):
    base = tmp_path / "root"
    base.mkdir()
    assert (
        normalize_realtime_path(base / "../outside.py", base)
        == (tmp_path / "outside.py").resolve()
    )


def test_unrelated_absolute_path_still_rejected(tmp_path):
    base = tmp_path / "root"
    base.mkdir()
    with pytest.raises(ValueError):
        normalize_path_for_lookup(tmp_path / "other.py", base)


def test_absolute_path_requires_base(tmp_path):
    with pytest.raises(ValueError, match="without base_dir"):
        normalize_path_for_lookup(tmp_path / "file.py")


@pytest.mark.parametrize(
    "path", ["file.py", "src/file.py", "../relative-contract-unchanged.py"]
)
def test_relative_lookup_contract_is_not_changed(path):
    assert normalize_path_for_lookup(path) == Path(path).as_posix()
