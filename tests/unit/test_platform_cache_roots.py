"""User-visible cache locations of independent persisted features."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from chunkhound.providers.llm.capability_cache import LLMCapabilityStore
from chunkhound.watchman_runtime.loader import _default_runtime_cache_dir

_REPO_ROOT = Path(__file__).resolve().parents[2]

# hatch_build.py imports the watchman loader in the isolated build environment to
# hydrate the runtime and compute wheel tags. That environment installs only
# build-system requirements, so the loader's import graph -- including the cache
# root it resolves during hydration -- must not reach runtime-only parser deps.
_IMPORT_WITHOUT_PARSER_DEPS = """
import importlib.abc
import sys


class _BlockTreeSitterLanguagePack(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tree_sitter_language_pack" or fullname.startswith(
            "tree_sitter_language_pack."
        ):
            raise ModuleNotFoundError(f"No module named '{fullname}'")
        return None


sys.meta_path.insert(0, _BlockTreeSitterLanguagePack())
from chunkhound.watchman_runtime.loader import _default_runtime_cache_dir

# The build hydrates the runtime, which resolves this cache dir.
_default_runtime_cache_dir()
print("watchman-loader-import-ok")
"""


def test_watchman_loader_build_path_survives_missing_parser_dependencies() -> None:
    """The build-time loader path must not pull in runtime-only parser deps."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_PARSER_DEPS],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "watchman-loader-import-ok" in result.stdout


@pytest.mark.parametrize(
    ("platform", "environment", "relative_root"),
    [
        ("linux", "xdg", Path("xdg/chunkhound")),
        ("linux", "home", Path("home/.cache/chunkhound")),
        ("win32", "local-appdata", Path("local-appdata/ChunkHound")),
        ("win32", "home", Path("home/AppData/Local/ChunkHound")),
    ],
)
def test_platform_default_cache_root_shared_by_capabilities_and_watchman(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
    environment: str,
    relative_root: Path,
) -> None:
    monkeypatch.delenv("CHUNKHOUND_LLM_CAPABILITY_CACHE", raising=False)
    monkeypatch.delenv("CHUNKHOUND_WATCHMAN_RUNTIME_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    if environment != "home":
        env_name = "XDG_CACHE_HOME" if environment == "xdg" else "LOCALAPPDATA"
        monkeypatch.setenv(env_name, str(tmp_path / environment))

    expected = tmp_path / relative_root
    assert LLMCapabilityStore()._path == expected / "llm-capabilities.json"
    assert _default_runtime_cache_dir() == expected / "watchman-runtime"


def test_feature_specific_cache_overrides_remain_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    llm_path = tmp_path / "llm.json"
    watchman_path = tmp_path / "watchman"
    monkeypatch.setenv("CHUNKHOUND_LLM_CAPABILITY_CACHE", str(llm_path))
    monkeypatch.setenv("CHUNKHOUND_WATCHMAN_RUNTIME_CACHE_DIR", str(watchman_path))

    assert LLMCapabilityStore()._path == llm_path
    assert _default_runtime_cache_dir() == watchman_path
