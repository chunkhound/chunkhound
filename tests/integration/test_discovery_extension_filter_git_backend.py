"""Coverage for the discovery-time extension filter on the git-discovery
backend (`_discover_files_via_git`), which is a separate return point from
the Python os.walk path already covered in test_root_file_discovery_defaults.py.
"""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.utils.file_patterns import normalize_include_patterns

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


@pytest.mark.asyncio
async def test_git_backend_filters_unsupported_extensions_under_custom_include(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    pkg_dir = repo / "src" / "pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "module.py").write_text("print('ok')\n")
    (pkg_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    _git(repo, "init")
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "CI")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")

    db = DuckDBProvider(":memory:", base_directory=repo)
    db.connect()

    parser = create_parser_for_language(Language.PYTHON)
    cfg = IndexingConfig(include=["src/**/*"], discovery_backend="git_only")
    coordinator = IndexingCoordinator(
        db,
        repo,
        None,
        {Language.PYTHON: parser},
        None,
        SimpleNamespace(indexing=cfg),
    )

    files = await coordinator._discover_files(
        repo,
        patterns=normalize_include_patterns(list(cfg.include)),
        exclude_patterns=[],
        parallel_discovery=False,
    )

    assert pkg_dir / "module.py" in files
    assert pkg_dir / "image.png" not in files, (
        f"Unsupported-extension file leaked through git-backend discovery. "
        f"Files: {[p.name for p in files]}"
    )
