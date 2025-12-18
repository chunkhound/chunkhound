import subprocess
from pathlib import Path

from chunkhound.autodoc.models import HydeConfig
from chunkhound.autodoc.scope import collect_scope_files


def test_collect_scope_files_skips_gitignored_binaries(tmp_path: Path) -> None:
    """HyDE scope file collection should respect .gitignore rules."""
    project_root = tmp_path / "workspace"
    scope_path = project_root / "arguseek"
    bin_dir = scope_path / "bin"
    src_dir = scope_path / "src"

    src_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    # Included file
    (src_dir / "foo.py").write_text("print('ok')\n", encoding="utf-8")

    # Ignored file
    (bin_dir / "server").write_text(
        "#!/usr/bin/env bash\necho 'server'\n", encoding="utf-8"
    )

    project_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init"],
        cwd=str(project_root),
        check=True,
        capture_output=True,
        text=True,
    )

    (scope_path / ".gitignore").write_text("bin/\n", encoding="utf-8")

    hyde_cfg = HydeConfig.from_env()
    file_paths = collect_scope_files(
        scope_path=scope_path,
        project_root=project_root,
        hyde_cfg=hyde_cfg,
    )

    assert "arguseek/src/foo.py" in file_paths
    assert "arguseek/bin/server" not in file_paths
