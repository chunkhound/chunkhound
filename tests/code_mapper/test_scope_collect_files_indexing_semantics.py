from pathlib import Path

from chunkhound.code_mapper.models import HydeConfig
from chunkhound.code_mapper.scope import collect_scope_files


def test_collect_scope_files_respects_include_and_exclude(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    scope_path = project_root / "scope"
    src_dir = scope_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    (src_dir / "ok.py").write_text("print('ok')\n", encoding="utf-8")
    (src_dir / "secret.py").write_text("print('secret')\n", encoding="utf-8")
    (src_dir / "notes.xyz").write_text("not code\n", encoding="utf-8")

    hyde_cfg = HydeConfig(
        max_scope_files=1000,
        max_snippet_files=0,
        max_snippet_chars=0,
        max_completion_tokens=10,
        max_snippet_tokens=10,
    )

    file_paths = collect_scope_files(
        scope_path=scope_path,
        project_root=project_root,
        hyde_cfg=hyde_cfg,
        include_patterns=["**/*.py"],
        indexing_excludes=["scope/src/secret.py"],
        ignore_sources=["config"],
    )

    assert "scope/src/ok.py" in file_paths
    assert "scope/src/secret.py" not in file_paths
    assert "scope/src/notes.xyz" not in file_paths


def test_collect_scope_files_stops_at_max_scope_files(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    scope_path = project_root / "scope"
    src_dir = scope_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        (src_dir / f"f{i}.py").write_text("print('x')\n", encoding="utf-8")

    hyde_cfg = HydeConfig(
        max_scope_files=3,
        max_snippet_files=0,
        max_snippet_chars=0,
        max_completion_tokens=10,
        max_snippet_tokens=10,
    )

    file_paths = collect_scope_files(
        scope_path=scope_path,
        project_root=project_root,
        hyde_cfg=hyde_cfg,
        include_patterns=["**/*.py"],
        indexing_excludes=[],
        ignore_sources=["config"],
    )

    assert len(file_paths) == 3
