from pathlib import Path

from chunkhound.autodoc.hyde import build_hyde_scope_prompt
from chunkhound.autodoc.models import AgentDocMetadata, HydeConfig


def test_hyde_scope_prompt_includes_snippets_when_project_root_diff_cwd(
    tmp_path: Path,
) -> None:
    """HyDE scope prompt should include code snippets even when project_root != CWD."""
    project_root = tmp_path / "workspace"
    scope_path = project_root / "arguseek"
    scope_path.mkdir(parents=True, exist_ok=True)

    (scope_path / "foo.py").write_text(
        "def example() -> None:\n"
        "    x = 1\n"
        "    y = 2\n"
        "    return x + y\n",
        encoding="utf-8",
    )

    assert project_root.resolve() != Path.cwd().resolve()

    meta = AgentDocMetadata(
        created_from_sha="TEST_SHA",
        previous_target_sha="TEST_SHA",
        target_sha="TEST_SHA",
        generated_at="2025-01-01T00:00:00Z",
        llm_config={},
        generation_stats={},
    )
    hyde_cfg = HydeConfig.from_env()

    template = (
        "created={created}\n"
        "scope={scope_display}\n"
        "files:\n{files_block}\n"
        "snips:\n{code_context_block}\n"
    )

    prompt = build_hyde_scope_prompt(
        meta=meta,
        scope_label="arguseek",
        file_paths=["arguseek/foo.py"],
        hyde_cfg=hyde_cfg,
        project_root=project_root,
        template=template,
    )

    assert "File: arguseek/foo.py" in prompt
    assert "(no sample code snippets available)" not in prompt

