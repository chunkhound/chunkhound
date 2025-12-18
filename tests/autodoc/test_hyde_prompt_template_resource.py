from chunkhound.autodoc.hyde import build_hyde_scope_prompt
from chunkhound.autodoc.models import AgentDocMetadata, HydeConfig


def test_build_hyde_scope_prompt_loads_packaged_template() -> None:
    meta = AgentDocMetadata(
        created_from_sha="AAA",
        previous_target_sha="AAA",
        target_sha="AAA",
        generated_at="2025-01-01T00:00:00Z",
        llm_config={},
        generation_stats={},
    )
    hyde_cfg = HydeConfig.from_env()

    prompt = build_hyde_scope_prompt(
        meta=meta,
        scope_label="/",
        file_paths=[],
        hyde_cfg=hyde_cfg,
        template=None,
        project_root=None,
    )

    assert "HyDE objective:" in prompt
    assert "created_from_sha: AAA" in prompt

