from __future__ import annotations

from chunkhound.autodoc.models import CodeMapperIndex


def _parse_metadata_block(metadata: str) -> dict[str, object]:
    data: dict[str, object] = {}
    current_top: str | None = None
    current_sub: str | None = None

    for raw in metadata.splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()

        if indent == 2:
            current_top = key
            current_sub = None
            if value:
                data[key] = value
            else:
                data.setdefault(key, {})
            continue

        if indent == 4:
            if current_top == "llm_config":
                llm_config = data.setdefault("llm_config", {})
                if isinstance(llm_config, dict):
                    llm_config[key] = value
            elif current_top == "generation_stats":
                generation_stats = data.setdefault("generation_stats", {})
                if isinstance(generation_stats, dict):
                    if value:
                        generation_stats[key] = value
                        current_sub = None
                    else:
                        generation_stats.setdefault(key, {})
                        current_sub = key
            elif current_top:
                section = data.setdefault(current_top, {})
                if isinstance(section, dict):
                    section[key] = value
            continue

        if indent >= 6 and current_top == "generation_stats" and current_sub:
            generation_stats = data.setdefault("generation_stats", {})
            if isinstance(generation_stats, dict):
                subsection = generation_stats.setdefault(current_sub, {})
                if isinstance(subsection, dict):
                    subsection[key] = value

    return data


def _render_index_metadata(index: CodeMapperIndex) -> list[str]:
    if not index.metadata_block:
        return []

    metadata = _parse_metadata_block(index.metadata_block)
    lines: list[str] = []

    generated_at = metadata.get("generated_at")
    if isinstance(generated_at, str):
        lines.append(f"- Generated at: {generated_at}")

    created_from_sha = metadata.get("created_from_sha")
    if isinstance(created_from_sha, str):
        lines.append(f"- Source SHA: {created_from_sha}")

    llm_config = metadata.get("llm_config")
    if isinstance(llm_config, dict):
        provider = llm_config.get("provider") or llm_config.get("synthesis_provider")
        if provider:
            lines.append(f"- LLM provider: {provider}")
        synthesis_provider = llm_config.get("synthesis_provider")
        if synthesis_provider and synthesis_provider != provider:
            lines.append(f"- Synthesis provider: {synthesis_provider}")
        synthesis_model = llm_config.get("synthesis_model")
        if synthesis_model:
            lines.append(f"- Synthesis model: {synthesis_model}")
        utility_model = llm_config.get("utility_model")
        if utility_model:
            lines.append(f"- Utility model: {utility_model}")
        synth_effort = llm_config.get("codex_reasoning_effort_synthesis")
        if synth_effort:
            lines.append(f"- Synthesis reasoning effort: {synth_effort}")
        util_effort = llm_config.get("codex_reasoning_effort_utility")
        if util_effort:
            lines.append(f"- Utility reasoning effort: {util_effort}")
        hyde_provider = llm_config.get("map_hyde_provider")
        if hyde_provider:
            lines.append(f"- HyDE planning provider: {hyde_provider}")
        hyde_model = llm_config.get("map_hyde_model")
        if hyde_model:
            lines.append(f"- HyDE planning model: {hyde_model}")
        hyde_effort = llm_config.get("map_hyde_reasoning_effort")
        if hyde_effort:
            lines.append(f"- HyDE planning reasoning effort: {hyde_effort}")

    generation_stats = metadata.get("generation_stats")
    if isinstance(generation_stats, dict):
        generator_mode = generation_stats.get("generator_mode")
        if generator_mode:
            lines.append(f"- Generator mode: {generator_mode}")
        comprehensiveness = generation_stats.get(
            "autodoc_comprehensiveness"
        ) or generation_stats.get("code_mapper_comprehensiveness")
        if comprehensiveness:
            lines.append(f"- Comprehensiveness: {comprehensiveness}")
        total_calls = generation_stats.get("total_research_calls")
        if total_calls:
            lines.append(f"- Research calls: {total_calls}")

        files = generation_stats.get("files")
        if isinstance(files, dict):
            referenced = files.get("referenced")
            total = files.get("total_indexed")
            coverage = files.get("coverage")
            basis = files.get("basis")
            if referenced is not None and total is not None:
                detail = f"{referenced} / {total}"
                if coverage:
                    detail = f"{detail} ({coverage})"
                if basis:
                    detail = f"{detail}, basis: {basis}"
                lines.append(f"- Files referenced: {detail}")
            referenced_in_scope = files.get("referenced_in_scope")
            if referenced_in_scope is not None:
                lines.append(f"- Files referenced in scope: {referenced_in_scope}")
            unreferenced = files.get("unreferenced_in_scope")
            if unreferenced is not None:
                lines.append(f"- Files unreferenced in scope: {unreferenced}")

        chunks = generation_stats.get("chunks")
        if isinstance(chunks, dict):
            referenced = chunks.get("referenced")
            total = chunks.get("total_indexed")
            coverage = chunks.get("coverage")
            basis = chunks.get("basis")
            if referenced is not None and total is not None:
                detail = f"{referenced} / {total}"
                if coverage:
                    detail = f"{detail} ({coverage})"
                if basis:
                    detail = f"{detail}, basis: {basis}"
                lines.append(f"- Chunks referenced: {detail}")

    return lines

