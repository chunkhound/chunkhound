from __future__ import annotations

import os
from typing import Any

from chunkhound.core.config.config import Config
from chunkhound.interfaces.llm_provider import LLMProvider
from chunkhound.llm_manager import LLMManager


def build_llm_metadata_and_assembly(
    *,
    config: Config,
    llm_manager: LLMManager | None,
) -> tuple[dict[str, str], LLMProvider | None]:
    """Capture LLM configuration snapshot and optional assembly provider."""
    llm_meta: dict[str, str] = {}
    assembly_provider: LLMProvider | None = None

    if not config.llm:
        return llm_meta, assembly_provider

    llm = config.llm
    llm_meta["provider"] = llm.provider
    if llm.synthesis_provider:
        llm_meta["synthesis_provider"] = llm.synthesis_provider
    if llm.synthesis_model:
        llm_meta["synthesis_model"] = llm.synthesis_model
    if llm.utility_model:
        llm_meta["utility_model"] = llm.utility_model
    if llm.codex_reasoning_effort_synthesis:
        llm_meta["codex_reasoning_effort_synthesis"] = (
            llm.codex_reasoning_effort_synthesis
        )
    if llm.codex_reasoning_effort_utility:
        llm_meta["codex_reasoning_effort_utility"] = llm.codex_reasoning_effort_utility

    assembly_provider_name = os.getenv("CH_AGENT_DOC_ASSEMBLY_PROVIDER")
    assembly_model_name = os.getenv("CH_AGENT_DOC_ASSEMBLY_MODEL")
    assembly_effort = os.getenv("CH_AGENT_DOC_ASSEMBLY_REASONING_EFFORT")

    if not assembly_provider_name and getattr(llm, "assembly_provider", None):
        assembly_provider_name = llm.assembly_provider
    if not assembly_model_name and getattr(llm, "assembly_model", None):
        assembly_model_name = llm.assembly_model
    if not assembly_model_name and getattr(llm, "assembly_synthesis_model", None):
        assembly_model_name = llm.assembly_synthesis_model
    if not assembly_effort and getattr(llm, "assembly_reasoning_effort", None):
        assembly_effort = llm.assembly_reasoning_effort

    _utility_cfg, synth_cfg = llm.get_provider_configs()

    needs_custom_assembly = bool(
        assembly_provider_name or assembly_model_name or assembly_effort
    )

    if llm_manager is not None and needs_custom_assembly:
        try:
            assembly_cfg: dict[str, Any] = synth_cfg.copy()
            if assembly_provider_name:
                assembly_cfg["provider"] = assembly_provider_name
            if assembly_model_name:
                assembly_cfg["model"] = assembly_model_name
            if assembly_effort:
                assembly_cfg["reasoning_effort"] = assembly_effort.strip().lower()

            assembly_provider = llm_manager.create_provider_for_config(assembly_cfg)

            llm_meta["assembly_synthesis_provider"] = str(
                assembly_cfg.get("provider", assembly_provider.name)
            )
            llm_meta["assembly_synthesis_model"] = str(
                assembly_cfg.get("model", assembly_provider.model)
            )
            if "reasoning_effort" in assembly_cfg:
                llm_meta["assembly_reasoning_effort"] = str(
                    assembly_cfg["reasoning_effort"]
                )
        except Exception:
            assembly_provider = None

    if assembly_provider is None:
        synth_provider = synth_cfg.get("provider")
        synth_model = synth_cfg.get("model")
        if synth_provider:
            llm_meta["assembly_synthesis_provider"] = str(synth_provider)
        if synth_model:
            llm_meta["assembly_synthesis_model"] = str(synth_model)
        effort = synth_cfg.get("reasoning_effort")
        if effort:
            llm_meta["assembly_reasoning_effort"] = str(effort)

    return llm_meta, assembly_provider
