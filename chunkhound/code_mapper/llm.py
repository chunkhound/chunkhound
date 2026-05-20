from __future__ import annotations

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.interfaces.llm_provider import LLMProvider
from chunkhound.llm_manager import LLMManager


def build_llm_metadata_and_map_hyde(
    *,
    config: Config,
    llm_manager: LLMManager | None,
) -> tuple[dict[str, str], LLMProvider | None]:
    """Capture LLM configuration snapshot and optional HyDE-planning override."""
    llm_meta: dict[str, str] = {}
    map_hyde_provider: LLMProvider | None = None

    if not config.llm:
        return llm_meta, map_hyde_provider

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

    map_hyde_provider_name = getattr(llm, "map_hyde_provider", None)
    map_hyde_model_name = getattr(llm, "map_hyde_model", None)
    map_hyde_effort = getattr(llm, "map_hyde_reasoning_effort", None)
    map_hyde_cfg = llm.get_provider_config_for_role("map_hyde")

    needs_custom_map_hyde = bool(
        map_hyde_provider_name or map_hyde_model_name or map_hyde_effort
    )

    if llm_manager is not None and needs_custom_map_hyde:
        try:
            map_hyde_provider = llm_manager.create_provider_for_config(map_hyde_cfg)

            llm_meta["map_hyde_provider"] = str(
                map_hyde_cfg.get("provider", map_hyde_provider.name)
            )
            llm_meta["map_hyde_model"] = str(
                map_hyde_cfg.get("model", map_hyde_provider.model)
            )
            if "reasoning_effort" in map_hyde_cfg:
                llm_meta["map_hyde_reasoning_effort"] = str(
                    map_hyde_cfg["reasoning_effort"]
                )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug(f"Code Mapper: failed to create HyDE planning provider: {exc}")
            map_hyde_provider = None

    if map_hyde_provider is None:
        if provider := map_hyde_cfg.get("provider"):
            llm_meta["map_hyde_provider"] = str(provider)
        if model := map_hyde_cfg.get("model"):
            llm_meta["map_hyde_model"] = str(model)
        if effort := map_hyde_cfg.get("reasoning_effort"):
            llm_meta["map_hyde_reasoning_effort"] = str(effort)

    return llm_meta, map_hyde_provider
