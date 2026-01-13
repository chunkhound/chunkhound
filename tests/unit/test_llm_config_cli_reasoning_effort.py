"""CLI argument parsing for Codex reasoning effort."""

import argparse

from chunkhound.core.config.llm_config import LLMConfig


def test_cli_accepts_xhigh_reasoning_effort() -> None:
    parser = argparse.ArgumentParser()
    LLMConfig.add_cli_arguments(parser)

    args = parser.parse_args(
        [
            "--llm-codex-reasoning-effort",
            "xhigh",
            "--llm-codex-reasoning-effort-utility",
            "xhigh",
            "--llm-codex-reasoning-effort-synthesis",
            "xhigh",
        ]
    )

    assert args.llm_codex_reasoning_effort == "xhigh"
    assert args.llm_codex_reasoning_effort_utility == "xhigh"
    assert args.llm_codex_reasoning_effort_synthesis == "xhigh"
