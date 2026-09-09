"""Remote configuration sub-config for ChunkHound.

Holds the URL and optional Authorization header used by the remote-config
pipeline to fetch a settings envelope on every process start. The pipeline
lives in ``chunkhound.core.config.remote``; this module is intentionally
data-only.

Precedence rules:
- ``url`` and ``auth_header`` are accepted from CLI, env, or the global
  config file (``~/.chunkhound.json`` / ``CHUNKHOUND_GLOBAL_CONFIG_FILE``).
- ``.chunkhound.json`` and ``--config`` files with a ``remote_config`` key
  have that subtree scrubbed at load time with a WARNING (trust boundary in
  ``Config.__init__``) — a checked-in project file must not be able to
  redirect the operator's URL.
- The URL scheme is enforced by the fetcher: HTTPS is required, with a
  narrow loopback exception (``localhost``, ``127.0.0.0/8``, ``::1``) for
  local development. The rule applies per hop across any redirect chain,
  so an HTTPS endpoint can never be silently downgraded to cleartext.
- On a successful fetch the pipeline self-registers the source-layer's
  raw values back into the global JSON so subsequent runs converge to the
  same discovery inputs. Intentional trade-off: a literal ``auth_header``
  lands on disk in plaintext — use ``${VAR}`` to persist only the placeholder.

Startup-latency cost:
- Enabling ``url`` adds a synchronous fetch to every ChunkHound invocation
  that goes through ``create_validated_config`` — including short-lived
  commands like ``search`` and ``index``. The fetch has a hard 10-second
  wall-clock budget (see ``remote.fetcher``); a slow or unreachable server
  therefore adds up to 10s of startup delay per invocation. There is no
  per-invocation opt-out and no client-side cache — unset the URL (from
  the layer that supplies it) to disable for the current run.
- ``_quickresearch`` and ``_daemon`` skip the fetch; their parents
  (``websearch``, ``mcp``) already applied it and the child reads the result
  from disk. Without this, ``chunkhound websearch`` and MCP-proxy → daemon
  startup would each pay the 10s tax twice. The skip set lives on
  ``chunkhound.core.config.remote._SUBPROCESS_SKIP``; add any new child that
  inherits its parent's fetched config there.
"""

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RemoteConfig(BaseModel):
    """Discovery inputs for the remote-config fetch pipeline."""

    model_config = ConfigDict(extra="ignore")

    url: str | None = Field(
        default=None,
        description="URL to fetch the remote-config envelope from",
    )
    auth_header: str | None = Field(
        default=None,
        description=(
            "Optional Authorization header value. Supports ${VAR} interpolation "
            "against the process environment at fetch time."
        ),
    )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load remote-config settings from ``CHUNKHOUND_REMOTE_CONFIG__*`` env vars."""
        config: dict[str, Any] = {}

        if url := os.getenv("CHUNKHOUND_REMOTE_CONFIG__URL"):
            config["url"] = url
        if auth_header := os.getenv("CHUNKHOUND_REMOTE_CONFIG__AUTH_HEADER"):
            config["auth_header"] = auth_header

        return config

    @classmethod
    def extract_cli_overrides(cls, args: Any) -> dict[str, Any]:
        """Extract remote-config settings from CLI arguments."""
        overrides: dict[str, Any] = {}

        if getattr(args, "remote_config_url", None) is not None:
            overrides["url"] = args.remote_config_url
        if getattr(args, "remote_config_auth_header", None) is not None:
            overrides["auth_header"] = args.remote_config_auth_header

        return overrides
