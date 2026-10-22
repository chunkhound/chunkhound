# Test isolation and live opt-in

The suite is offline by default. Do not add network calls to ordinary tests.

## Defaults

- `tests/helpers/isolation.py` gives each test a private HOME, cwd, XDG dirs,
  and native runtime copy. It strips provider credentials and proxy settings.
- `tests/helpers/network.py` blocks outbound connect/send/DNS. Loopback and
  Unix sockets stay available.
- `@pytest.mark.live` tests are deselected unless `--run-live` is passed.
- Linux CI wraps pytest in `scripts/run_offline.sh` (private network and PID
  namespaces). macOS and Windows get only the Python socket guard; subprocess
  and native traffic there is not isolated.

## Native artifact cache

Shared VSS and Watchman artifacts live outside pytest's session
`TemporaryDirectory`, under
`<system temp>/chunkhound-pytest-artifacts/run-<pid>-<uuid>/`. The DuckDB VSS
DLL stays mapped until the pytest process exits. Windows cannot delete a mapped
DLL, so deleting the cache during pytest teardown would fail with
`PermissionError`.

Each test HOME links to the one frozen VSS tree. If Windows denies symlink
creation, the harness creates a directory junction. It never copies the DLL
into a temp sandbox because that recreates the Windows cleanup lock. A later
pytest run deletes run caches whose lock is free, which proves the loader
process has exited.

## Running offline

```bash
uv run python scripts/prepare_offline_tests.py
uv run pytest tests/
```

On Linux, add the OS firewall:

```bash
bash scripts/run_offline.sh uv run --offline --no-sync pytest tests/
```

`run_offline.sh` needs Linux and passwordless `sudo -n`. Prepare artifacts
before the network goes away. The script fails loudly when `CI` is set and
sudo is unavailable.

## Running live tests

```bash
uv run pytest tests/ -q --run-live --live-config /abs/path/live.json
```

`--live-config` must be an absolute path to a JSON object. Missing inputs fail;
they never skip silently. Example:

```json
{
  "embedding": {"provider": "openai", "api_key": "sk-...", "model": "text-embedding-3-small"},
  "anthropic": {"api_key": "sk-ant-..."},
  "codex": {"model": "gpt-5-codex", "auth": {"OPENAI_API_KEY": "..."}},
  "opencode": {"models": ["provider/model"]},
  "environment": {"OPENAI_API_KEY": "..."}
}
```

`environment` accepts only `OPENAI_`, `ANTHROPIC_`, `OPENCODE_`, and
`CHUNKHOUND_LLM_` keys. Other keys fail the test.

## Controls

Runner switches are captured before sanitization. Read them with
`tests.helpers.isolation.run_control`; never read ambient `os.environ` in a
test control.

Current controls live in `CONTROL_NAMES` in `tests/helpers/isolation.py`:
heavy tests, Watchman validation, HNSW requirements, perf tests, HCL and
RapidYAML opt-outs, process-pool tests, Codex binary, docs version, and the
site-dist reuse flag.

When adding a control:

1. Add its exact name to `CONTROL_NAMES`.
2. Read it with `run_control(name, default)`.
3. Add an assertion to the isolation contract probe when the control is
   security- or coverage-sensitive.
