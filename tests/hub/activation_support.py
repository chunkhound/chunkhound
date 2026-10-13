import json
import os
import subprocess
from pathlib import Path

from chunkhound.core.config.config import Config
from tests.hub.artifacts import pack_database

ROOT = Path(__file__).resolve().parents[2]


def config_for(project, hub, backend="duckdb", model="hub-test"):
    project.mkdir(parents=True, exist_ok=True)
    return Config(
        target_dir=project,
        database={"path": project / ".chunkhound" / "db", "provider": backend},
        embedding={
            "provider": "openai",
            "model": model,
            "output_dims": 8,
            "api_key": "test",
            "base_url": "http://127.0.0.1:1/v1",
        },
        hub={"url": hub.url + "/catalogue?token=secret"},
        skip_layers={"global", "env", "local_config", "config_file", "cli"},
    )


def publish(hub, path, *, rel="latest", model="hub-test", prefix="base"):
    archive = path.parent / f"{prefix}.tar.gz"
    metadata = pack_database(path, archive)
    route = f"/{prefix}.tar.gz?token=secret"
    hub.resources[route] = archive.read_bytes()
    variant = {
        "embedding_id": "test-base",
        "db_provider": "duckdb" if path.name == "chunks.db" else "lancedb",
        "provider": "openai",
        "model": model,
        "dims": 8,
        "index_format_version": 1,
        "versions": [{"rel": rel, "artifact": {"url": hub.url + route, **metadata}}],
    }
    manifest = {
        "manifest_version": 2,
        "project_id": "fixture-project",
        "embeddings": [variant],
    }
    hub.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()
    return manifest, route


def cli_env(project, rust="0"):
    global_config = project.parent / "empty-global.json"
    global_config.write_text("{}", encoding="utf-8")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CHUNKHOUND_", "OPENAI_", "VOYAGE_"))
    }
    env.update(
        UV_NO_SYNC="1",
        UV_PROJECT=str(ROOT),
        CHUNKHOUND_GLOBAL_CONFIG_FILE=str(global_config),
        CHUNKHOUND_USE_RUST=rust,
        CHUNKHOUND_NO_PROMPTS="1",
        CHUNKHOUND_NO_MODEL_SUGGESTIONS="1",
        CHUNKHOUND_NO_RICH="1",
        NO_COLOR="1",
    )
    return env


def cli(project, *args, rust="0"):
    return subprocess.run(
        ["uv", "run", "chunkhound", *map(str, args)],
        cwd=project,
        env=cli_env(project, rust),
        text=True,
        capture_output=True,
        timeout=180,
    )


def write_config(project, backend, hub_url, embedding_url, model="hub-test"):
    project.mkdir(parents=True, exist_ok=True)
    data = {
        "database": {"provider": backend, "path": str(project / ".chunkhound" / "db")},
        "embedding": {
            "provider": "openai",
            "model": model,
            "output_dims": 8,
            "api_key": "test",
            "batch_size": 100,
            "base_url": embedding_url,
        },
        "indexing": {"include": ["*.py"], "exclude": [".chunkhound/**"]},
        "analytics": {"enabled": False},
        "hub": {"url": hub_url},
    }
    (project / ".chunkhound.json").write_text(json.dumps(data), encoding="utf-8")
    return data
