"""Real publisher artifacts for hub tests; never imported by production code."""

import hashlib
import tarfile
from pathlib import Path

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.providers.database.lancedb_provider import LanceDBProvider


def build_database(
    root: Path,
    backend: str = "duckdb",
    provider: str = "openai",
    model: str = "hub-test",
    dims: int = 8,
) -> Path:
    """Publish a small real index with a full batch of stored embeddings."""
    root.mkdir(parents=True, exist_ok=True)
    if backend not in {"duckdb", "lancedb"}:
        raise ValueError("Unsupported database backend")
    name = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
    db = cls(root / name, base_directory=root)
    db.connect()
    try:
        file_id = db.insert_file(
            File(path="source.py", mtime=1.0, language=Language.PYTHON, size_bytes=64)
        )
        chunks = [
            Chunk(
                file_id=file_id,
                code=f"def function_{i}(): return {i}",
                start_line=i + 1,
                end_line=i + 1,
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
                symbol=f"function_{i}",
            )
            for i in range(100)
        ]
        ids = db.insert_chunks_batch(chunks)
        db.insert_embeddings_batch(
            [
                {
                    "chunk_id": cid,
                    "provider": provider,
                    "model": model,
                    "dims": dims,
                    "embedding": [0.25] * dims,
                }
                for cid in ids
            ]
        )
    finally:
        db.disconnect()
    return root / name


def pack_database(path: Path, archive: Path) -> dict[str, int | str]:
    """Package just the backend database with portable archive metadata."""

    def portable(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mode = 0
        info.mtime = 0
        return info

    with tarfile.open(archive, "w:gz") as tar:
        tar.add(path, arcname=path.name, filter=portable)
    content = archive.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
