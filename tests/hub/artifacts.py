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


def replace_with_view(path: Path, table: str, external: bool) -> Path | None:
    """Replace owned storage by executable SQL; optionally depend on Parquet."""
    import duckdb

    outside = path.parent.parent / f"outside-{table}.parquet"
    with duckdb.connect(str(path)) as conn:
        if table == "files":
            # Remove the FK so the fixture can replace its parent table.
            conn.execute("CREATE TABLE chunks_copy AS SELECT * FROM chunks")
            conn.execute("DROP TABLE chunks")
            conn.execute("ALTER TABLE chunks_copy RENAME TO chunks")
        if external:
            conn.execute(f"COPY {table} TO '{outside}' (FORMAT PARQUET)")
            query = f"SELECT * FROM read_parquet('{outside}')"
            if table == "embeddings_8":
                query = (
                    "SELECT * REPLACE(embedding::FLOAT[8] AS embedding) "
                    f"FROM read_parquet('{outside}')"
                )
        elif table == "schema_version":
            query = "SELECT 1 AS version"
        elif table == "embeddings_8":
            query = (
                "SELECT 1 AS chunk_id, 'openai' AS provider, 'hub-test' AS model, "
                "[0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]::FLOAT[8] AS embedding, "
                "8 AS dims, TIMESTAMP '2026-01-01' AS created_at"
            )
        else:
            conn.execute(f"CREATE TABLE retained_{table} AS SELECT * FROM {table}")
            query = f"SELECT * FROM retained_{table}"
        conn.execute(f"DROP TABLE {table}")
        conn.execute(f'CREATE VIEW "{table}" AS {query}')
    return outside if external else None


def replace_with_generated_column(path: Path, table: str, column: str) -> None:
    """Retain real storage except one context-dependent owned contract column."""
    import duckdb

    with duckdb.connect(str(path)) as conn:
        if table == "files":
            conn.execute("CREATE TABLE chunks_copy AS SELECT * FROM chunks")
            conn.execute("DROP TABLE chunks")
            conn.execute("ALTER TABLE chunks_copy RENAME TO chunks")
        columns = conn.execute(f"DESCRIBE {table}").fetchall()
        conn.execute(f"CREATE TABLE retained AS SELECT * FROM {table}")
        conn.execute(f"DROP TABLE {table}")
        definitions = []
        stored = []
        for name, kind, *_ in columns:
            if name == column:
                expression = (
                    "CASE WHEN current_database()='memory' "
                    "THEN 'openai' ELSE 'wrong-provider' END"
                    if column == "provider"
                    else f"retained_{column}"
                )
                definitions.append(
                    f'"{name}" {kind} GENERATED ALWAYS AS ({expression})'
                )
            else:
                definitions.append(f'"{name}" {kind}')
                stored.append(f'"{name}"')
        definitions.append(
            f"retained_{column} {next(c[1] for c in columns if c[0] == column)}"
        )
        conn.execute(f"CREATE TABLE {table} ({', '.join(definitions)})")
        conn.execute(
            f"INSERT INTO {table} ({', '.join(stored)}, retained_{column}) "
            f'SELECT {", ".join(stored)}, "{column}" FROM retained'
        )
        conn.execute("DROP TABLE retained")
        if column == "provider":
            assert conn.execute(
                f"SELECT DISTINCT provider FROM {table}"
            ).fetchall() == [("wrong-provider",)]


def collate_identities(path: Path, collation: str) -> tuple[str, str]:
    """One exact selected identity versus 100 collating-but-distinct vectors."""
    import duckdb

    dominant = (
        ("OPENAI", "HUB-TEST") if collation == "nocase" else ("opénai", "húb-test")
    )
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE vectors_copy AS SELECT * FROM embeddings_8")
        conn.execute("DROP TABLE embeddings_8")
        conn.execute(
            "CREATE TABLE embeddings_8 (chunk_id INTEGER, "
            f"provider VARCHAR COLLATE {collation}, model VARCHAR COLLATE {collation}, "
            "embedding FLOAT[8], dims INTEGER, created_at TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO embeddings_8 SELECT chunk_id, provider, model, "
            "embedding, dims, created_at FROM vectors_copy LIMIT 1"
        )
        conn.execute(
            "INSERT INTO embeddings_8 SELECT chunk_id, ?, ?, "
            "embedding, dims, created_at FROM vectors_copy",
            list(dominant),
        )
        conn.execute("DROP TABLE vectors_copy")
    return dominant


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
