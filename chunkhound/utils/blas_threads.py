"""Run numpy's BLAS single-threaded in ChunkHound unless the user chose otherwise.

numpy's wheels bundle OpenBLAS, which on import starts a thread per core and commits
a work buffer for each. On a 10-core Windows machine `import numpy` alone committed
308 MB, against 18 MB single-threaded. ChunkHound imports numpy in the CLI, the MCP
server, the daemon and every spawned parse worker, so each of those paid it.
ChunkHound's own numpy work is small vector math: embeddings come from the provider
and vector search runs in DuckDB, so the extra threads gain nothing.

Call it from an entry package before anything imports numpy. Spawned workers inherit
the environment, so one call covers the whole process tree. An explicit
OPENBLAS_NUM_THREADS wins.
"""

from __future__ import annotations

import os


def limit_blas_threads() -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
