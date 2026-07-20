"""Mock deterministic embedding provider for contract tests.

Returns fixed 8-dim vectors derived from a text hash. This makes
Python-vs-Rust pipeline comparisons deterministic and fast.
"""

import hashlib
import struct

MOCK_DIMS = 8
MOCK_PROVIDER = "mock-contract"
MOCK_MODEL = "mock-v1"


def embed_texts(texts: list[str], provider: str = "", model: str = "") -> list[list[float]]:
    """Generate deterministic 8-dim embedding vectors from text hashes.

    Each vector element is a float in [0, 1] derived from the SHA-256 hash
    of ``f"{provider}|{model}|{text}"``. Same input → same output every time.
    """
    vectors = []
    for text in texts:
        hash_input = f"{MOCK_PROVIDER}|{MOCK_MODEL}|{text}"
        digest = hashlib.sha256(hash_input.encode("utf-8")).digest()
        floats = []
        for i in range(0, MOCK_DIMS * 4, 4):
            val = struct.unpack(">I", digest[i : i + 4])[0]
            floats.append(val / 0xFFFFFFFF)
        vectors.append(floats)
    return vectors


class MockEmbeddingProvider:
    """Adapter that presents the mock embed logic as an APIEmbeddingProvider-compatible object."""

    name: str = MOCK_PROVIDER
    model: str = MOCK_MODEL
    dims: int = MOCK_DIMS
    native_dims: int = MOCK_DIMS
    output_dims: int | None = None
    base_url: str = "mock://local"
    api_key: str | None = None
    timeout: int = 30
    retry_attempts: int = 0
    batch_size: int = 1000
    max_tokens: int | None = 8192
    client_side_truncation: bool = False
    distance: str = "cosine"
    supported_dimensions: tuple[int, ...] = (MOCK_DIMS,)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    async def embed_single(self, text: str) -> list[float]:
        return embed_texts([text])[0]

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        return embed_texts(texts)

    def get_max_tokens_per_batch(self) -> int:
        return 8192

    def get_max_documents_per_batch(self) -> int:
        return 2048

    def get_recommended_concurrency(self) -> int:
        return 4

    async def validate_api_key(self) -> bool:
        return True

    def get_rate_limits(self) -> dict:
        return {}

    def get_request_headers(self) -> dict:
        return {}

    def supports_reranking(self) -> bool:
        return False

    @property
    def config(self):  # minimal EmbeddingConfig duck-type
        from collections import namedtuple
        Cfg = namedtuple("EmbeddingConfig", ["provider", "model", "dims"])
        return Cfg(provider=self.name, model=self.model, dims=self.dims)