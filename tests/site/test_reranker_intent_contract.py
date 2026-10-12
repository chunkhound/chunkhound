"""The configurator and backend agree on complete reranker intent."""

import json

import pytest
from pydantic import ValidationError

from chunkhound.core.config.config import Config
from tests.site.tsx_runner import run_tsx_json

# Provider, URL, explicit format/model, accepted by both layers.
CASES = [
    ("openai-embed", "/unused", "", "", True),
    ("ollama-embed", "ftp://unused", "", "", True),
    ("openai-embed", "HTTPS://example.com/rerank", "tei", "", True),
    ("ollama-embed", "http:/broken", "tei", "", False),
    ("ollama-embed", "http:///broken", "tei", "", False),
    ("ollama-embed", "/rerank", "cohere", "", False),
    ("ollama-embed", "/rerank", "cohere", "model", True),
    ("openai-embed", "", "tei", "", False),
    ("voyageai", "", "cohere", "", True),
    ("voyageai", "", "", "model", True),
]


def test_complete_reranker_intent_matches_backend(tmp_path, clean_environment):
    verdicts = run_tsx_json(f"""
import {{ findEmbeddingProvider, findLlmProvider, buildChunkhoundConfig }}
  from './site/src/components/configurator/index.ts';
import {{ rerankerValidationError }}
  from './site/src/components/configurator/rerank-validate.ts';
console.log(JSON.stringify({json.dumps(CASES)}.map(([id, url, format, model]) => {{
  const retrieval = findEmbeddingProvider(id);
  return {{
    error: rerankerValidationError(retrieval, url, format, model) ?? null,
    config: buildChunkhoundConfig(retrieval, findLlmProvider('ollama-llm'),
      {{ url, format, model }}),
  }};
}})));
""")
    for case, verdict in zip(CASES, verdicts, strict=True):
        accepted = case[-1]
        assert (verdict["error"] is None) is accepted, case
        if accepted:
            Config(target_dir=tmp_path, **verdict["config"])
        else:
            with pytest.raises(ValidationError):
                Config(target_dir=tmp_path, **verdict["config"])
