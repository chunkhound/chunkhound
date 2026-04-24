# Reranking Test Setup (For Developers Only)

This document describes how to set up a local reranking service for **testing** multi-hop semantic search functionality. 

**Note:** Production users should use providers with built-in reranking support like VoyageAI. This setup is only needed for testing and development.

## Quick Start: Automatic Mock Server

The easiest way to run tests with reranking is using the automatic mock server:

```bash
# Run all multi-hop semantic search tests with automatic mock server
python tests/run_with_rerank_server.py

# Run specific tests with automatic mock server
python tests/run_with_rerank_server.py tests/test_embeddings.py -v

# Run any pytest command with automatic mock server
python tests/run_with_rerank_server.py -k rerank -v
```

The script will:
1. Check if a reranking server is already running
2. If not, start a lightweight mock server
3. Run your tests
4. Automatically clean up the server when done

This mock server provides Cohere-compatible `/rerank` API for testing without heavy dependencies.

## Prerequisites

- Python 3.10+
- Ollama installed and running
- GPU recommended for vLLM (but CPU mode available)

## Option 1: vLLM Reranking Server (Recommended for Testing)

vLLM provides a Cohere-compatible `/rerank` API endpoint that ChunkHound expects.

### Installation

```bash
# Install vLLM separately (not part of ChunkHound dependencies)
pip install vllm

# For CPU-only systems
pip install vllm --extra-index-url https://download.pytorch.org/whl/cpu
```

### Starting the Reranking Server

```bash
# Start vLLM with a reranking model on port 8000
vllm serve Qwen/Qwen3-Reranker-0.6B --port 8000 --dtype auto

# Or with a different model
vllm serve Qwen/Qwen3-Reranker-0.6B --port 8000 --dtype auto
```

The server will provide:
- Health endpoint: `http://localhost:8000/health`
- Rerank endpoint: `http://localhost:8000/rerank` (Cohere-compatible)

### Testing the Reranking Server

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test reranking
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "What is the capital of France?",
    "documents": [
      "The capital of Brazil is Brasilia.",
      "The capital of France is Paris.",
      "Horses and cows are both animals"
    ]
  }'
```

## Option 2: Ollama Native Reranking (Simplest)

Ollama now supports reranking natively. Pull a reranker model and point ChunkHound at it:

```bash
ollama pull qwen3-embedding
ollama pull qwen3-reranker
```

Configure ChunkHound to use Ollama for both embeddings and reranking:

```bash
export CHUNKHOUND_EMBEDDING__PROVIDER=openai
export CHUNKHOUND_EMBEDDING__BASE_URL=http://localhost:11434/v1
export CHUNKHOUND_EMBEDDING__MODEL=qwen3-embedding
export CHUNKHOUND_EMBEDDING__API_KEY=dummy-key
export CHUNKHOUND_EMBEDDING__RERANK_MODEL=qwen3-reranker
export CHUNKHOUND_EMBEDDING__RERANK_FORMAT=cohere
```

This avoids the need for a separate vLLM server entirely.

## Option 3: Using Ollama for Embeddings + vLLM for Reranking

This is the typical test configuration:

### 1. Start Ollama (for embeddings)

```bash
# Pull an embedding model
ollama pull qwen3-embedding

# Ollama should already be running on port 11434
ollama serve  # If not already running
```

### 2. Start vLLM (for reranking)

```bash
# As described above
vllm serve Qwen/Qwen3-Reranker-0.6B --port 8000
```

### 3. Configure ChunkHound

Set environment variables:

```bash
# Configure embeddings (Ollama)
export CHUNKHOUND_EMBEDDING__PROVIDER=openai
export CHUNKHOUND_EMBEDDING__BASE_URL=http://localhost:11434/v1
export CHUNKHOUND_EMBEDDING__MODEL=qwen3-embedding
export CHUNKHOUND_EMBEDDING__API_KEY=dummy-key

# Configure reranking (vLLM)
export CHUNKHOUND_EMBEDDING__RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B
export CHUNKHOUND_EMBEDDING__RERANK_URL=http://localhost:8000/rerank
```

## Running Tests

Once both services are running:

```bash
# Run two-hop semantic search tests
uv run pytest tests/test_multi_hop_semantic_search.py -v

# Run specific provider tests
uv run pytest tests/test_embeddings.py::test_ollama_with_reranking_configuration -v
```

## Production Usage

For production use, we recommend:

1. **VoyageAI** - Has built-in reranking support, no additional setup needed
2. **OpenAI + Cohere** - Use OpenAI for embeddings and Cohere's cloud reranking API
3. **Custom Solutions** - Deploy your own reranking service with a Cohere-compatible API

## Troubleshooting

### Port Already in Use

If port 8000 is taken, use a different port:

```bash
vllm serve Qwen/Qwen3-Reranker-0.6B --port 8001
export CHUNKHOUND_EMBEDDING__RERANK_URL=http://localhost:8001/rerank
```

### Out of Memory

For systems with limited RAM/VRAM:

```bash
# Use smaller model
vllm serve Qwen/Qwen3-Reranker-0.6B --port 8000 --max-model-len 512

# Or use CPU mode (slower)
vllm serve Qwen/Qwen3-Reranker-0.6B --port 8000 --device cpu
```

### Services Not Found

Verify services are running:

```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check vLLM
curl http://localhost:8000/health
```

## API Format

ChunkHound expects the reranking service to implement this API:

### Request
```json
POST /rerank
{
  "model": "model-name",
  "query": "search query",
  "documents": ["doc1", "doc2", ...],
  "top_n": 10  // optional
}
```

### Response
```json
{
  "results": [
    {"index": 1, "relevance_score": 0.95},
    {"index": 0, "relevance_score": 0.82},
    ...
  ]
}
```

This format is compatible with Cohere's Rerank API v1/v2.