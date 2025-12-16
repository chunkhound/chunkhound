"""Test to reproduce the oversized chunk error (325k+ tokens) in a controlled environment.

This test reproduces the exact error:
"Failed to generate embeddings: Error code: 400 - {'error': {'message': 'Requested 325998 tokens, max 300000 tokens per request'"
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.embedding_service import EmbeddingService
from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider
from .test_utils import get_api_key_for_tests


class TestOversizedChunkReproduction:
    """Reproduce the 325k+ token error in controlled environment."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create isolated temporary directory for test."""
        temp_dir = tempfile.mkdtemp(prefix="chunkhound_repro_test_")
        yield Path(temp_dir)
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration with isolated database."""
        db_path = temp_dir / "test.db"
        # Use fake args to prevent find_project_root call that fails in CI
        from types import SimpleNamespace
        fake_args = SimpleNamespace(path=temp_dir)
        return Config(
            args=fake_args,
            database={
                "provider": "duckdb",
                "path": str(db_path)
            },
            embedding={
                "provider": "openai",
                "api_key": os.getenv("OPENAI_API_KEY", "test-key"),
                "model": "text-embedding-3-small"
            },
            indexing={
                "max_chunk_size": 2000,  # Standard size limit
                "min_chunk_size": 50
            }
        )

    @pytest.fixture
    def services(self, test_config, temp_dir):
        """Create isolated services for testing."""
        return create_services(
            db_path=temp_dir / "test.db",
            config=test_config
        )

    def create_oversized_content(self, target_chars: int = 325998) -> str:
        """Create content with exact character count to reproduce the error."""
        # Create realistic code-like content that could cause chunking issues
        base_content = '''
def very_long_function_that_might_not_be_chunked_properly():
    """This is a function that demonstrates potential chunking issues.
    
    The function contains a lot of repetitive code that might cause
    the chunking system to fail to break it into appropriate sizes.
    This could happen with generated code, minified code, or code
    with very long lines that don't have natural break points.
    """
    # Simulate a very long data structure or configuration
    data = {
        '''
        
        # Add repetitive content to reach target size
        data_items = []
        item_template = '"key_{i}": "value_{i}_' + "x" * 100 + '",'
        
        current_size = len(base_content) + len("\n    }\n    return data")
        
        i = 0
        while current_size < target_chars:
            item = item_template.format(i=i)
            if current_size + len(item) > target_chars:
                # Adjust last item to hit exact target
                remaining = target_chars - current_size - 20  # Leave space for closing
                item = f'"final_key": "{"x" * remaining}"'
                data_items.append(item)
                break
            data_items.append(item)
            current_size += len(item) + 1  # +1 for newline
            i += 1
        
        full_content = base_content + "\n        ".join(data_items) + "\n    }\n    return data"
        
        # Ensure exact character count
        if len(full_content) < target_chars:
            padding = target_chars - len(full_content)
            full_content += "\n# " + "Padding comment " * (padding // 16)
            full_content = full_content[:target_chars]
        elif len(full_content) > target_chars:
            full_content = full_content[:target_chars]
            
        assert len(full_content) == target_chars, f"Expected {target_chars} chars, got {len(full_content)}"
        return full_content

    def create_test_file(self, temp_dir: Path, filename: str, content: str) -> Path:
        """Create a test file with specific content."""
        test_file = temp_dir / filename
        test_file.write_text(content)
        return test_file


    @pytest.mark.asyncio  
    async def test_reproduce_with_mock_provider_for_debugging(self, services, temp_dir, test_config):
        """Reproduce the chunking behavior without hitting OpenAI API for debugging."""
        # Create oversized content
        oversized_content = self.create_oversized_content(325998)
        test_file = self.create_test_file(temp_dir, "oversized.py", oversized_content)
        
        print(f"\n🔍 DEBUGGING MODE - Analyzing chunking behavior")
        print(f"Test file: {test_file}")
        print(f"Content size: {len(oversized_content):,} characters")
        
        # Create mock embedding provider to avoid API calls
        mock_provider = Mock(spec=OpenAIEmbeddingProvider)
        mock_provider.name = "mock-openai"
        mock_provider.model = "text-embedding-3-small"
        mock_provider.dims = 1536
        
        # Mock the embed method to simulate the token limit error
        async def mock_embed(texts):
            total_chars = sum(len(text) for text in texts)
            estimated_tokens = int(total_chars / 4)  # Rough estimate: 4 chars per token
            
            print(f"Mock provider called with:")
            print(f"  Texts: {len(texts)}")
            print(f"  Total chars: {total_chars:,}")
            print(f"  Estimated tokens: {estimated_tokens:,}")
            
            # Find oversized chunks
            for i, text in enumerate(texts):
                if len(text) > 200000:
                    print(f"  ⚠️  Chunk {i}: {len(text):,} characters")
            
            if estimated_tokens > 300000:
                from openai import BadRequestError
                error_msg = f"Requested {estimated_tokens} tokens, max 300000 tokens per request"
                raise BadRequestError(
                    message=error_msg,
                    response=None,
                    body={"error": {"message": error_msg, "type": "max_tokens_per_request"}}
                )
            
            # Return mock embeddings
            return [[0.1] * 1536 for _ in texts]
        
        mock_provider.embed = AsyncMock(side_effect=mock_embed)
        
        # Create coordinator using registry (like the CLI does)
        from chunkhound.registry import configure_registry, create_indexing_coordinator
        configure_registry(test_config)
        coordinator = create_indexing_coordinator()
        
        # Replace the embedding provider with our mock
        coordinator._embedding_provider = mock_provider
        
        # Index the file to see how it gets chunked
        print("\n📊 Starting indexing process...")
        result = await coordinator.process_file(temp_dir / "oversized.py")
        print(f"Indexing result: {result}")
        
        # Analyze the chunks that were created using alternative approach
        # Get chunks by querying files and then getting chunks for each file
        from chunkhound.core.models.chunk import Chunk
        chunks = []
        files = services.provider.execute_query("SELECT id FROM files")
        for file_record in files:
            file_chunks = services.provider.get_chunks_by_file_id(file_record["id"])
            chunks.extend([Chunk.from_dict(data) for data in file_chunks])
        print(f"\n📈 Chunk Analysis:")
        print(f"Total chunks created: {len(chunks)}")
        
        if chunks:
            chunk_sizes = [len(chunk.code) for chunk in chunks]
            chunk_sizes.sort(reverse=True)
            
            print(f"Chunk size distribution:")
            print(f"  Largest: {chunk_sizes[0]:,} chars")
            print(f"  Median: {chunk_sizes[len(chunk_sizes)//2]:,} chars")
            print(f"  Smallest: {chunk_sizes[-1]:,} chars")
            
            # Show chunks by size category
            categories = [
                ("Oversized (200k+)", lambda x: x > 200000),
                ("Large (100k-200k)", lambda x: 100000 <= x < 200000),
                ("Big (50k-100k)", lambda x: 50000 <= x < 100000),
                ("Medium (10k-50k)", lambda x: 10000 <= x < 50000),
                ("Small (0-10k)", lambda x: x < 10000)
            ]
            
            for category_name, condition in categories:
                count = sum(1 for size in chunk_sizes if condition(size))
                if count > 0:
                    print(f"  {category_name}: {count} chunks")
        
        # Try to generate embeddings with our mock provider
        print(f"\n🚀 Attempting embedding generation...")
        try:
            embed_result = await coordinator.generate_missing_embeddings()
            print(f"Embedding generation completed: {embed_result}")
        except Exception as e:
            print(f"Mock provider error (as expected): {e}")
            
            # This confirms our chunking system is creating oversized chunks
            if "300000 tokens per request" in str(e):
                print("✅ Confirmed: Chunking system creates oversized chunks!")
            
    @pytest.mark.asyncio
    async def test_analyze_chunking_behavior_with_large_files(self, services, temp_dir, test_config):
        """Analyze how the chunking system handles various large file patterns."""
        print(f"\n🔬 CHUNKING BEHAVIOR ANALYSIS")
        
        test_cases = [
            ("Small normal file", 5000, "Normal code structure"),
            ("Medium file", 50000, "Larger code file"),
            ("Large single function", 150000, "One massive function"),
            ("Oversized data file", 325998, "Huge data structure"),
        ]
        
        for name, size, description in test_cases:
            print(f"\n--- Testing: {name} ({size:,} chars) ---")
            
            # Create test content of specific size
            content = self.create_oversized_content(size)
            test_file = self.create_test_file(temp_dir, f"test_{size}.py", content)
            
            # Clear previous chunks
            services.provider.execute_query("DELETE FROM chunks")
            
            # Create simple mock provider for this test
            mock_provider = Mock(spec=OpenAIEmbeddingProvider)
            mock_provider.name = "analysis-mock"
            mock_provider.model = "text-embedding-3-small"
            mock_provider.dims = 1536
            mock_provider.embed = AsyncMock(return_value=[[0.1] * 1536])
            
            # Index and analyze
            coordinator = IndexingCoordinator(
                database_provider=services.provider,
                base_directory=temp_dir,
                embedding_provider=mock_provider
            )
            
            await coordinator.process_file(test_file)
            # Get chunks using alternative approach
            from chunkhound.core.models.chunk import Chunk
            file_record = services.provider.get_file_by_path(str(test_file))
            if file_record:
                chunk_data = services.provider.get_chunks_by_file_id(file_record["id"])
                chunks = [Chunk.from_dict(data) for data in chunk_data]
            else:
                chunks = []
            
            chunk_count = len(chunks)
            if chunk_count > 0:
                chunk_sizes = [len(chunk.code) for chunk in chunks]
                max_size = max(chunk_sizes)
                avg_size = sum(chunk_sizes) // chunk_count
                
                print(f"  Chunks created: {chunk_count}")
                print(f"  Largest chunk: {max_size:,} chars")
                print(f"  Average chunk: {avg_size:,} chars")
                
                # Check if any chunks are problematic
                oversized = [size for size in chunk_sizes if size > 100000]
                if oversized:
                    print(f"  ⚠️  Oversized chunks: {len(oversized)} (sizes: {oversized})")
                else:
                    print(f"  ✅ All chunks under 100k chars")
            else:
                print(f"  ❌ No chunks created!")

if __name__ == "__main__":
    # Allow running the test directly for quick debugging
    import asyncio
    
    async def quick_test():
        test = TestOversizedChunkReproduction()
        temp_dir = Path(tempfile.mkdtemp(prefix="chunkhound_quick_test_"))
        
        try:
            content = test.create_oversized_content(325998)
            print(f"Created content: {len(content):,} characters")
            print(f"Preview: {content[:200]}...")
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    asyncio.run(quick_test())