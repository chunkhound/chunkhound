"""
Test fixtures for MCP server functionality.
"""

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config


class MCPServerTestFixture:
    """Test fixture for MCP server."""

    def __init__(self, project_dir: Path, config: Config):
        self.project_dir = project_dir
        self.config = config
        self.process: subprocess.Popen | None = None
        self.db_path = Path(config.database.path)

    async def start_mcp_server(self, transport: str = "stdio") -> bool:
        """Start MCP server in background."""
        cmd = ["uv", "run", "chunkhound", "mcp", transport]

        env = os.environ.copy()
        env.update(
            {
                "CHUNKHOUND_DATABASE__PATH": str(self.db_path),
                "CHUNKHOUND_MCP_MODE": "1",
                "CHUNKHOUND_DEBUG": "1",
            }
        )

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.project_dir,
            env=env,
        )

        # Wait for startup
        await asyncio.sleep(3)
        return self.process.poll() is None

    async def stop_mcp_server(self):
        """Stop MCP server gracefully."""
        if self.process:
            try:
                # First try graceful termination
                self.process.terminate()
                try:
                    await asyncio.wait_for(self._wait_for_process(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Force kill if graceful termination fails
                    self.process.kill()
                    await asyncio.wait_for(self._wait_for_process(), timeout=2.0)
            except Exception:
                # Ensure process is killed even if other steps fail
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                # Ensure we wait for final cleanup
                if self.process and self.process.poll() is None:
                    try:
                        self.process.wait()
                    except Exception:
                        pass
                self.process = None

    async def _wait_for_process(self):
        """Wait for process to exit using proper asyncio subprocess waiting."""
        if not self.process:
            return

        # Use a more efficient waiting approach
        loop = asyncio.get_event_loop()

        def check_process():
            return self.process.poll() is not None

        while not check_process():
            await asyncio.sleep(0.05)  # Reduced polling interval


class FileOperationSimulator:
    """Simulate controlled file operations for testing."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.created_files: list[Path] = []

    def create_file(self, name: str, content: str) -> Path:
        """Create a test file."""
        file_path = self.project_dir / name
        file_path.write_text(content)
        self.created_files.append(file_path)
        return file_path

    def modify_file(self, file_path: Path, content: str):
        """Modify an existing file."""
        file_path.write_text(content)

    def delete_file(self, file_path: Path):
        """Delete a file."""
        if file_path.exists():
            file_path.unlink()
            if file_path in self.created_files:
                self.created_files.remove(file_path)

    def move_file(self, src: Path, dst: Path):
        """Move/rename a file."""
        src.rename(dst)
        if src in self.created_files:
            self.created_files.remove(src)
            self.created_files.append(dst)

    def cleanup(self):
        """Clean up all created files."""
        for file_path in self.created_files[:]:
            if file_path.exists():
                file_path.unlink()
        self.created_files.clear()


@pytest.fixture
def file_operations(temp_project_with_monitoring):
    """File operation simulator for testing."""
    fixture = temp_project_with_monitoring
    simulator = FileOperationSimulator(fixture.project_dir)

    yield simulator

    # Cleanup
    simulator.cleanup()
