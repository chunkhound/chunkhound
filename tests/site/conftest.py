import subprocess
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session", autouse=True)
def built_site() -> None:
    """Build the Astro site once per test session."""
    # Import NPM from tsx_runner to get the platform-correct npm path
    from tests.site.tsx_runner import NPM
    subprocess.run(
        [NPM, "run", "build", "--prefix", "site"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
