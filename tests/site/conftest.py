import subprocess
import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session", autouse=True)
def built_site() -> None:
    """Build the Astro site once per test session."""
    if (ROOT / "site" / "dist" / "index.html").exists():
        return  # already built (e.g. downloaded from CI artifact)
    # Import NPM from tsx_runner to get the platform-correct npm path
    from tests.site.tsx_runner import NPM
    result = subprocess.run(
        [NPM, "run", "build", "--prefix", "site"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        result.check_returncode()
