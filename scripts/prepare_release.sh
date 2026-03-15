#!/bin/bash
set -e

# ChunkHound Release Preparation Script
# Modern uv-based release process with dependency locking

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Preparing ChunkHound Release..."
echo "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Check if we're in a clean git state
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Warning: You have uncommitted changes. Consider committing them first."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Release preparation cancelled."
        exit 1
    fi
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/

# Run smoke tests (mandatory before release)
echo "🧪 Running smoke tests..."
if ! uv run python -m pytest tests/test_smoke.py -v; then
    echo "❌ Smoke tests failed! Fix critical issues before releasing."
    exit 1
fi
echo "✅ Smoke tests passing"

# Regenerate locked requirements for reproducible installs
echo "🔒 Regenerating locked requirements..."
uv pip compile pyproject.toml --all-extras -o requirements-lock.txt
echo "✅ Updated requirements-lock.txt with exact versions"

# Build Python distributions
echo "📦 Building distributions..."
uv build
echo "✅ Built wheel and source distribution"

# Verify AutoDoc packaged resources exist in the built wheel(s)
echo "🔎 Verifying AutoDoc wheel resources..."
WHEEL_PATHS=(dist/*.whl)
if [[ ! -e "${WHEEL_PATHS[0]}" ]]; then
    echo "❌ No wheel found in dist/. Did the build fail?"
    exit 1
fi
uv run python scripts/verify_autodoc_wheel_resources.py "${WHEEL_PATHS[@]}"
echo "✅ AutoDoc wheel resources verified"

# Verify Watchman packaged runtime resources exist in the built wheel(s)
echo "🔎 Verifying Watchman runtime wheel resources..."
uv run python scripts/verify_watchman_runtime_resources.py "${WHEEL_PATHS[@]}"
echo "✅ Watchman runtime wheel resources verified"

# Verify installed wheels satisfy the managed Watchman live-indexing contract
echo "🔎 Verifying Watchman installed-wheel live indexing..."
uv run python scripts/verify_watchman_live_indexing_e2e.py "${WHEEL_PATHS[@]}"
echo "✅ Watchman installed-wheel live indexing verified"

# Generate checksums for release artifacts
echo "🔐 Generating checksums..."
cd dist/
find . -name "*.tar.gz" -o -name "*.whl" | xargs sha256sum > SHA256SUMS
cd "$PROJECT_ROOT"

# Display release summary
echo ""
echo "✅ Release preparation complete!"
echo ""
echo "📦 Release artifacts in dist/:"
ls -la dist/
echo ""
echo "🎯 Next steps:"
echo "1. Test the built distributions locally:"
for wheel_path in "${WHEEL_PATHS[@]}"; do
    echo "   pip install ${wheel_path}"
done
echo "2. Publish to PyPI (requires API token):"
echo "   uv publish"
echo "3. Create GitHub release with artifacts from dist/"
echo ""
echo "🔒 Dependency locking:"
echo "  - requirements-lock.txt updated with exact versions"
echo "  - SHA256SUMS generated for verification"
echo "  - Reproducible installs guaranteed"
echo ""
echo "🎉 Ready for release!"
