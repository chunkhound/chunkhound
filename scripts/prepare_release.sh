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

# Clean previous local build scratch without deleting staged release wheels
echo "🧹 Cleaning previous build scratch..."
rm -rf build/
mkdir -p dist

# Run smoke tests (mandatory before release)
echo "🧪 Running smoke tests..."
if ! uv run pytest tests/test_smoke.py -v -n auto; then
    echo "❌ Smoke tests failed! Fix critical issues before releasing."
    exit 1
fi
echo "✅ Smoke tests passing"

# Regenerate locked requirements for reproducible installs
echo "🔒 Regenerating locked requirements..."
uv pip compile pyproject.toml --all-extras -o requirements-lock.txt
echo "✅ Updated requirements-lock.txt with exact versions"

# Build source distribution without deleting or replacing staged platform wheels
echo "📦 Building source distribution..."
uv build --sdist --out-dir dist
echo "✅ Built source distribution"

# Verify staged platform wheels exist before release verification
echo "📦 Checking staged platform wheel artifacts..."
shopt -s nullglob
WHEEL_PATHS=(dist/*.whl)
if [[ ! -e "${WHEEL_PATHS[0]}" ]]; then
    echo "❌ No wheel found in dist/."
    echo "   Stage the supported Linux and Windows wheels in dist/ before running prepare_release.sh."
    exit 1
fi
echo "✅ Found staged wheel artifacts:"
printf '   %s\n' "${WHEEL_PATHS[@]}"

# Verify AutoDoc packaged resources exist in the staged wheel(s)
echo "🔎 Verifying AutoDoc wheel resources..."
uv run python scripts/verify_autodoc_wheel_resources.py "${WHEEL_PATHS[@]}"
echo "✅ AutoDoc wheel resources verified"

# Verify Watchman packaged runtime resources across the full staged wheel matrix
echo "🔎 Verifying Watchman runtime wheel resources..."
uv run python scripts/verify_watchman_runtime_resources.py --require-supported-matrix "${WHEEL_PATHS[@]}"
echo "✅ Watchman runtime wheel resources verified"

# Verify host-compatible staged wheels satisfy the managed Watchman live-indexing contract
echo "🔎 Verifying Watchman installed-wheel live indexing..."
uv run python scripts/verify_watchman_live_indexing_e2e.py "${WHEEL_PATHS[@]}"
echo "✅ Watchman installed-wheel live indexing verified"

# Verify the documented source/editable fallback contract from the current checkout
echo "🔎 Verifying Watchman source/editable fallback behavior..."
uv run python scripts/verify_watchman_live_indexing_e2e.py --verify-source-fallback --source-root "$PROJECT_ROOT"
echo "✅ Watchman source/editable fallback behavior verified"

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
