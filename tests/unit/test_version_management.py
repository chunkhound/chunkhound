#!/usr/bin/env python3
"""Tests for version management utilities."""

import pytest
from scripts.update_version import (
    validate_version,
    parse_version,
    bump_version,
)


class TestValidateVersion:
    """Test PEP 440 version validation."""

    def test_standard_releases(self):
        """Standard semantic versions should pass."""
        validate_version("4.1.0")
        validate_version("1.0.0")
        validate_version("99.99.99")

    def test_canonical_prereleases(self):
        """Canonical pre-release formats should pass."""
        validate_version("4.1.0a1")
        validate_version("4.1.0b2")
        validate_version("4.1.0rc3")

    def test_alternative_spellings(self):
        """Alternative pre-release spellings should pass."""
        validate_version("4.1.0alpha1")
        validate_version("4.1.0beta2")
        validate_version("4.1.0c1")  # Alternative to rc
        validate_version("4.1.0preview1")
        validate_version("4.1.0pre1")

    def test_reject_separators(self):
        """Versions with separators should be rejected."""
        with pytest.raises(ValueError, match="Invalid version format"):
            validate_version("4.1.0-b1")
        with pytest.raises(ValueError, match="Invalid version format"):
            validate_version("4.1.0.rc1")
        with pytest.raises(ValueError, match="Invalid version format"):
            validate_version("4.1.0_a1")

    def test_reject_incomplete_prerelease(self):
        """Pre-release without number should be rejected."""
        with pytest.raises(ValueError, match="Invalid version format"):
            validate_version("4.1.0b")

    def test_reject_malformed(self):
        """Malformed versions should be rejected."""
        with pytest.raises(ValueError, match="Invalid version format"):
            validate_version("v4.1.0")  # Leading v
        with pytest.raises(ValueError, match="Invalid version format"):
            validate_version("4.1")  # Missing patch
        with pytest.raises(ValueError, match="Invalid version format"):
            validate_version("4.1.0.1")  # Four segments
        with pytest.raises(ValueError, match="Invalid version format"):
            validate_version("4.1.0-SNAPSHOT")  # Maven-style
        with pytest.raises(ValueError, match="Invalid version format"):
            validate_version("4.1.0dev1")  # Dev release (no separator)
        with pytest.raises(ValueError, match="Invalid version format"):
            validate_version("4.1.0.dev1")  # Dev release (with separator)
        with pytest.raises(ValueError, match="Invalid version format"):
            validate_version("4.1.0+local")  # Local version identifier


class TestParseVersion:
    """Test version string parsing."""

    def test_parse_standard_release(self):
        """Parse standard semantic version."""
        assert parse_version("4.1.0") == (4, 1, 0, None)
        assert parse_version("1.2.3") == (1, 2, 3, None)

    def test_parse_prerelease(self):
        """Parse pre-release versions."""
        assert parse_version("4.1.0b2") == (4, 1, 0, "b2")
        assert parse_version("4.1.0rc3") == (4, 1, 0, "rc3")
        assert parse_version("4.1.0alpha1") == (4, 1, 0, "alpha1")

    def test_parse_invalid(self):
        """Invalid format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("invalid")
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_version("4.1")


class TestBumpVersion:
    """Test semantic version bumping."""

    def test_bump_major(self):
        """Bump major version resets minor and patch."""
        assert bump_version("4.1.2", "major") == "5.0.0"
        assert bump_version("1.0.0", "major") == "2.0.0"

    def test_bump_minor(self):
        """Bump minor version resets patch."""
        assert bump_version("4.1.2", "minor") == "4.2.0"
        assert bump_version("1.0.0", "minor") == "1.1.0"

    def test_bump_patch(self):
        """Bump patch version."""
        assert bump_version("4.1.2", "patch") == "4.1.3"
        assert bump_version("1.0.0", "patch") == "1.0.1"

    def test_bump_with_prerelease(self):
        """Bump with pre-release suffix."""
        assert bump_version("4.0.1", "minor", "b1") == "4.1.0b1"
        assert bump_version("4.0.1", "major", "rc1") == "5.0.0rc1"
        assert bump_version("4.0.1", "patch", "a1") == "4.0.2a1"

    def test_bump_removes_existing_prerelease(self):
        """Bumping a pre-release removes the suffix unless new one provided."""
        assert bump_version("4.1.0b1", "patch") == "4.1.1"
        assert bump_version("4.1.0rc2", "minor") == "4.2.0"
        assert bump_version("4.1.0b1", "patch", "b2") == "4.1.1b2"

    def test_bump_invalid_type(self):
        """Invalid bump type should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid bump type"):
            bump_version("4.1.0", "invalid")
