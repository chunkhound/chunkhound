"""Unit tests for angular_distance utility function."""

import numpy as np
import pytest

from chunkhound.providers.database import usearch_wrapper


class TestAngularDistance:
    """Test angular distance computation for vector similarity."""

    def test_identical_vectors_zero_distance(self) -> None:
        """Identical vectors should have zero angular distance."""
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert usearch_wrapper.angular_distance(v, v) == pytest.approx(0.0)

    def test_orthogonal_vectors_half_pi(self) -> None:
        """Orthogonal vectors should have pi/2 angular distance."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert usearch_wrapper.angular_distance(v1, v2) == pytest.approx(
            np.pi / 2, abs=1e-6
        )

    def test_opposite_vectors_pi(self) -> None:
        """Opposite vectors should have pi angular distance."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        assert usearch_wrapper.angular_distance(v1, v2) == pytest.approx(np.pi, abs=1e-6)

    def test_zero_vector_returns_max_distance(self) -> None:
        """Zero vector should return maximum distance (pi)."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        assert usearch_wrapper.angular_distance(v1, v2) == pytest.approx(np.pi)

    def test_both_zero_vectors_returns_max_distance(self) -> None:
        """Two zero vectors should return maximum distance (pi)."""
        v1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        assert usearch_wrapper.angular_distance(v1, v2) == pytest.approx(np.pi)

    def test_unnormalized_vectors_normalized_before_comparison(self) -> None:
        """Vectors of different magnitudes should give same angle as normalized."""
        v1 = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        v2 = np.array([10.0, 10.0, 0.0], dtype=np.float32)  # Same direction, 10x magnitude
        # Same direction means near-zero distance (float32 precision limits)
        assert usearch_wrapper.angular_distance(v1, v2) == pytest.approx(0.0, abs=1e-3)

    def test_high_dimensional_vectors(self) -> None:
        """Angular distance should work for high-dimensional vectors."""
        np.random.seed(42)
        v1 = np.random.randn(1024).astype(np.float32)
        v2 = np.random.randn(1024).astype(np.float32)

        dist = usearch_wrapper.angular_distance(v1, v2)
        # Distance should be in valid range [0, pi]
        assert 0.0 <= dist <= np.pi

    def test_triangle_inequality(self) -> None:
        """Angular distance should satisfy triangle inequality."""
        np.random.seed(123)
        v1 = np.random.randn(128).astype(np.float32)
        v2 = np.random.randn(128).astype(np.float32)
        v3 = np.random.randn(128).astype(np.float32)

        d12 = usearch_wrapper.angular_distance(v1, v2)
        d23 = usearch_wrapper.angular_distance(v2, v3)
        d13 = usearch_wrapper.angular_distance(v1, v3)

        # Triangle inequality: d13 <= d12 + d23
        assert d13 <= d12 + d23 + 1e-6  # Small epsilon for numerical stability

    def test_symmetry(self) -> None:
        """Angular distance should be symmetric: d(a,b) = d(b,a)."""
        np.random.seed(456)
        v1 = np.random.randn(64).astype(np.float32)
        v2 = np.random.randn(64).astype(np.float32)

        assert usearch_wrapper.angular_distance(v1, v2) == pytest.approx(
            usearch_wrapper.angular_distance(v2, v1), abs=1e-10
        )
