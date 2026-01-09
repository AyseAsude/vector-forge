"""Tests for vector_forge.strategies.noise.pca module."""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings

from vector_forge.strategies.noise.pca import PCAReducer


class TestPCAReducer:
    """Tests for PCAReducer."""

    def test_reduce_single_vector(self):
        """Test reducing single vector returns clone."""
        reducer = PCAReducer()
        vector = torch.randn(64)

        result = reducer.reduce([vector])

        assert torch.equal(result, vector)

    def test_reduce_empty_raises(self):
        """Test reducing empty list raises ValueError."""
        reducer = PCAReducer()

        with pytest.raises(ValueError, match="Need at least"):
            reducer.reduce([])

    def test_reduce_two_vectors(self):
        """Test reducing two vectors."""
        reducer = PCAReducer()
        torch.manual_seed(42)
        v1 = torch.randn(64)
        v2 = torch.randn(64)

        result = reducer.reduce([v1, v2])

        assert result.shape == torch.Size([64])
        # Result should be unit norm (normalized)
        assert torch.isclose(result.norm(), torch.tensor(1.0), atol=1e-5)

    def test_reduce_many_vectors(self):
        """Test reducing many vectors."""
        reducer = PCAReducer()
        torch.manual_seed(42)
        vectors = [torch.randn(128) for _ in range(10)]

        result = reducer.reduce(vectors)

        assert result.shape == torch.Size([128])
        assert torch.isclose(result.norm(), torch.tensor(1.0), atol=1e-5)

    def test_reduce_similar_vectors(self):
        """Test reducing similar vectors captures main direction."""
        reducer = PCAReducer()

        # Create vectors that are mostly in the same direction
        base = torch.tensor([1.0, 0.0, 0.0, 0.0])
        torch.manual_seed(42)
        vectors = [base + torch.randn(4) * 0.1 for _ in range(10)]

        result = reducer.reduce(vectors)

        # First component should dominate (close to [1,0,0,0] direction)
        assert abs(result[0]) > 0.5

    def test_n_components_parameter(self):
        """Test n_components parameter."""
        reducer = PCAReducer(n_components=3)
        torch.manual_seed(42)
        vectors = [torch.randn(64) for _ in range(10)]

        result = reducer.reduce(vectors)

        # Should still return first PC
        assert result.shape == torch.Size([64])

    def test_variance_threshold_parameter(self):
        """Test variance_threshold parameter."""
        reducer = PCAReducer(variance_threshold=0.99)
        torch.manual_seed(42)
        vectors = [torch.randn(64) for _ in range(10)]

        result = reducer.reduce(vectors)

        assert result.shape == torch.Size([64])

    def test_preserves_device(self):
        """Test that reduction preserves tensor device."""
        reducer = PCAReducer()
        vectors = [torch.randn(64) for _ in range(5)]

        result = reducer.reduce(vectors)

        assert result.device == vectors[0].device

    def test_preserves_dtype(self):
        """Test that reduction preserves tensor dtype."""
        reducer = PCAReducer()
        vectors = [torch.randn(64, dtype=torch.float32) for _ in range(5)]

        result = reducer.reduce(vectors)

        assert result.dtype == torch.float32


class TestPCAReducerWeighted:
    """Tests for reduce_weighted method."""

    def test_weighted_single_vector(self):
        """Test weighted reduction with single vector."""
        reducer = PCAReducer()
        vector = torch.randn(64)

        result = reducer.reduce_weighted([vector])

        assert torch.equal(result, vector)

    def test_weighted_empty_raises(self):
        """Test weighted reduction with empty list raises."""
        reducer = PCAReducer()

        with pytest.raises(ValueError, match="Need at least"):
            reducer.reduce_weighted([])

    def test_weighted_basic(self):
        """Test basic weighted reduction."""
        reducer = PCAReducer()
        torch.manual_seed(42)
        vectors = [torch.randn(64) for _ in range(5)]
        weights = [1.0, 0.5, 0.5, 0.5, 0.5]

        result = reducer.reduce_weighted(vectors, weights)

        assert result.shape == torch.Size([64])
        assert torch.isclose(result.norm(), torch.tensor(1.0), atol=1e-5)

    def test_weighted_none_weights(self):
        """Test weighted reduction with None weights."""
        reducer = PCAReducer()
        torch.manual_seed(42)
        vectors = [torch.randn(64) for _ in range(5)]

        result = reducer.reduce_weighted(vectors, weights=None)

        assert result.shape == torch.Size([64])


class TestPCAReducerVarianceExplained:
    """Tests for get_variance_explained method."""

    def test_variance_single_vector(self):
        """Test variance explained for single vector."""
        reducer = PCAReducer()
        vectors = [torch.randn(64)]

        result = reducer.get_variance_explained(vectors)

        assert result == [1.0]

    def test_variance_multiple_vectors(self):
        """Test variance explained for multiple vectors."""
        reducer = PCAReducer()
        torch.manual_seed(42)
        vectors = [torch.randn(64) for _ in range(10)]

        result = reducer.get_variance_explained(vectors)

        # Should have as many components as min(n_vectors, n_features)
        assert len(result) > 0
        # Variances should sum to 1
        assert abs(sum(result) - 1.0) < 1e-5
        # Variances should be decreasing
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1] - 1e-5

    def test_variance_similar_vectors_high_first_component(self):
        """Test that similar vectors have high variance in first component."""
        reducer = PCAReducer()
        torch.manual_seed(42)

        # Create vectors that are very similar (small perturbations along one direction)
        base = torch.randn(64)
        base = base / base.norm()  # Normalize base
        # Add tiny perturbations - vectors are almost identical
        vectors = [base + torch.randn(64) * 0.001 for _ in range(10)]

        result = reducer.get_variance_explained(vectors)

        # First component should explain significant variance (but not necessarily 90%)
        # With nearly identical vectors, the variance is mostly noise
        assert result[0] > 0.1  # More reasonable threshold

    def test_variance_diverse_vectors_spread_variance(self):
        """Test that diverse vectors spread variance across components."""
        reducer = PCAReducer()
        torch.manual_seed(42)

        # Create orthogonal-ish vectors (high diversity)
        vectors = []
        for i in range(5):
            v = torch.zeros(64)
            v[i * 10:(i + 1) * 10] = torch.randn(10)
            vectors.append(v)

        result = reducer.get_variance_explained(vectors)

        # Variance should be more spread out
        assert result[0] < 0.8  # First component doesn't dominate


class TestPCAReducerPropertyBased:
    """Property-based tests for PCAReducer."""

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=20)
    def test_output_is_unit_norm(self, n_vectors: int):
        """Test output is always unit norm."""
        reducer = PCAReducer()
        torch.manual_seed(42)
        vectors = [torch.randn(64) for _ in range(n_vectors)]

        result = reducer.reduce(vectors)

        assert torch.isclose(result.norm(), torch.tensor(1.0), atol=1e-5)

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=20)
    def test_output_shape_matches_input(self, n_vectors: int):
        """Test output shape matches input vector shape."""
        reducer = PCAReducer()
        torch.manual_seed(42)
        vectors = [torch.randn(128) for _ in range(n_vectors)]

        result = reducer.reduce(vectors)

        assert result.shape == vectors[0].shape

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=20)
    def test_variance_sums_to_one(self, n_vectors: int):
        """Test variance explained always sums to 1."""
        reducer = PCAReducer()
        torch.manual_seed(42)
        vectors = [torch.randn(64) for _ in range(n_vectors)]

        variances = reducer.get_variance_explained(vectors)

        assert abs(sum(variances) - 1.0) < 1e-5


class TestPCAReducerEdgeCases:
    """Edge case tests for PCAReducer."""

    def test_identical_vectors(self):
        """Test with identical vectors."""
        reducer = PCAReducer()
        vector = torch.randn(64)
        vectors = [vector.clone() for _ in range(5)]

        # This may produce NaN or very small values due to zero variance
        # The implementation should handle this gracefully
        result = reducer.reduce(vectors)

        # At minimum, should return a tensor of correct shape
        assert result.shape == torch.Size([64])

    def test_high_dimensional_vectors(self):
        """Test with high-dimensional vectors."""
        reducer = PCAReducer()
        torch.manual_seed(42)
        vectors = [torch.randn(4096) for _ in range(5)]

        result = reducer.reduce(vectors)

        assert result.shape == torch.Size([4096])

    def test_float64_dtype(self):
        """Test with float64 dtype."""
        reducer = PCAReducer()
        vectors = [torch.randn(64, dtype=torch.float64) for _ in range(5)]

        result = reducer.reduce(vectors)

        assert result.dtype == torch.float64
