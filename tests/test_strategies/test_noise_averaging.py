"""Tests for vector_forge.strategies.noise.averaging module."""

import pytest
import torch
from hypothesis import given, strategies as st, settings

from vector_forge.strategies.noise.averaging import AveragingReducer


class TestAveragingReducer:
    """Tests for AveragingReducer."""

    def test_reduce_empty_raises(self):
        """Test reducing empty list raises ValueError."""
        reducer = AveragingReducer()

        with pytest.raises(ValueError, match="Cannot average empty"):
            reducer.reduce([])

    def test_reduce_single_vector(self):
        """Test reducing single vector returns clone."""
        reducer = AveragingReducer()
        vector = torch.randn(768)

        result = reducer.reduce([vector])

        assert torch.equal(result, vector)
        # Verify it's a clone, not same reference
        assert result is not vector

    def test_reduce_two_vectors(self):
        """Test reducing two vectors."""
        reducer = AveragingReducer(normalize=False)
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0])

        result = reducer.reduce([v1, v2])

        expected = torch.tensor([0.5, 0.5, 0.0])
        assert torch.allclose(result, expected)

    def test_reduce_with_normalization(self):
        """Test reduction with normalization enabled."""
        reducer = AveragingReducer(normalize=True)
        v1 = torch.tensor([2.0, 0.0])
        v2 = torch.tensor([0.0, 2.0])

        result = reducer.reduce([v1, v2])

        assert torch.isclose(result.norm(), torch.tensor(1.0))

    def test_reduce_without_normalization(self):
        """Test reduction with normalization disabled."""
        reducer = AveragingReducer(normalize=False)
        v1 = torch.tensor([2.0, 0.0])
        v2 = torch.tensor([0.0, 2.0])

        result = reducer.reduce([v1, v2])

        # Average is [1.0, 1.0], norm is sqrt(2)
        assert not torch.isclose(result.norm(), torch.tensor(1.0))

    def test_reduce_mismatched_shapes_raises(self):
        """Test reducing vectors with mismatched shapes raises."""
        reducer = AveragingReducer()
        v1 = torch.randn(768)
        v2 = torch.randn(512)

        with pytest.raises(ValueError, match="doesn't match"):
            reducer.reduce([v1, v2])

    def test_reduce_with_weights(self):
        """Test weighted averaging."""
        reducer = AveragingReducer(normalize=False)
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([0.0, 1.0])
        weights = [0.75, 0.25]

        result = reducer.reduce([v1, v2], weights=weights)

        expected = torch.tensor([0.75, 0.25])
        assert torch.allclose(result, expected)

    def test_reduce_weights_wrong_length_raises(self):
        """Test weights with wrong length raises."""
        reducer = AveragingReducer()
        vectors = [torch.randn(64) for _ in range(3)]
        weights = [0.5, 0.5]  # Only 2 weights for 3 vectors

        with pytest.raises(ValueError, match="Weights length"):
            reducer.reduce(vectors, weights=weights)

    def test_reduce_preserves_device(self):
        """Test that reduction preserves tensor device."""
        reducer = AveragingReducer()
        vectors = [torch.randn(64) for _ in range(3)]

        result = reducer.reduce(vectors)

        assert result.device == vectors[0].device

    def test_reduce_preserves_dtype(self):
        """Test that reduction preserves tensor dtype."""
        reducer = AveragingReducer()
        vectors = [torch.randn(64, dtype=torch.float32) for _ in range(3)]

        result = reducer.reduce(vectors)

        assert result.dtype == torch.float32

    def test_reduce_many_vectors(self):
        """Test reducing many vectors."""
        reducer = AveragingReducer()
        torch.manual_seed(42)
        vectors = [torch.randn(768) for _ in range(10)]

        result = reducer.reduce(vectors)

        assert result.shape == torch.Size([768])

    def test_normalize_handles_zero_norm(self):
        """Test normalization handles zero vector gracefully."""
        reducer = AveragingReducer(normalize=True)
        # Two opposite vectors that cancel out
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([-1.0, 0.0])

        result = reducer.reduce([v1, v2])

        # Zero vector, normalization should handle this
        assert result.shape == torch.Size([2])


class TestAveragingReducerWithQuality:
    """Tests for reduce_with_quality method."""

    def test_quality_weighted_averaging(self):
        """Test quality-weighted averaging."""
        reducer = AveragingReducer(normalize=False)
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([0.0, 1.0])
        quality_scores = [0.9, 0.1]  # v1 has much higher quality

        result = reducer.reduce_with_quality([v1, v2], quality_scores)

        # v1 should dominate
        assert result[0] > result[1]

    def test_quality_scores_normalized(self):
        """Test that quality scores are normalized."""
        reducer = AveragingReducer(normalize=False)
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([0.0, 1.0])

        # Equal quality scores that don't sum to 1
        quality_scores = [5.0, 5.0]

        result = reducer.reduce_with_quality([v1, v2], quality_scores)

        # Should be equal to simple average
        expected = torch.tensor([0.5, 0.5])
        assert torch.allclose(result, expected, atol=0.01)

    def test_zero_quality_scores(self):
        """Test handling zero quality scores."""
        reducer = AveragingReducer(normalize=False)
        vectors = [torch.randn(64) for _ in range(3)]
        quality_scores = [0.0, 0.0, 0.0]  # All zero

        # Should fall back to uniform weights
        result = reducer.reduce_with_quality(vectors, quality_scores)

        assert result.shape == torch.Size([64])


class TestAveragingReducerPropertyBased:
    """Property-based tests using hypothesis."""

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=20)
    def test_output_shape_matches_input(self, n_vectors: int):
        """Test output shape always matches input shape."""
        reducer = AveragingReducer()
        torch.manual_seed(42)
        vectors = [torch.randn(128) for _ in range(n_vectors)]

        result = reducer.reduce(vectors)

        assert result.shape == vectors[0].shape

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=20)
    def test_normalized_output_has_unit_norm(self, n_vectors: int):
        """Test normalized output has unit norm (when non-zero)."""
        reducer = AveragingReducer(normalize=True)
        torch.manual_seed(42)
        vectors = [torch.randn(128) for _ in range(n_vectors)]

        result = reducer.reduce(vectors)

        # Skip if result is near-zero
        if result.norm() > 1e-6:
            assert torch.isclose(result.norm(), torch.tensor(1.0), atol=1e-5)

    @given(st.lists(st.floats(min_value=0.01, max_value=10.0), min_size=3, max_size=3))
    @settings(max_examples=20)
    def test_weights_produce_weighted_result(self, weights: list):
        """Test that weights influence the result."""
        reducer = AveragingReducer(normalize=False)

        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0])
        v3 = torch.tensor([0.0, 0.0, 1.0])

        # Normalize weights
        total = sum(weights)
        normalized_weights = [w / total for w in weights]

        result = reducer.reduce([v1, v2, v3], weights=normalized_weights)

        # Result should be approximately the weighted combination
        expected = torch.tensor(normalized_weights)
        assert torch.allclose(result, expected, atol=1e-5)
