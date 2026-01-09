"""Averaging noise reduction strategy."""

from typing import List, Any

import torch


class AveragingReducer:
    """
    Reduce noise by averaging multiple vectors.

    When vectors are trained with different random seeds, the consistent
    signal (the actual behavior direction) will be reinforced while
    random noise will cancel out.

    Example:
        >>> reducer = AveragingReducer()
        >>> vectors = [train_with_seed(i) for i in range(5)]
        >>> clean_vector = reducer.reduce(vectors)
    """

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: If True, normalize the averaged vector to unit norm.
        """
        self.normalize = normalize

    def reduce(
        self,
        vectors: List[torch.Tensor],
        weights: List[float] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Average multiple vectors to reduce noise.

        Args:
            vectors: List of steering vectors to average.
            weights: Optional weights for weighted average (must sum to 1).
            **kwargs: Additional arguments (unused).

        Returns:
            Averaged (and optionally normalized) vector.

        Raises:
            ValueError: If vectors list is empty or shapes don't match.
        """
        if not vectors:
            raise ValueError("Cannot average empty list of vectors")

        if len(vectors) == 1:
            return vectors[0].clone()

        # Check shapes match
        shape = vectors[0].shape
        for i, v in enumerate(vectors[1:], 1):
            if v.shape != shape:
                raise ValueError(f"Vector {i} shape {v.shape} doesn't match {shape}")

        # Stack and average
        stacked = torch.stack(vectors)

        if weights:
            if len(weights) != len(vectors):
                raise ValueError("Weights length must match vectors length")
            weights_tensor = torch.tensor(weights, device=stacked.device, dtype=stacked.dtype)
            weights_tensor = weights_tensor.view(-1, *([1] * len(shape)))
            averaged = (stacked * weights_tensor).sum(dim=0)
        else:
            averaged = stacked.mean(dim=0)

        if self.normalize:
            norm = averaged.norm()
            if norm > 0:
                averaged = averaged / norm

        return averaged

    def reduce_with_quality(
        self,
        vectors: List[torch.Tensor],
        quality_scores: List[float],
    ) -> torch.Tensor:
        """
        Average vectors weighted by their quality scores.

        Higher quality vectors contribute more to the final result.

        Args:
            vectors: List of steering vectors.
            quality_scores: Quality score for each vector (higher = better).

        Returns:
            Quality-weighted averaged vector.
        """
        # Normalize scores to sum to 1
        total = sum(quality_scores)
        if total == 0:
            weights = [1.0 / len(vectors)] * len(vectors)
        else:
            weights = [s / total for s in quality_scores]

        return self.reduce(vectors, weights=weights)
