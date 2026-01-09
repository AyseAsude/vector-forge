"""PCA-based noise reduction strategy."""

from typing import List, Any, Optional

import torch
import numpy as np


class PCAReducer:
    """
    Reduce noise by projecting vectors to principal components.

    The idea is that the "signal" (actual behavior direction) will be
    captured in the top principal components, while noise will be
    in the lower components.

    Example:
        >>> reducer = PCAReducer(n_components=5)
        >>> vectors = [train_on_subset(i) for i in range(10)]
        >>> clean_vector = reducer.reduce(vectors)
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
    ):
        """
        Args:
            n_components: Number of principal components to keep.
                         If None, determined by variance_threshold.
            variance_threshold: Keep components explaining this much variance.
                              Only used if n_components is None.
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold

    def reduce(
        self,
        vectors: List[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Project vectors to principal components and return the first PC.

        Args:
            vectors: List of steering vectors.
            **kwargs: Additional arguments (unused).

        Returns:
            The first principal component as the cleaned vector.

        Raises:
            ValueError: If vectors list has fewer than 2 vectors.
        """
        if len(vectors) < 2:
            if vectors:
                return vectors[0].clone()
            raise ValueError("Need at least 1 vector")

        # Stack vectors into matrix (n_vectors x hidden_dim)
        device = vectors[0].device
        dtype = vectors[0].dtype

        matrix = torch.stack(vectors).cpu().numpy()

        # Center the data
        mean = matrix.mean(axis=0)
        centered = matrix - mean

        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Determine number of components
        if self.n_components is not None:
            n_components = min(self.n_components, len(S))
        else:
            # Use variance threshold
            total_var = (S ** 2).sum()
            cumulative_var = np.cumsum(S ** 2) / total_var
            n_components = np.searchsorted(cumulative_var, self.variance_threshold) + 1
            n_components = min(n_components, len(S))

        # The first principal component (direction of most variance)
        # This is the first row of Vt
        first_pc = Vt[0]

        # Convert back to tensor
        result = torch.tensor(first_pc, device=device, dtype=dtype)

        # Normalize
        norm = result.norm()
        if norm > 0:
            result = result / norm

        return result

    def reduce_weighted(
        self,
        vectors: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        PCA with weighted vectors.

        Args:
            vectors: List of steering vectors.
            weights: Optional weights for each vector.

        Returns:
            First principal component of weighted data.
        """
        if len(vectors) < 2:
            if vectors:
                return vectors[0].clone()
            raise ValueError("Need at least 1 vector")

        device = vectors[0].device
        dtype = vectors[0].dtype

        matrix = torch.stack(vectors).cpu().numpy()

        if weights:
            weights_array = np.array(weights)
            weights_array = weights_array / weights_array.sum()  # Normalize
            weights_sqrt = np.sqrt(weights_array).reshape(-1, 1)
            matrix = matrix * weights_sqrt

        mean = matrix.mean(axis=0)
        centered = matrix - mean

        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        first_pc = Vt[0]
        result = torch.tensor(first_pc, device=device, dtype=dtype)

        norm = result.norm()
        if norm > 0:
            result = result / norm

        return result

    def get_variance_explained(
        self,
        vectors: List[torch.Tensor],
    ) -> List[float]:
        """
        Get the variance explained by each principal component.

        Useful for understanding how much "signal" vs "noise" is in the vectors.

        Args:
            vectors: List of steering vectors.

        Returns:
            List of variance ratios for each component.
        """
        if len(vectors) < 2:
            return [1.0]

        matrix = torch.stack(vectors).cpu().numpy()
        mean = matrix.mean(axis=0)
        centered = matrix - mean

        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        variances = S ** 2
        total_var = variances.sum()

        return (variances / total_var).tolist()
