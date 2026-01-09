"""Adaptive layer search strategy."""

from typing import List, Optional, Dict


class AdaptiveLayerStrategy:
    """
    Adaptively search for the best layer.

    First iteration: Try a coarse set of layers spread across the model.
    Subsequent iterations: Refine around the best layer found so far.

    Example:
        >>> strategy = AdaptiveLayerStrategy()
        >>> # First iteration: coarse search
        >>> layers = strategy.get_layers_to_try(32, iteration=0)  # [4, 8, 12, 16, 20, 24, 28]
        >>> # After eval, best was layer 16
        >>> # Second iteration: refine around 16
        >>> layers = strategy.get_layers_to_try(32, iteration=1, {16: 0.8, 12: 0.6, ...})
        >>> # [14, 15, 16, 17, 18]
    """

    def __init__(
        self,
        initial_step: int = 4,
        refinement_radius: int = 2,
        max_refinements: int = 3,
    ):
        """
        Args:
            initial_step: Step size for initial coarse search.
            refinement_radius: How many layers around best to try in refinement.
            max_refinements: Maximum refinement iterations before stopping.
        """
        self.initial_step = initial_step
        self.refinement_radius = refinement_radius
        self.max_refinements = max_refinements

    def get_layers_to_try(
        self,
        total_layers: int,
        iteration: int = 0,
        previous_results: Optional[Dict[int, float]] = None,
    ) -> List[int]:
        """
        Get layers for this iteration.

        Args:
            total_layers: Total number of layers in the model.
            iteration: Current search iteration (0 = initial, 1+ = refinement).
            previous_results: Results from previous iterations {layer: score}.
                            Higher scores are better.

        Returns:
            List of layer indices to try.
        """
        if iteration == 0 or previous_results is None:
            # Initial coarse search
            return self._coarse_search(total_layers)

        if iteration > self.max_refinements:
            # Max refinements reached, return best
            best_layer = max(previous_results.keys(), key=lambda l: previous_results[l])
            return [best_layer]

        # Refinement: search around best layer
        return self._refine_search(total_layers, previous_results)

    def _coarse_search(self, total_layers: int) -> List[int]:
        """Generate initial coarse layer set."""
        # Skip very early and very late layers
        start = max(2, total_layers // 8)
        end = min(total_layers - 2, total_layers - total_layers // 8)

        layers = list(range(start, end, self.initial_step))

        # Ensure we have at least some layers
        if not layers:
            middle = total_layers // 2
            layers = [middle]

        return layers

    def _refine_search(
        self,
        total_layers: int,
        previous_results: Dict[int, float],
    ) -> List[int]:
        """Refine search around the best layer."""
        best_layer = max(previous_results.keys(), key=lambda l: previous_results[l])

        # Generate layers around the best
        layers = []
        for offset in range(-self.refinement_radius, self.refinement_radius + 1):
            layer = best_layer + offset
            if 0 <= layer < total_layers and layer not in previous_results:
                layers.append(layer)

        # If all nearby layers already tested, we're done
        if not layers:
            return [best_layer]

        return sorted(layers)

    def should_continue(
        self,
        iteration: int,
        previous_results: Dict[int, float],
        score_threshold: float = 0.01,
    ) -> bool:
        """
        Determine if search should continue.

        Args:
            iteration: Current iteration.
            previous_results: Results so far.
            score_threshold: Minimum improvement to continue.

        Returns:
            True if search should continue, False to stop.
        """
        if iteration == 0:
            return True

        if iteration > self.max_refinements:
            return False

        # Check if we're still improving
        if len(previous_results) < 2:
            return True

        scores = sorted(previous_results.values(), reverse=True)
        improvement = scores[0] - scores[1]

        return improvement > score_threshold
