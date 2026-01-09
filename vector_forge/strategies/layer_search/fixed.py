"""Fixed layer search strategy."""

from typing import List, Optional, Dict


class FixedLayerStrategy:
    """
    Use a fixed set of layers.

    The simplest strategy - just return the specified layers.

    Example:
        >>> strategy = FixedLayerStrategy(layers=[10, 15, 20])
        >>> layers = strategy.get_layers_to_try(32)  # [10, 15, 20]
    """

    def __init__(self, layers: Optional[List[int]] = None):
        """
        Args:
            layers: Specific layers to use. If None, defaults to middle layers.
        """
        self._layers = layers

    def get_layers_to_try(
        self,
        total_layers: int,
        iteration: int = 0,
        previous_results: Optional[Dict[int, float]] = None,
    ) -> List[int]:
        """
        Return the fixed set of layers.

        Args:
            total_layers: Total number of layers in the model.
            iteration: Ignored for fixed strategy.
            previous_results: Ignored for fixed strategy.

        Returns:
            List of layer indices to try.
        """
        if self._layers:
            # Filter to valid layers
            return [l for l in self._layers if 0 <= l < total_layers]

        # Default: middle layers (typically where semantic features are)
        middle = total_layers // 2
        return [
            max(0, middle - 4),
            max(0, middle - 2),
            middle,
            min(total_layers - 1, middle + 2),
            min(total_layers - 1, middle + 4),
        ]

    @classmethod
    def for_model(cls, model_name: str) -> "FixedLayerStrategy":
        """
        Create a strategy with layers optimized for a specific model.

        Args:
            model_name: Model name (e.g., "llama-7b", "gpt2").

        Returns:
            FixedLayerStrategy with recommended layers.
        """
        # Recommended layers based on common findings
        model_layers = {
            "llama": [8, 12, 16, 20],
            "gpt2": [4, 6, 8, 10],
            "mistral": [8, 12, 16, 20],
            "phi": [8, 12, 16],
        }

        model_lower = model_name.lower()
        for key, layers in model_layers.items():
            if key in model_lower:
                return cls(layers=layers)

        # Default
        return cls()
