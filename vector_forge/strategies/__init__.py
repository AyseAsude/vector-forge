"""Pluggable strategies for the extraction pipeline."""

from vector_forge.strategies.datapoint import (
    ContrastiveStrategy,
    SingleDirectionStrategy,
)
from vector_forge.strategies.noise import (
    AveragingReducer,
    PCAReducer,
)
from vector_forge.strategies.layer_search import (
    FixedLayerStrategy,
    AdaptiveLayerStrategy,
)

__all__ = [
    "ContrastiveStrategy",
    "SingleDirectionStrategy",
    "AveragingReducer",
    "PCAReducer",
    "FixedLayerStrategy",
    "AdaptiveLayerStrategy",
]
