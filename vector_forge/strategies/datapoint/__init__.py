"""Datapoint generation strategies."""

from vector_forge.strategies.datapoint.contrastive import ContrastiveStrategy
from vector_forge.strategies.datapoint.single_direction import SingleDirectionStrategy

__all__ = [
    "ContrastiveStrategy",
    "SingleDirectionStrategy",
]
