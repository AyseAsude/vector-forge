"""Data adapters for converting between vector-forge and steering-vectors formats.

Implements the Adapter pattern to bridge the gap between our contrast generation
pipeline and the steering-vectors optimization library.

Architecture:
- Single Responsibility: Each adapter handles one conversion
- Dependency Inversion: Adapters depend on abstractions (protocols)
- Interface Segregation: Separate adapters for different conversion needs
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
import json
import random

from steering_vectors import TrainingDatapoint

from vector_forge.contrast.protocols import (
    ValidatedPair,
    SampleDataset,
    BehaviorAnalysis,
)


# ============================================================================
# Data Adapter - ValidatedPair to TrainingDatapoint
# ============================================================================


class ContrastToTrainingAdapter:
    """Adapts ValidatedPairs to TrainingDatapoints for optimization.

    Converts our contrast generation output format to the format expected
    by the steering-vectors optimization library.

    Example:
        >>> adapter = ContrastToTrainingAdapter()
        >>> datapoints = adapter.convert_dataset(sample_dataset)
        >>> # Now use with SteeringOptimizer
    """

    def convert_pair(self, pair: ValidatedPair) -> TrainingDatapoint:
        """Convert a single ValidatedPair to TrainingDatapoint.

        Args:
            pair: Validated contrast pair with prompt, dst, and src.

        Returns:
            TrainingDatapoint ready for optimization.
        """
        return TrainingDatapoint(
            prompt=pair.prompt,
            dst_completions=[pair.dst],  # What we want the model to say
            src_completions=[pair.src],  # What we don't want
        )

    def convert_pairs(self, pairs: List[ValidatedPair]) -> List[TrainingDatapoint]:
        """Convert multiple pairs to datapoints.

        Args:
            pairs: List of validated contrast pairs.

        Returns:
            List of TrainingDatapoints.
        """
        return [self.convert_pair(p) for p in pairs]

    def convert_dataset(
        self,
        dataset: SampleDataset,
        include_invalid: bool = False,
    ) -> List[TrainingDatapoint]:
        """Convert a SampleDataset to list of TrainingDatapoints.

        Args:
            dataset: Dataset containing core and unique pairs.
            include_invalid: If True, include pairs that failed validation.

        Returns:
            List of TrainingDatapoints.
        """
        pairs = dataset.valid_pairs if not include_invalid else dataset.all_pairs
        return self.convert_pairs(pairs)

    def convert_with_bootstrap(
        self,
        pairs: List[ValidatedPair],
        ratio: float = 1.0,
        seed: Optional[int] = None,
    ) -> List[TrainingDatapoint]:
        """Convert pairs with optional bootstrap sampling.

        Args:
            pairs: List of validated pairs.
            ratio: Fraction of pairs to use (0.5 to 1.0).
            seed: Random seed for reproducibility.

        Returns:
            Bootstrapped list of TrainingDatapoints.
        """
        if ratio >= 1.0:
            return self.convert_pairs(pairs)

        if seed is not None:
            random.seed(seed)

        n_select = max(1, int(len(pairs) * ratio))
        selected = random.sample(pairs, n_select)
        return self.convert_pairs(selected)


# ============================================================================
# Serialization for Storage
# ============================================================================


@dataclass
class SerializedDatapoint:
    """Serializable representation of a training datapoint.

    Used for saving datapoints to disk for reproducibility.
    """

    prompt: str
    dst_completions: List[str]
    src_completions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_training_datapoint(self) -> TrainingDatapoint:
        """Convert back to TrainingDatapoint."""
        return TrainingDatapoint(
            prompt=self.prompt,
            dst_completions=self.dst_completions,
            src_completions=self.src_completions,
        )

    @classmethod
    def from_training_datapoint(
        cls,
        dp: TrainingDatapoint,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SerializedDatapoint":
        """Create from TrainingDatapoint."""
        return cls(
            prompt=dp.prompt,
            dst_completions=list(dp.dst_completions),
            src_completions=list(dp.src_completions),
            metadata=metadata or {},
        )

    @classmethod
    def from_validated_pair(
        cls,
        pair: ValidatedPair,
    ) -> "SerializedDatapoint":
        """Create from ValidatedPair with full metadata."""
        metadata = {
            "seed_scenario": pair.seed.scenario if pair.seed else None,
            "seed_context": pair.seed.context if pair.seed else None,
            "seed_quality": pair.seed.quality_score if pair.seed else None,
            "validation": {
                "is_valid": pair.is_valid,
                "contrast_quality": pair.validation.contrast_quality if pair.validation else None,
                "scores": pair.validation.evaluated_scores if pair.validation else None,
                "weakest_dimension": pair.validation.weakest_dimension if pair.validation else None,
            } if pair.validation else None,
            "attempts": pair.attempts,
            "pair_metadata": pair.metadata,
        }
        return cls(
            prompt=pair.prompt,
            dst_completions=[pair.dst],
            src_completions=[pair.src],
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SerializedDatapoint":
        """Create from dictionary."""
        return cls(
            prompt=data["prompt"],
            dst_completions=data["dst_completions"],
            src_completions=data["src_completions"],
            metadata=data.get("metadata", {}),
        )


class DatapointSerializer:
    """Serializes and deserializes training datapoints.

    Handles conversion between TrainingDatapoints and JSON-serializable formats
    for storage and reproducibility.
    """

    def serialize_datapoints(
        self,
        datapoints: List[TrainingDatapoint],
    ) -> str:
        """Serialize datapoints to JSON string.

        Args:
            datapoints: List of TrainingDatapoints.

        Returns:
            JSON string representation.
        """
        serialized = [
            SerializedDatapoint.from_training_datapoint(dp)
            for dp in datapoints
        ]
        return json.dumps([s.to_dict() for s in serialized], indent=2)

    def deserialize_datapoints(
        self,
        json_str: str,
    ) -> List[TrainingDatapoint]:
        """Deserialize datapoints from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            List of TrainingDatapoints.
        """
        data = json.loads(json_str)
        serialized = [SerializedDatapoint.from_dict(d) for d in data]
        return [s.to_training_datapoint() for s in serialized]

    def serialize_from_pairs(
        self,
        pairs: List[ValidatedPair],
    ) -> str:
        """Serialize validated pairs with full metadata.

        Args:
            pairs: List of ValidatedPairs.

        Returns:
            JSON string with full pair metadata.
        """
        serialized = [
            SerializedDatapoint.from_validated_pair(p)
            for p in pairs
        ]
        return json.dumps([s.to_dict() for s in serialized], indent=2)


# ============================================================================
# Optimization Result Adapter
# ============================================================================


@dataclass
class OptimizationResultData:
    """Serializable optimization result for storage.

    Contains all information needed to reproduce and analyze
    the optimization process.
    """

    # Core result
    vector_shape: Tuple[int, ...]
    layer: int
    final_loss: float
    iterations: int

    # History
    loss_history: List[float] = field(default_factory=list)

    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "vector_shape": list(self.vector_shape),
            "layer": self.layer,
            "final_loss": self.final_loss,
            "iterations": self.iterations,
            "loss_history": self.loss_history,
            "config": self.config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResultData":
        """Create from dictionary."""
        return cls(
            vector_shape=tuple(data["vector_shape"]),
            layer=data["layer"],
            final_loss=data["final_loss"],
            iterations=data["iterations"],
            loss_history=data.get("loss_history", []),
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def pairs_to_datapoints(pairs: List[ValidatedPair]) -> List[TrainingDatapoint]:
    """Quick conversion from pairs to datapoints.

    Args:
        pairs: List of ValidatedPairs.

    Returns:
        List of TrainingDatapoints.
    """
    adapter = ContrastToTrainingAdapter()
    return adapter.convert_pairs(pairs)


def dataset_to_datapoints(
    dataset: SampleDataset,
    include_invalid: bool = False,
) -> List[TrainingDatapoint]:
    """Quick conversion from dataset to datapoints.

    Args:
        dataset: SampleDataset with pairs.
        include_invalid: Include invalid pairs.

    Returns:
        List of TrainingDatapoints.
    """
    adapter = ContrastToTrainingAdapter()
    return adapter.convert_dataset(dataset, include_invalid)
