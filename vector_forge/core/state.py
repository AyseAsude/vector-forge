"""Mutable state for the extraction process."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import copy
import uuid

import torch

from steering_vectors import SteeringVectorTrainingSample as TrainingDatapoint

from vector_forge.core.results import (
    OptimizationMetrics,
    EvaluationResult,
    DatapointQuality,
)


@dataclass
class TranscriptEntry:
    """A single entry in the extraction transcript."""

    timestamp: datetime
    action: str
    details: Dict[str, Any]
    agent: str = "extractor"  # "extractor" or "judge"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "details": self.details,
            "agent": self.agent,
        }


@dataclass
class Checkpoint:
    """A saved state checkpoint for rollback capability."""

    id: str
    timestamp: datetime
    description: str
    state_snapshot: "ExtractionState"

    @classmethod
    def create(cls, state: "ExtractionState", description: str) -> "Checkpoint":
        return cls(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            description=description,
            state_snapshot=copy.deepcopy(state),
        )


@dataclass
class ExtractionState:
    """
    Mutable state for the extraction process.

    Tracks datapoints, optimization results, evaluations, and checkpoints.
    Designed to be serializable for saving/resuming extractions.
    """

    # Training data
    datapoints: List[TrainingDatapoint] = field(default_factory=list)
    datapoint_qualities: Dict[str, DatapointQuality] = field(default_factory=dict)

    # Optimization results per layer
    vectors: Dict[int, torch.Tensor] = field(default_factory=dict)
    optimization_metrics: Dict[int, OptimizationMetrics] = field(default_factory=dict)

    # Best result tracking
    best_layer: Optional[int] = None
    best_strength: float = 1.0
    best_score: float = 0.0

    # Evaluation results
    evaluations: List[EvaluationResult] = field(default_factory=list)
    current_evaluation: Optional[EvaluationResult] = None

    # Iteration tracking
    outer_iteration: int = 0
    inner_iteration: int = 0

    # Checkpoints for rollback
    checkpoints: Dict[str, Checkpoint] = field(default_factory=dict)

    # Transcript for judge review
    transcript: List[TranscriptEntry] = field(default_factory=list)

    def add_datapoint(self, datapoint: TrainingDatapoint) -> str:
        """
        Add a datapoint and return its ID.

        Args:
            datapoint: The training datapoint to add.

        Returns:
            Generated ID for the datapoint.
        """
        dp_id = f"dp_{len(self.datapoints)}"
        self.datapoints.append(datapoint)
        self.datapoint_qualities[dp_id] = DatapointQuality(datapoint_id=dp_id)
        return dp_id

    def remove_datapoint(self, dp_id: str) -> bool:
        """
        Remove a datapoint by ID.

        Args:
            dp_id: ID of the datapoint to remove.

        Returns:
            True if removed, False if not found.
        """
        try:
            idx = int(dp_id.split("_")[1])
            if 0 <= idx < len(self.datapoints):
                self.datapoints.pop(idx)
                self.datapoint_qualities.pop(dp_id, None)
                return True
        except (ValueError, IndexError):
            pass
        return False

    def set_vector(self, layer: int, vector: torch.Tensor, metrics: OptimizationMetrics) -> None:
        """
        Store optimization result for a layer.

        Args:
            layer: Layer index.
            vector: The optimized steering vector.
            metrics: Optimization metrics.
        """
        self.vectors[layer] = vector
        self.optimization_metrics[layer] = metrics

    def update_best(self, layer: int, strength: float, score: float) -> None:
        """
        Update best result if this is better.

        Args:
            layer: Layer index.
            strength: Steering strength.
            score: Evaluation score.
        """
        if score > self.best_score:
            self.best_layer = layer
            self.best_strength = strength
            self.best_score = score

    def create_checkpoint(self, description: str) -> str:
        """
        Create a checkpoint of current state.

        Args:
            description: Human-readable description of this checkpoint.

        Returns:
            Checkpoint ID.
        """
        checkpoint = Checkpoint.create(self, description)
        self.checkpoints[checkpoint.id] = checkpoint
        self.log_action("checkpoint_created", {"id": checkpoint.id, "description": description})
        return checkpoint.id

    def rollback_to(self, checkpoint_id: str) -> bool:
        """
        Rollback state to a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to rollback to.

        Returns:
            True if successful, False if checkpoint not found.
        """
        if checkpoint_id not in self.checkpoints:
            return False

        checkpoint = self.checkpoints[checkpoint_id]
        snapshot = checkpoint.state_snapshot

        # Restore state (but keep transcript and checkpoints)
        self.datapoints = copy.deepcopy(snapshot.datapoints)
        self.datapoint_qualities = copy.deepcopy(snapshot.datapoint_qualities)
        self.vectors = copy.deepcopy(snapshot.vectors)
        self.optimization_metrics = copy.deepcopy(snapshot.optimization_metrics)
        self.best_layer = snapshot.best_layer
        self.best_strength = snapshot.best_strength
        self.best_score = snapshot.best_score
        self.evaluations = copy.deepcopy(snapshot.evaluations)
        self.current_evaluation = copy.deepcopy(snapshot.current_evaluation)
        self.outer_iteration = snapshot.outer_iteration
        self.inner_iteration = snapshot.inner_iteration

        self.log_action("rollback", {"checkpoint_id": checkpoint_id})
        return True

    def log_action(self, action: str, details: Dict[str, Any], agent: str = "extractor") -> None:
        """
        Log an action to the transcript.

        Args:
            action: Action identifier.
            details: Action details.
            agent: Which agent performed this action.
        """
        entry = TranscriptEntry(
            timestamp=datetime.now(),
            action=action,
            details=details,
            agent=agent,
        )
        self.transcript.append(entry)

    def get_transcript_summary(self) -> List[Dict[str, Any]]:
        """Get transcript as list of dicts for serialization."""
        return [entry.to_dict() for entry in self.transcript]

    def clear_vectors(self) -> None:
        """Clear all optimized vectors (for re-optimization)."""
        self.vectors.clear()
        self.optimization_metrics.clear()
        self.best_layer = None
        self.best_score = 0.0
