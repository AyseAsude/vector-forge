"""Mutable state for the extraction process."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime
import copy
import uuid

import torch

from steering_vectors import TrainingDatapoint

from vector_forge.core.results import (
    OptimizationMetrics,
    EvaluationResult,
    DatapointQuality,
)

if TYPE_CHECKING:
    from vector_forge.storage import SessionStore


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
    Emits events to session store for complete reproducibility.
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

    # Session store for event capture (not part of state serialization)
    _store: Optional["SessionStore"] = field(default=None, repr=False, compare=False)

    # Vector version tracking per layer
    _vector_versions: Dict[int, int] = field(default_factory=dict, repr=False, compare=False)

    def set_store(self, store: Optional["SessionStore"]) -> None:
        """Set the session store for event capture.

        Args:
            store: Session store to use.
        """
        self._store = store

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

        # Emit event
        self._emit_datapoint_added(dp_id, datapoint)

        return dp_id

    def remove_datapoint(self, dp_id: str, reason: str = "") -> bool:
        """
        Remove a datapoint by ID.

        Args:
            dp_id: ID of the datapoint to remove.
            reason: Reason for removal.

        Returns:
            True if removed, False if not found.
        """
        try:
            idx = int(dp_id.split("_")[1])
            if 0 <= idx < len(self.datapoints):
                self.datapoints.pop(idx)
                self.datapoint_qualities.pop(dp_id, None)

                # Emit event
                self._emit_datapoint_removed(dp_id, reason)

                return True
        except (ValueError, IndexError):
            pass
        return False

    def update_datapoint_quality(
        self,
        dp_id: str,
        quality: DatapointQuality,
    ) -> None:
        """Update quality metrics for a datapoint.

        Args:
            dp_id: Datapoint ID.
            quality: Quality metrics.
        """
        self.datapoint_qualities[dp_id] = quality

        # Emit event
        self._emit_datapoint_quality(dp_id, quality)

    def set_vector(
        self,
        layer: int,
        vector: torch.Tensor,
        metrics: OptimizationMetrics,
    ) -> None:
        """
        Store optimization result for a layer.

        Args:
            layer: Layer index.
            vector: The optimized steering vector.
            metrics: Optimization metrics.
        """
        self.vectors[layer] = vector
        self.optimization_metrics[layer] = metrics

        # Emit event and save vector
        self._emit_vector_created(layer, vector, metrics)

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

            # Emit event
            self._emit_vector_selected(layer, strength, score)

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

        # Emit event
        self._emit_checkpoint_created(checkpoint.id, description)

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

        # Emit event
        self._emit_checkpoint_rollback(checkpoint_id)

        return True

    def set_iteration(self, iteration_type: str, iteration: int) -> None:
        """Set iteration number and emit event.

        Args:
            iteration_type: "outer" or "inner".
            iteration: Iteration number.
        """
        if iteration_type == "outer":
            self.outer_iteration = iteration
        else:
            self.inner_iteration = iteration

        self._emit_iteration_started(iteration_type, iteration)

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

    # =========================================================================
    # Event emission methods
    # =========================================================================

    def _emit_datapoint_added(self, dp_id: str, datapoint: TrainingDatapoint) -> None:
        """Emit datapoint added event."""
        if self._store is None:
            return

        from vector_forge.storage import DatapointAddedEvent

        event = DatapointAddedEvent(
            datapoint_id=dp_id,
            prompt=getattr(datapoint, 'prompt', ''),
            positive_completion=getattr(datapoint, 'positive_str', getattr(datapoint, 'positive', '')),
            negative_completion=getattr(datapoint, 'negative_str', getattr(datapoint, 'negative', None)),
        )

        self._store.append_event(event, source="state")

    def _emit_datapoint_removed(self, dp_id: str, reason: str) -> None:
        """Emit datapoint removed event."""
        if self._store is None:
            return

        from vector_forge.storage import DatapointRemovedEvent

        event = DatapointRemovedEvent(
            datapoint_id=dp_id,
            reason=reason,
        )

        self._store.append_event(event, source="state")

    def _emit_datapoint_quality(self, dp_id: str, quality: DatapointQuality) -> None:
        """Emit datapoint quality event."""
        if self._store is None:
            return

        from vector_forge.storage import DatapointQualityEvent

        event = DatapointQualityEvent(
            datapoint_id=dp_id,
            leave_one_out_influence=quality.leave_one_out_influence,
            gradient_alignment=quality.gradient_alignment or 0.0,
            avg_loss_contribution=quality.avg_loss_contribution or 0.0,
            quality_score=0.0,  # Computed elsewhere
            recommendation="KEEP",  # Default
            is_outlier=quality.is_outlier or False,
        )

        self._store.append_event(event, source="state")

    def _emit_vector_created(
        self,
        layer: int,
        vector: torch.Tensor,
        metrics: OptimizationMetrics,
    ) -> None:
        """Emit vector created event and save vector file."""
        if self._store is None:
            return

        from vector_forge.storage import VectorCreatedEvent

        # Increment version for this layer
        version = self._vector_versions.get(layer, 0) + 1
        self._vector_versions[layer] = version

        # Save vector to file
        vector_ref = self._store.save_vector(vector, layer, version)

        # Generate vector ID
        vector_id = f"vec_L{layer}_v{version}"

        event = VectorCreatedEvent(
            vector_id=vector_id,
            layer=layer,
            vector_ref=vector_ref,
            shape=list(vector.shape),
            dtype=str(vector.dtype),
            norm=vector.norm().item(),
            optimization_metrics={
                "final_loss": metrics.final_loss,
                "iterations": metrics.iterations,
                "vector_norm": metrics.vector_norm,
            },
        )

        self._store.append_event(event, source="state")

    def _emit_vector_selected(self, layer: int, strength: float, score: float) -> None:
        """Emit vector selected event."""
        if self._store is None:
            return

        from vector_forge.storage import VectorSelectedEvent

        # Find vector ID for this layer
        version = self._vector_versions.get(layer, 1)
        vector_id = f"vec_L{layer}_v{version}"

        event = VectorSelectedEvent(
            vector_id=vector_id,
            layer=layer,
            strength=strength,
            score=score,
            reason="highest_score",
        )

        self._store.append_event(event, source="state")

    def _emit_checkpoint_created(self, checkpoint_id: str, description: str) -> None:
        """Emit checkpoint created event."""
        if self._store is None:
            return

        from vector_forge.storage import CheckpointCreatedEvent

        event = CheckpointCreatedEvent(
            checkpoint_id=checkpoint_id,
            description=description,
        )

        self._store.append_event(event, source="state")

    def _emit_checkpoint_rollback(self, checkpoint_id: str) -> None:
        """Emit checkpoint rollback event."""
        if self._store is None:
            return

        from vector_forge.storage import CheckpointRollbackEvent

        # Get current sequence for rollback tracking
        # (Events before this point are effectively "rolled back")
        event = CheckpointRollbackEvent(
            checkpoint_id=checkpoint_id,
        )

        self._store.append_event(event, source="state")

    def _emit_iteration_started(self, iteration_type: str, iteration: int) -> None:
        """Emit iteration started event."""
        if self._store is None:
            return

        from vector_forge.storage import IterationStartedEvent

        event = IterationStartedEvent(
            iteration_type=iteration_type,
            iteration=iteration,
        )

        self._store.append_event(event, source="state")
