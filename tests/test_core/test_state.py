"""Tests for vector_forge.core.state module."""

import pytest
import torch
from datetime import datetime
from unittest.mock import MagicMock, patch

from vector_forge.core.state import (
    ExtractionState,
    TranscriptEntry,
    Checkpoint,
)
from vector_forge.core.results import DatapointQuality, OptimizationMetrics


class TestTranscriptEntry:
    """Tests for TranscriptEntry."""

    def test_creation(self):
        """Test creating a transcript entry."""
        entry = TranscriptEntry(
            timestamp=datetime.now(),
            action="generate_prompts",
            details={"num_prompts": 10},
            agent="extractor",
        )
        assert entry.action == "generate_prompts"
        assert entry.details["num_prompts"] == 10
        assert entry.agent == "extractor"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        entry = TranscriptEntry(
            timestamp=timestamp,
            action="optimize",
            details={"layer": 15, "loss": 0.5},
            agent="extractor",
        )
        result = entry.to_dict()

        assert result["timestamp"] == timestamp.isoformat()
        assert result["action"] == "optimize"
        assert result["details"]["layer"] == 15
        assert result["details"]["loss"] == 0.5
        assert result["agent"] == "extractor"

    def test_default_agent(self):
        """Test default agent is extractor."""
        entry = TranscriptEntry(
            timestamp=datetime.now(),
            action="test",
            details={},
        )
        assert entry.agent == "extractor"


class TestCheckpoint:
    """Tests for Checkpoint."""

    def test_create_checkpoint(self):
        """Test creating a checkpoint from state."""
        state = ExtractionState()
        state.best_score = 0.85
        state.outer_iteration = 2

        checkpoint = Checkpoint.create(state, "Before major change")

        assert checkpoint.description == "Before major change"
        assert len(checkpoint.id) == 8
        assert checkpoint.state_snapshot.best_score == 0.85
        assert checkpoint.state_snapshot.outer_iteration == 2
        assert isinstance(checkpoint.timestamp, datetime)

    def test_checkpoint_creates_deep_copy(self):
        """Test that checkpoint creates a deep copy."""
        state = ExtractionState()
        state.vectors[10] = torch.randn(768)

        checkpoint = Checkpoint.create(state, "Test")

        # Modify original state
        state.vectors[10] = torch.randn(768)
        state.vectors[15] = torch.randn(768)

        # Checkpoint should have original value
        assert 10 in checkpoint.state_snapshot.vectors
        assert 15 not in checkpoint.state_snapshot.vectors


class TestExtractionState:
    """Tests for ExtractionState."""

    def test_initial_state(self):
        """Test initial state values."""
        state = ExtractionState()

        assert state.datapoints == []
        assert state.datapoint_qualities == {}
        assert state.vectors == {}
        assert state.optimization_metrics == {}
        assert state.best_layer is None
        assert state.best_strength == 1.0
        assert state.best_score == 0.0
        assert state.evaluations == []
        assert state.current_evaluation is None
        assert state.outer_iteration == 0
        assert state.inner_iteration == 0
        assert state.checkpoints == {}
        assert state.transcript == []

    def test_add_datapoint(self):
        """Test adding a datapoint."""
        state = ExtractionState()

        # Create a mock datapoint
        datapoint = MagicMock()
        dp_id = state.add_datapoint(datapoint)

        assert dp_id == "dp_0"
        assert len(state.datapoints) == 1
        assert state.datapoints[0] is datapoint
        assert "dp_0" in state.datapoint_qualities
        assert state.datapoint_qualities["dp_0"].datapoint_id == "dp_0"

    def test_add_multiple_datapoints(self):
        """Test adding multiple datapoints."""
        state = ExtractionState()

        for i in range(5):
            dp = MagicMock()
            dp_id = state.add_datapoint(dp)
            assert dp_id == f"dp_{i}"

        assert len(state.datapoints) == 5
        assert len(state.datapoint_qualities) == 5

    def test_remove_datapoint(self):
        """Test removing a datapoint."""
        state = ExtractionState()

        dp1 = MagicMock()
        dp2 = MagicMock()
        state.add_datapoint(dp1)
        state.add_datapoint(dp2)

        result = state.remove_datapoint("dp_0")

        assert result is True
        assert len(state.datapoints) == 1
        assert state.datapoints[0] is dp2
        assert "dp_0" not in state.datapoint_qualities

    def test_remove_nonexistent_datapoint(self):
        """Test removing a non-existent datapoint returns False."""
        state = ExtractionState()
        result = state.remove_datapoint("dp_999")
        assert result is False

    def test_remove_invalid_id(self):
        """Test removing with invalid ID format returns False."""
        state = ExtractionState()
        assert state.remove_datapoint("invalid") is False
        assert state.remove_datapoint("dp_abc") is False
        assert state.remove_datapoint("") is False

    def test_set_vector(self):
        """Test storing optimization result."""
        state = ExtractionState()
        vector = torch.randn(768)
        metrics = OptimizationMetrics(
            layer=15,
            final_loss=0.1,
            iterations=50,
            vector_norm=1.0,
        )

        state.set_vector(15, vector, metrics)

        assert 15 in state.vectors
        assert torch.equal(state.vectors[15], vector)
        assert state.optimization_metrics[15] is metrics

    def test_update_best(self):
        """Test updating best result."""
        state = ExtractionState()

        # First update
        state.update_best(10, 1.0, 0.7)
        assert state.best_layer == 10
        assert state.best_strength == 1.0
        assert state.best_score == 0.7

        # Better score should update
        state.update_best(15, 1.5, 0.85)
        assert state.best_layer == 15
        assert state.best_strength == 1.5
        assert state.best_score == 0.85

        # Worse score should not update
        state.update_best(20, 2.0, 0.6)
        assert state.best_layer == 15
        assert state.best_score == 0.85

    def test_update_best_equal_score(self):
        """Test that equal score does not update."""
        state = ExtractionState()
        state.update_best(10, 1.0, 0.7)
        state.update_best(15, 1.5, 0.7)

        # Should keep original (equal score doesn't update)
        assert state.best_layer == 10

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        state = ExtractionState()
        state.best_score = 0.8
        state.outer_iteration = 1

        checkpoint_id = state.create_checkpoint("Before experiment")

        assert len(checkpoint_id) == 8
        assert checkpoint_id in state.checkpoints
        assert state.checkpoints[checkpoint_id].description == "Before experiment"

        # Check transcript was updated
        assert len(state.transcript) == 1
        assert state.transcript[0].action == "checkpoint_created"

    def test_rollback_to(self):
        """Test rolling back to a checkpoint."""
        state = ExtractionState()
        state.best_score = 0.5

        # Create checkpoint
        checkpoint_id = state.create_checkpoint("Initial")

        # Modify state
        state.best_score = 0.9
        state.outer_iteration = 3
        state.vectors[10] = torch.randn(768)

        # Rollback
        result = state.rollback_to(checkpoint_id)

        assert result is True
        assert state.best_score == 0.5
        assert state.outer_iteration == 0
        assert 10 not in state.vectors

    def test_rollback_preserves_transcript(self):
        """Test that rollback preserves transcript."""
        state = ExtractionState()
        checkpoint_id = state.create_checkpoint("Test")
        state.log_action("some_action", {})

        state.rollback_to(checkpoint_id)

        # Transcript should have 3 entries: checkpoint_created, some_action, rollback
        assert len(state.transcript) == 3
        assert state.transcript[-1].action == "rollback"

    def test_rollback_preserves_checkpoints(self):
        """Test that rollback preserves checkpoints."""
        state = ExtractionState()
        cp1 = state.create_checkpoint("First")
        cp2 = state.create_checkpoint("Second")

        state.rollback_to(cp1)

        # Both checkpoints should still exist
        assert cp1 in state.checkpoints
        assert cp2 in state.checkpoints

    def test_rollback_nonexistent_checkpoint(self):
        """Test rollback to non-existent checkpoint returns False."""
        state = ExtractionState()
        result = state.rollback_to("nonexistent")
        assert result is False

    def test_log_action(self):
        """Test logging an action."""
        state = ExtractionState()
        state.log_action("test_action", {"key": "value"}, agent="judge")

        assert len(state.transcript) == 1
        entry = state.transcript[0]
        assert entry.action == "test_action"
        assert entry.details["key"] == "value"
        assert entry.agent == "judge"
        assert isinstance(entry.timestamp, datetime)

    def test_get_transcript_summary(self):
        """Test getting transcript summary."""
        state = ExtractionState()
        state.log_action("action1", {"a": 1})
        state.log_action("action2", {"b": 2}, agent="judge")

        summary = state.get_transcript_summary()

        assert len(summary) == 2
        assert summary[0]["action"] == "action1"
        assert summary[1]["action"] == "action2"
        assert "timestamp" in summary[0]

    def test_clear_vectors(self):
        """Test clearing all vectors."""
        state = ExtractionState()
        state.vectors[10] = torch.randn(768)
        state.vectors[15] = torch.randn(768)
        state.optimization_metrics[10] = MagicMock()
        state.best_layer = 15
        state.best_score = 0.9

        state.clear_vectors()

        assert state.vectors == {}
        assert state.optimization_metrics == {}
        assert state.best_layer is None
        assert state.best_score == 0.0


class TestExtractionStateDataPointOperations:
    """Additional tests for datapoint operations."""

    def test_datapoint_quality_initialization(self):
        """Test that datapoint quality is initialized correctly."""
        state = ExtractionState()
        dp = MagicMock()
        dp_id = state.add_datapoint(dp)

        quality = state.datapoint_qualities[dp_id]
        assert quality.datapoint_id == dp_id
        assert quality.leave_one_out_influence is None
        assert quality.avg_loss_contribution == 0.0

    def test_remove_datapoint_updates_qualities(self):
        """Test that removing datapoint also removes quality."""
        state = ExtractionState()
        dp = MagicMock()
        dp_id = state.add_datapoint(dp)

        assert dp_id in state.datapoint_qualities
        state.remove_datapoint(dp_id)
        assert dp_id not in state.datapoint_qualities
