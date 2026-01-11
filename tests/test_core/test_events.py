"""Tests for vector_forge.core.events module."""

import pytest
import time
from unittest.mock import MagicMock

from vector_forge.core.events import (
    Event,
    EventType,
    create_event,
)
from vector_forge.core.protocols import EventEmitter


class TestEventType:
    """Tests for EventType enum."""

    def test_pipeline_events(self):
        """Test pipeline lifecycle event types."""
        assert EventType.PIPELINE_STARTED.value == "pipeline.started"
        assert EventType.PIPELINE_COMPLETED.value == "pipeline.completed"
        assert EventType.PIPELINE_FAILED.value == "pipeline.failed"

    def test_iteration_events(self):
        """Test iteration event types."""
        assert EventType.OUTER_ITERATION_STARTED.value == "iteration.outer.started"
        assert EventType.OUTER_ITERATION_COMPLETED.value == "iteration.outer.completed"
        assert EventType.INNER_ITERATION_STARTED.value == "iteration.inner.started"
        assert EventType.INNER_ITERATION_COMPLETED.value == "iteration.inner.completed"

    def test_datapoint_events(self):
        """Test datapoint event types."""
        assert EventType.DATAPOINT_GENERATION_STARTED.value == "datapoint.generation.started"
        assert EventType.DATAPOINT_GENERATION_PROGRESS.value == "datapoint.generation.progress"
        assert EventType.DATAPOINT_GENERATION_COMPLETED.value == "datapoint.generation.completed"
        assert EventType.DATAPOINT_REMOVED.value == "datapoint.removed"

    def test_optimization_events(self):
        """Test optimization event types."""
        assert EventType.OPTIMIZATION_STARTED.value == "optimization.started"
        assert EventType.OPTIMIZATION_STEP.value == "optimization.step"
        assert EventType.OPTIMIZATION_COMPLETED.value == "optimization.completed"

    def test_evaluation_events(self):
        """Test evaluation event types."""
        assert EventType.QUICK_EVAL_STARTED.value == "evaluation.quick.started"
        assert EventType.THOROUGH_EVAL_STARTED.value == "evaluation.thorough.started"
        assert EventType.JUDGE_VERDICT.value == "judge.verdict"

    def test_error_events(self):
        """Test error event types."""
        assert EventType.ERROR.value == "error"
        assert EventType.WARNING.value == "warning"

    def test_all_event_types_unique(self):
        """Test that all event type values are unique."""
        values = [e.value for e in EventType]
        assert len(values) == len(set(values))


class TestEvent:
    """Tests for Event dataclass."""

    def test_creation(self):
        """Test creating an event."""
        event = Event(
            type=EventType.PIPELINE_STARTED,
            data={"behavior": "sycophancy"},
            source="pipeline",
        )

        assert event.type == EventType.PIPELINE_STARTED
        assert event.data["behavior"] == "sycophancy"
        assert event.source == "pipeline"

    def test_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        before = time.time()
        event = Event(type=EventType.PIPELINE_STARTED)
        after = time.time()

        assert event.timestamp is not None
        assert before <= event.timestamp <= after

    def test_timestamp_explicit(self):
        """Test explicitly setting timestamp."""
        explicit_time = 1234567890.0
        event = Event(
            type=EventType.PIPELINE_STARTED,
            timestamp=explicit_time,
        )

        assert event.timestamp == explicit_time

    def test_default_data(self):
        """Test default empty data dict."""
        event = Event(type=EventType.PIPELINE_STARTED)
        assert event.data == {}

    def test_default_source(self):
        """Test default source is pipeline."""
        event = Event(type=EventType.PIPELINE_STARTED)
        assert event.source == "pipeline"


class TestCreateEvent:
    """Tests for create_event helper function."""

    def test_basic_creation(self):
        """Test basic event creation."""
        event = create_event(EventType.PIPELINE_STARTED)

        assert event.type == EventType.PIPELINE_STARTED
        assert event.source == "pipeline"

    def test_with_source(self):
        """Test creating event with custom source."""
        event = create_event(
            EventType.AGENT_TOOL_CALL,
            source="generator",
        )

        assert event.source == "generator"

    def test_with_data_kwargs(self):
        """Test creating event with data as kwargs."""
        event = create_event(
            EventType.JUDGE_VERDICT,
            source="judge",
            verdict="accepted",
            score=0.85,
        )

        assert event.data["verdict"] == "accepted"
        assert event.data["score"] == 0.85

    def test_complex_data(self):
        """Test creating event with complex data."""
        event = create_event(
            EventType.OPTIMIZATION_COMPLETED,
            source="optimizer",
            layer=15,
            metrics={"loss": 0.1, "iterations": 50},
            vectors_trained=[10, 15, 20],
        )

        assert event.data["layer"] == 15
        assert event.data["metrics"]["loss"] == 0.1
        assert event.data["vectors_trained"] == [10, 15, 20]


class TestEventEmitter:
    """Tests for EventEmitter base class."""

    class ConcreteEmitter(EventEmitter):
        """Concrete implementation for testing."""
        pass

    def test_subscribe_and_emit(self):
        """Test subscribing and emitting events."""
        emitter = self.ConcreteEmitter()
        handler = MagicMock()

        emitter.on(EventType.PIPELINE_STARTED.value, handler)

        event = Event(type=EventType.PIPELINE_STARTED, data={"test": True})
        emitter.emit(event)

        handler.assert_called_once()
        call_arg = handler.call_args[0][0]
        assert call_arg.type == EventType.PIPELINE_STARTED
        assert call_arg.data["test"] is True

    def test_multiple_handlers(self):
        """Test multiple handlers for same event type."""
        emitter = self.ConcreteEmitter()
        handler1 = MagicMock()
        handler2 = MagicMock()

        emitter.on(EventType.PIPELINE_STARTED.value, handler1)
        emitter.on(EventType.PIPELINE_STARTED.value, handler2)

        event = Event(type=EventType.PIPELINE_STARTED)
        emitter.emit(event)

        handler1.assert_called_once()
        handler2.assert_called_once()

    def test_wildcard_handler(self):
        """Test wildcard handler receives all events."""
        emitter = self.ConcreteEmitter()
        handler = MagicMock()

        emitter.on("*", handler)

        emitter.emit(Event(type=EventType.PIPELINE_STARTED))
        emitter.emit(Event(type=EventType.PIPELINE_COMPLETED))
        emitter.emit(Event(type=EventType.ERROR))

        assert handler.call_count == 3

    def test_specific_and_wildcard(self):
        """Test both specific and wildcard handlers are called."""
        emitter = self.ConcreteEmitter()
        specific_handler = MagicMock()
        wildcard_handler = MagicMock()

        emitter.on(EventType.PIPELINE_STARTED.value, specific_handler)
        emitter.on("*", wildcard_handler)

        event = Event(type=EventType.PIPELINE_STARTED)
        emitter.emit(event)

        specific_handler.assert_called_once()
        wildcard_handler.assert_called_once()

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        emitter = self.ConcreteEmitter()
        handler = MagicMock()

        emitter.on(EventType.PIPELINE_STARTED.value, handler)
        emitter.off(EventType.PIPELINE_STARTED.value, handler)

        emitter.emit(Event(type=EventType.PIPELINE_STARTED))

        handler.assert_not_called()

    def test_unsubscribe_one_of_many(self):
        """Test unsubscribing one handler keeps others."""
        emitter = self.ConcreteEmitter()
        handler1 = MagicMock()
        handler2 = MagicMock()

        emitter.on(EventType.PIPELINE_STARTED.value, handler1)
        emitter.on(EventType.PIPELINE_STARTED.value, handler2)
        emitter.off(EventType.PIPELINE_STARTED.value, handler1)

        emitter.emit(Event(type=EventType.PIPELINE_STARTED))

        handler1.assert_not_called()
        handler2.assert_called_once()

    def test_no_handlers_no_error(self):
        """Test emitting event with no handlers doesn't raise."""
        emitter = self.ConcreteEmitter()
        event = Event(type=EventType.PIPELINE_STARTED)

        # Should not raise
        emitter.emit(event)

    def test_emit_sets_timestamp(self):
        """Test that emit sets timestamp if not set."""
        emitter = self.ConcreteEmitter()
        handler = MagicMock()

        emitter.on("*", handler)

        event = Event(type=EventType.PIPELINE_STARTED, timestamp=None)
        before = time.time()
        emitter.emit(event)
        after = time.time()

        call_arg = handler.call_args[0][0]
        assert call_arg.timestamp is not None
        # Allow small tolerance for timing issues
        assert before - 0.001 <= call_arg.timestamp <= after + 0.001

    def test_emit_preserves_existing_timestamp(self):
        """Test that emit preserves existing timestamp."""
        emitter = self.ConcreteEmitter()
        handler = MagicMock()

        emitter.on("*", handler)

        explicit_time = 1234567890.0
        event = Event(type=EventType.PIPELINE_STARTED, timestamp=explicit_time)
        emitter.emit(event)

        call_arg = handler.call_args[0][0]
        assert call_arg.timestamp == explicit_time

    def test_handler_receives_exact_event(self):
        """Test handler receives the exact event object."""
        emitter = self.ConcreteEmitter()
        received_events = []

        def handler(event: Event):
            received_events.append(event)

        emitter.on("*", handler)

        original_event = Event(
            type=EventType.PIPELINE_STARTED,
            data={"key": "value"},
            source="test",
        )
        emitter.emit(original_event)

        assert len(received_events) == 1
        # Same object (timestamp may be modified)
        assert received_events[0].type == original_event.type
        assert received_events[0].data == original_event.data
        assert received_events[0].source == original_event.source
