"""Tests for storage event types.

Verifies that storage event models:
- Have correct literal type discriminators
- Validate field types correctly
- Serialize and deserialize properly
- Work with the discriminated union pattern
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict
from pydantic import ValidationError

from vector_forge.storage.events import (
    # Categories
    EventCategory,
    # Session events
    SessionStartedEvent,
    SessionCompletedEvent,
    # LLM events
    LLMRequestEvent,
    LLMResponseEvent,
    # Tool events
    ToolCallEvent,
    ToolResultEvent,
    # Vector events
    VectorCreatedEvent,
    VectorComparisonEvent,
    VectorSelectedEvent,
    # Datapoint events
    DatapointAddedEvent,
    DatapointRemovedEvent,
    DatapointQualityEvent,
    # Evaluation events
    EvaluationStartedEvent,
    EvaluationOutputEvent,
    EvaluationCompletedEvent,
    # Checkpoint events
    CheckpointCreatedEvent,
    CheckpointRollbackEvent,
    # State events
    StateUpdateEvent,
    IterationStartedEvent,
    IterationCompletedEvent,
    # Envelope
    EventEnvelope,
    # Union type
    EventPayload,
)


# =============================================================================
# EventCategory Tests
# =============================================================================


class TestEventCategory:
    """Tests for EventCategory enum."""

    def test_all_categories_exist(self):
        """Test all expected categories exist."""
        assert EventCategory.SESSION == "session"
        assert EventCategory.LLM == "llm"
        assert EventCategory.TOOL == "tool"
        assert EventCategory.VECTOR == "vector"
        assert EventCategory.DATAPOINT == "datapoint"
        assert EventCategory.EVALUATION == "evaluation"
        assert EventCategory.CHECKPOINT == "checkpoint"
        assert EventCategory.STATE == "state"

    def test_categories_are_strings(self):
        """Test categories are string enums."""
        for cat in EventCategory:
            assert isinstance(cat.value, str)


# =============================================================================
# Session Event Tests
# =============================================================================


class TestSessionStartedEvent:
    """Tests for SessionStartedEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = SessionStartedEvent(
            behavior_name="sycophancy",
            behavior_description="Test description",
            config={"key": "value"}
        )
        assert event.event_type == "session.started"

    def test_required_fields(self):
        """Test required fields must be provided."""
        with pytest.raises(ValidationError):
            SessionStartedEvent()  # type: ignore

    def test_config_accepts_any_dict(self):
        """Test config accepts any dict structure."""
        event = SessionStartedEvent(
            behavior_name="test",
            behavior_description="desc",
            config={
                "nested": {"key": "value"},
                "list": [1, 2, 3],
                "number": 42
            }
        )
        assert event.config["number"] == 42


class TestSessionCompletedEvent:
    """Tests for SessionCompletedEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = SessionCompletedEvent(success=True)
        assert event.event_type == "session.completed"

    def test_success_required(self):
        """Test success is required."""
        with pytest.raises(ValidationError):
            SessionCompletedEvent()  # type: ignore

    def test_optional_fields_default_none_or_zero(self):
        """Test optional fields have correct defaults."""
        event = SessionCompletedEvent(success=True)
        assert event.final_vector_ref is None
        assert event.final_layer is None
        assert event.final_score is None
        assert event.total_llm_calls == 0
        assert event.total_tokens == 0
        assert event.duration_seconds == 0.0
        assert event.error is None

    def test_full_event(self):
        """Test event with all fields."""
        event = SessionCompletedEvent(
            success=True,
            final_vector_ref="vectors/final.pt",
            final_layer=16,
            final_score=0.85,
            total_llm_calls=150,
            total_tokens=50000,
            duration_seconds=300.5
        )
        assert event.final_layer == 16


# =============================================================================
# LLM Event Tests
# =============================================================================


class TestLLMRequestEvent:
    """Tests for LLMRequestEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = LLMRequestEvent(
            request_id="req_123",
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert event.event_type == "llm.request"

    def test_messages_accepts_list_of_dicts(self):
        """Test messages accepts list of message dicts."""
        event = LLMRequestEvent(
            request_id="req_123",
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ]
        )
        assert len(event.messages) == 2

    def test_optional_fields(self):
        """Test optional fields default correctly."""
        event = LLMRequestEvent(
            request_id="req_123",
            model="gpt-4",
            messages=[]
        )
        assert event.tools is None
        assert event.temperature is None
        assert event.max_tokens is None
        assert event.extra_params == {}


class TestLLMResponseEvent:
    """Tests for LLMResponseEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = LLMResponseEvent(request_id="req_123")
        assert event.event_type == "llm.response"

    def test_defaults(self):
        """Test default values."""
        event = LLMResponseEvent(request_id="req_123")
        assert event.content is None
        assert event.tool_calls == []
        assert event.finish_reason == "stop"
        assert event.usage is None
        assert event.latency_ms == 0.0
        assert event.error is None

    def test_with_tool_calls(self):
        """Test event with tool calls."""
        event = LLMResponseEvent(
            request_id="req_123",
            content=None,
            tool_calls=[
                {"id": "call_1", "name": "search", "arguments": "{}"}
            ],
            finish_reason="tool_calls"
        )
        assert len(event.tool_calls) == 1


# =============================================================================
# Tool Event Tests
# =============================================================================


class TestToolCallEvent:
    """Tests for ToolCallEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = ToolCallEvent(
            call_id="call_123",
            tool_name="search",
            arguments={"query": "test"}
        )
        assert event.event_type == "tool.call"

    def test_agent_id_default(self):
        """Test agent_id defaults to 'extractor'."""
        event = ToolCallEvent(
            call_id="call_123",
            tool_name="search",
            arguments={}
        )
        assert event.agent_id == "extractor"


class TestToolResultEvent:
    """Tests for ToolResultEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = ToolResultEvent(
            call_id="call_123",
            success=True
        )
        assert event.event_type == "tool.result"

    def test_output_accepts_any(self):
        """Test output accepts any type."""
        event1 = ToolResultEvent(call_id="1", success=True, output="string")
        event2 = ToolResultEvent(call_id="2", success=True, output={"key": "value"})
        event3 = ToolResultEvent(call_id="3", success=True, output=[1, 2, 3])

        assert event1.output == "string"
        assert event2.output["key"] == "value"
        assert event3.output == [1, 2, 3]


# =============================================================================
# Vector Event Tests
# =============================================================================


class TestVectorCreatedEvent:
    """Tests for VectorCreatedEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = VectorCreatedEvent(
            vector_id="vec_123",
            layer=16,
            vector_ref="vectors/layer_16.pt",
            shape=[4096],
            dtype="float32",
            norm=1.0
        )
        assert event.event_type == "vector.created"

    def test_shape_accepts_list_of_ints(self):
        """Test shape accepts list of integers."""
        event = VectorCreatedEvent(
            vector_id="vec_123",
            layer=16,
            vector_ref="test.pt",
            shape=[32, 4096],
            dtype="float16",
            norm=0.5
        )
        assert event.shape == [32, 4096]


class TestVectorSelectedEvent:
    """Tests for VectorSelectedEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = VectorSelectedEvent(
            vector_id="vec_123",
            layer=16,
            strength=1.0
        )
        assert event.event_type == "vector.selected"

    def test_optional_fields(self):
        """Test optional fields."""
        event = VectorSelectedEvent(
            vector_id="vec_123",
            layer=16,
            strength=1.5,
            score=0.85,
            reason="Best score"
        )
        assert event.score == 0.85
        assert event.reason == "Best score"


# =============================================================================
# Datapoint Event Tests
# =============================================================================


class TestDatapointAddedEvent:
    """Tests for DatapointAddedEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = DatapointAddedEvent(
            datapoint_id="dp_123",
            prompt="Test prompt",
            positive_completion="Positive response"
        )
        assert event.event_type == "datapoint.added"

    def test_optional_negative_completion(self):
        """Test negative_completion is optional."""
        event = DatapointAddedEvent(
            datapoint_id="dp_123",
            prompt="Test prompt",
            positive_completion="Positive"
        )
        assert event.negative_completion is None


class TestDatapointQualityEvent:
    """Tests for DatapointQualityEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = DatapointQualityEvent(datapoint_id="dp_123")
        assert event.event_type == "datapoint.quality"

    def test_defaults(self):
        """Test default values."""
        event = DatapointQualityEvent(datapoint_id="dp_123")
        assert event.leave_one_out_influence is None
        assert event.gradient_alignment == 0.0
        assert event.avg_loss_contribution == 0.0
        assert event.quality_score == 0.0
        assert event.recommendation == "KEEP"
        assert event.is_outlier is False


# =============================================================================
# Evaluation Event Tests
# =============================================================================


class TestEvaluationStartedEvent:
    """Tests for EvaluationStartedEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = EvaluationStartedEvent(
            evaluation_id="eval_123",
            eval_type="quick",
            vector_id="vec_123",
            layer=16
        )
        assert event.event_type == "evaluation.started"


class TestEvaluationCompletedEvent:
    """Tests for EvaluationCompletedEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = EvaluationCompletedEvent(evaluation_id="eval_123")
        assert event.event_type == "evaluation.completed"

    def test_scores_accepts_dict(self):
        """Test scores accepts dict of floats."""
        event = EvaluationCompletedEvent(
            evaluation_id="eval_123",
            scores={
                "behavior_strength": 0.8,
                "coherence": 0.9,
                "specificity": 0.7
            }
        )
        assert event.scores["behavior_strength"] == 0.8


# =============================================================================
# Checkpoint Event Tests
# =============================================================================


class TestCheckpointCreatedEvent:
    """Tests for CheckpointCreatedEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = CheckpointCreatedEvent(
            checkpoint_id="cp_123",
            description="Before optimization"
        )
        assert event.event_type == "checkpoint.created"


class TestCheckpointRollbackEvent:
    """Tests for CheckpointRollbackEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = CheckpointRollbackEvent(checkpoint_id="cp_123")
        assert event.event_type == "checkpoint.rollback"


# =============================================================================
# State Event Tests
# =============================================================================


class TestStateUpdateEvent:
    """Tests for StateUpdateEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = StateUpdateEvent(field="best_score")
        assert event.event_type == "state.update"

    def test_values_accept_any(self):
        """Test old_value and new_value accept any type."""
        event = StateUpdateEvent(
            field="config",
            old_value={"key": "old"},
            new_value={"key": "new"}
        )
        assert event.old_value["key"] == "old"


class TestIterationStartedEvent:
    """Tests for IterationStartedEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = IterationStartedEvent(
            iteration_type="outer",
            iteration=1
        )
        assert event.event_type == "state.iteration_started"


class TestIterationCompletedEvent:
    """Tests for IterationCompletedEvent type."""

    def test_event_type_literal(self):
        """Test event_type is correct literal."""
        event = IterationCompletedEvent(
            iteration_type="inner",
            iteration=3
        )
        assert event.event_type == "state.iteration_completed"


# =============================================================================
# EventEnvelope Tests
# =============================================================================


class TestEventEnvelope:
    """Tests for EventEnvelope type."""

    def test_required_fields(self):
        """Test required fields."""
        envelope = EventEnvelope(
            session_id="sess_123",
            sequence=1,
            category=EventCategory.SESSION,
            event_type="session.started",
            source="test",
            payload={"key": "value"}
        )
        assert envelope.session_id == "sess_123"
        assert envelope.sequence == 1

    def test_auto_generated_fields(self):
        """Test auto-generated event_id and timestamp."""
        envelope = EventEnvelope(
            session_id="sess_123",
            sequence=1,
            category=EventCategory.SESSION,
            event_type="session.started",
            source="test",
            payload={}
        )
        assert envelope.event_id is not None
        assert len(envelope.event_id) == 36  # UUID format
        assert envelope.timestamp is not None

    def test_timestamp_type(self):
        """Test timestamp is datetime."""
        envelope = EventEnvelope(
            session_id="sess_123",
            sequence=1,
            category=EventCategory.SESSION,
            event_type="session.started",
            source="test",
            payload={}
        )
        assert isinstance(envelope.timestamp, datetime)

    def test_category_accepts_enum(self):
        """Test category accepts EventCategory enum."""
        for category in EventCategory:
            envelope = EventEnvelope(
                session_id="sess_123",
                sequence=1,
                category=category,
                event_type="test",
                source="test",
                payload={}
            )
            assert envelope.category == category


# =============================================================================
# Event Serialization Tests
# =============================================================================


class TestEventSerialization:
    """Tests for event serialization/deserialization."""

    def test_event_to_json_and_back(self):
        """Test event can be serialized and deserialized."""
        event = SessionStartedEvent(
            behavior_name="sycophancy",
            behavior_description="Test",
            config={"model": "gpt-4"}
        )

        # Serialize
        json_str = event.model_dump_json()
        assert isinstance(json_str, str)

        # Deserialize
        restored = SessionStartedEvent.model_validate_json(json_str)
        assert restored.behavior_name == event.behavior_name
        assert restored.event_type == event.event_type

    def test_envelope_to_json_and_back(self):
        """Test envelope can be serialized and deserialized."""
        envelope = EventEnvelope(
            session_id="sess_123",
            sequence=42,
            category=EventCategory.LLM,
            event_type="llm.request",
            source="extractor",
            payload={"model": "gpt-4", "messages": []}
        )

        # Serialize
        json_str = envelope.model_dump_json()
        assert isinstance(json_str, str)

        # Deserialize
        restored = EventEnvelope.model_validate_json(json_str)
        assert restored.session_id == envelope.session_id
        assert restored.sequence == 42
        assert restored.category == EventCategory.LLM

    def test_nested_payload_serialization(self):
        """Test deeply nested payloads serialize correctly."""
        event = LLMRequestEvent(
            request_id="req_123",
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                    "metadata": {
                        "nested": {"deep": {"value": 42}}
                    }
                }
            ],
            extra_params={"top_p": 0.9, "presence_penalty": 0.1}
        )

        json_str = event.model_dump_json()
        restored = LLMRequestEvent.model_validate_json(json_str)

        assert restored.messages[0]["metadata"]["nested"]["deep"]["value"] == 42


# =============================================================================
# Discriminated Union Tests
# =============================================================================


class TestEventPayloadUnion:
    """Tests for EventPayload discriminated union."""

    def test_session_started_discriminator(self):
        """Test SessionStartedEvent is correctly discriminated."""
        data = {
            "event_type": "session.started",
            "behavior_name": "test",
            "behavior_description": "desc",
            "config": {}
        }
        # This should work via the Pydantic discriminated union
        from pydantic import TypeAdapter
        adapter = TypeAdapter(EventPayload)
        event = adapter.validate_python(data)
        assert isinstance(event, SessionStartedEvent)

    def test_llm_request_discriminator(self):
        """Test LLMRequestEvent is correctly discriminated."""
        data = {
            "event_type": "llm.request",
            "request_id": "req_123",
            "model": "gpt-4",
            "messages": []
        }
        from pydantic import TypeAdapter
        adapter = TypeAdapter(EventPayload)
        event = adapter.validate_python(data)
        assert isinstance(event, LLMRequestEvent)

    def test_vector_created_discriminator(self):
        """Test VectorCreatedEvent is correctly discriminated."""
        data = {
            "event_type": "vector.created",
            "vector_id": "vec_123",
            "layer": 16,
            "vector_ref": "test.pt",
            "shape": [4096],
            "dtype": "float32",
            "norm": 1.0
        }
        from pydantic import TypeAdapter
        adapter = TypeAdapter(EventPayload)
        event = adapter.validate_python(data)
        assert isinstance(event, VectorCreatedEvent)

    def test_invalid_event_type_raises(self):
        """Test invalid event_type raises validation error."""
        data = {
            "event_type": "invalid.event",
            "some_field": "value"
        }
        from pydantic import TypeAdapter
        adapter = TypeAdapter(EventPayload)
        with pytest.raises(ValidationError):
            adapter.validate_python(data)
