"""Tests for SessionStore and StorageManager."""

import tempfile
from pathlib import Path

import pytest
import torch

from vector_forge.storage import (
    StorageManager,
    SessionStore,
    SessionReplayer,
    LLMRequestEvent,
    LLMResponseEvent,
    ToolCallEvent,
    ToolResultEvent,
    DatapointAddedEvent,
    VectorCreatedEvent,
    VectorSelectedEvent,
    SessionStartedEvent,
    SessionCompletedEvent,
)


@pytest.fixture
def temp_storage_path():
    """Create a temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def storage_manager(temp_storage_path):
    """Create a storage manager with temp path."""
    return StorageManager(base_path=temp_storage_path)


class TestStorageManager:
    """Tests for StorageManager."""

    def test_create_session(self, storage_manager):
        """Test creating a new session."""
        store = storage_manager.create_session(
            behavior="test_behavior",
            config={"model": "test-model", "temperature": 0.7},
        )

        assert store.session_id is not None
        assert "test_behavior" in store.session_id
        assert store.session_path.exists()
        assert store.events_path.parent.exists()

    def test_list_sessions(self, storage_manager):
        """Test listing sessions."""
        # Create multiple sessions
        storage_manager.create_session("behavior_a", {})
        storage_manager.create_session("behavior_b", {})
        storage_manager.create_session("behavior_a", {})

        sessions = storage_manager.list_sessions()
        assert len(sessions) == 3

        # Filter by behavior
        sessions_a = storage_manager.list_sessions(behavior="behavior_a")
        assert len(sessions_a) == 2

    def test_get_session(self, storage_manager):
        """Test retrieving an existing session."""
        store1 = storage_manager.create_session("test", {"key": "value"})
        session_id = store1.session_id

        # Retrieve it
        store2 = storage_manager.get_session(session_id)
        assert store2.session_id == session_id
        assert store2.session_path.exists()

    def test_delete_session(self, storage_manager):
        """Test deleting a session."""
        store = storage_manager.create_session("test", {})
        session_id = store.session_id

        assert store.session_path.exists()

        result = storage_manager.delete_session(session_id)
        assert result is True
        assert not store.session_path.exists()

        # Deleting again should return False
        result = storage_manager.delete_session(session_id)
        assert result is False


class TestSessionStore:
    """Tests for SessionStore."""

    def test_append_event(self, storage_manager):
        """Test appending events."""
        store = storage_manager.create_session("test", {})

        event1 = LLMRequestEvent(
            request_id="req1",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        envelope1 = store.append_event(event1, source="test")
        assert envelope1.sequence == 1

        event2 = LLMResponseEvent(
            request_id="req1",
            content="Hi there!",
            latency_ms=150.0,
        )
        envelope2 = store.append_event(event2, source="test")
        assert envelope2.sequence == 2

    def test_iter_events(self, storage_manager):
        """Test iterating events."""
        store = storage_manager.create_session("test", {})

        # Add mixed events
        store.append_event(
            LLMRequestEvent(request_id="r1", model="m", messages=[]),
            source="llm",
        )
        store.append_event(
            ToolCallEvent(call_id="c1", tool_name="test", arguments={}),
            source="tools",
        )
        store.append_event(
            LLMResponseEvent(request_id="r1", content="ok", latency_ms=100),
            source="llm",
        )

        # Get all events
        all_events = list(store.iter_events())
        assert len(all_events) == 3

        # Filter by type
        llm_events = list(store.iter_events(event_types=["llm.request", "llm.response"]))
        assert len(llm_events) == 2

    def test_save_and_load_vector(self, storage_manager):
        """Test saving and loading vectors."""
        store = storage_manager.create_session("test", {})

        # Create a test vector
        vector = torch.randn(128)
        vector_ref = store.save_vector(vector, layer=16)

        assert "vectors/" in vector_ref
        assert "layer_16" in vector_ref

        # Load it back
        loaded = store.load_vector(vector_ref)
        assert torch.allclose(vector, loaded)

    def test_save_final_vector(self, storage_manager):
        """Test saving final vector."""
        store = storage_manager.create_session("test", {})

        vector = torch.randn(256)
        ref = store.save_final_vector(vector)

        assert ref == "vectors/final.pt"

        loaded = store.load_vector(ref)
        assert torch.allclose(vector, loaded)

    def test_metadata(self, storage_manager):
        """Test metadata operations."""
        store = storage_manager.create_session("test_behavior", {})

        metadata = store.get_metadata()
        assert metadata["behavior"] == "test_behavior"
        assert metadata["status"] == "running"

        store.update_metadata(custom_field="custom_value")
        metadata = store.get_metadata()
        assert metadata["custom_field"] == "custom_value"

    def test_finalize(self, storage_manager):
        """Test finalizing a session."""
        store = storage_manager.create_session("test", {})

        store.finalize(success=True)
        metadata = store.get_metadata()
        assert metadata["status"] == "completed"

        # Test failure case
        store2 = storage_manager.create_session("test2", {})
        store2.finalize(success=False, error="Test error")
        metadata2 = store2.get_metadata()
        assert metadata2["status"] == "failed"
        assert metadata2["error"] == "Test error"


class TestSessionReplayer:
    """Tests for SessionReplayer."""

    def test_reconstruct_state(self, storage_manager):
        """Test reconstructing state from events."""
        store = storage_manager.create_session("test", {})

        # Add datapoints
        store.append_event(
            DatapointAddedEvent(
                datapoint_id="dp_0",
                prompt="Test prompt",
                positive_completion="Good response",
                negative_completion="Bad response",
            ),
            source="state",
        )
        store.append_event(
            DatapointAddedEvent(
                datapoint_id="dp_1",
                prompt="Another prompt",
                positive_completion="Another good",
            ),
            source="state",
        )

        # Add a vector
        vector = torch.randn(64)
        vector_ref = store.save_vector(vector, layer=12)
        store.append_event(
            VectorCreatedEvent(
                vector_id="vec_1",
                layer=12,
                vector_ref=vector_ref,
                shape=[64],
                dtype="torch.float32",
                norm=vector.norm().item(),
            ),
            source="state",
        )

        # Select best
        store.append_event(
            VectorSelectedEvent(
                vector_id="vec_1",
                layer=12,
                strength=1.2,
                score=0.85,
            ),
            source="state",
        )

        # Reconstruct
        replayer = SessionReplayer(store)
        state = replayer.reconstruct_state()

        assert len(state.datapoints) == 2
        assert state.datapoints["dp_0"].prompt == "Test prompt"
        assert state.datapoints["dp_1"].positive_completion == "Another good"

        assert len(state.vectors) == 1
        assert 12 in state.vectors

        assert state.best_layer == 12
        assert state.best_strength == 1.2
        assert state.best_score == 0.85

    def test_iter_llm_calls(self, storage_manager):
        """Test iterating LLM request/response pairs."""
        store = storage_manager.create_session("test", {})

        # Add pairs
        store.append_event(
            LLMRequestEvent(
                request_id="r1",
                model="gpt-4o",
                messages=[{"role": "user", "content": "Q1"}],
            ),
            source="llm",
        )
        store.append_event(
            LLMResponseEvent(
                request_id="r1",
                content="A1",
                latency_ms=100,
            ),
            source="llm",
        )
        store.append_event(
            LLMRequestEvent(
                request_id="r2",
                model="gpt-4o",
                messages=[{"role": "user", "content": "Q2"}],
            ),
            source="llm",
        )
        store.append_event(
            LLMResponseEvent(
                request_id="r2",
                content="A2",
                latency_ms=150,
            ),
            source="llm",
        )

        replayer = SessionReplayer(store)
        pairs = list(replayer.iter_llm_calls())

        assert len(pairs) == 2
        assert pairs[0][0]["request_id"] == "r1"
        assert pairs[0][1]["content"] == "A1"
        assert pairs[1][0]["request_id"] == "r2"
        assert pairs[1][1]["content"] == "A2"

    def test_iter_tool_calls(self, storage_manager):
        """Test iterating tool call/result pairs."""
        store = storage_manager.create_session("test", {})

        store.append_event(
            ToolCallEvent(
                call_id="c1",
                tool_name="generate_prompts",
                arguments={"num": 5},
            ),
            source="tools",
        )
        store.append_event(
            ToolResultEvent(
                call_id="c1",
                success=True,
                output={"prompts": ["p1", "p2"]},
                latency_ms=500,
            ),
            source="tools",
        )

        replayer = SessionReplayer(store)
        pairs = list(replayer.iter_tool_calls())

        assert len(pairs) == 1
        assert pairs[0][0]["tool_name"] == "generate_prompts"
        assert pairs[0][1]["success"] is True

    def test_get_statistics(self, storage_manager):
        """Test getting session statistics."""
        store = storage_manager.create_session("test_behavior", {})

        # Add events
        store.append_event(
            LLMRequestEvent(request_id="r1", model="m", messages=[]),
            source="llm",
        )
        store.append_event(
            LLMResponseEvent(
                request_id="r1",
                content="ok",
                usage={"total_tokens": 100},
                latency_ms=100,
            ),
            source="llm",
        )
        store.append_event(
            ToolCallEvent(call_id="c1", tool_name="test", arguments={}),
            source="tools",
        )
        store.append_event(
            ToolResultEvent(call_id="c1", success=True, latency_ms=50),
            source="tools",
        )

        replayer = SessionReplayer(store)
        stats = replayer.get_statistics()

        assert stats["behavior"] == "test_behavior"
        assert stats["total_events"] == 4
        assert stats["llm_call_count"] == 1
        assert stats["tool_call_count"] == 1
        assert stats["total_tokens"] == 100


class TestEventSerialization:
    """Tests for event serialization/deserialization."""

    def test_session_events_roundtrip(self, storage_manager):
        """Test session events serialize and deserialize correctly."""
        store = storage_manager.create_session("test", {})

        # Session started
        store.append_event(
            SessionStartedEvent(
                behavior_name="test",
                behavior_description="A test behavior",
                config={"model": "gpt-4o"},
            ),
            source="pipeline",
        )

        # Session completed
        store.append_event(
            SessionCompletedEvent(
                success=True,
                final_vector_ref="vectors/final.pt",
                final_layer=16,
                final_score=0.9,
                total_llm_calls=50,
                total_tokens=10000,
                duration_seconds=120.5,
            ),
            source="pipeline",
        )

        events = list(store.iter_events())
        assert len(events) == 2

        # Check deserialization
        assert events[0].event_type == "session.started"
        assert events[0].payload["behavior_name"] == "test"

        assert events[1].event_type == "session.completed"
        assert events[1].payload["final_score"] == 0.9
