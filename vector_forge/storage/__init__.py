"""Event sourcing storage system for Vector Forge.

This module provides complete session recording and replay capabilities
for steering vector extraction runs. Every LLM call, tool execution,
vector creation, and state change is captured as immutable events.

Key Components:
- SessionStore: Manages events and vectors for a single session
- StorageManager: Creates and manages multiple sessions
- SessionReplayer: Reconstructs state from event log

Example:
    >>> from vector_forge.storage import StorageManager, SessionReplayer
    >>>
    >>> # Create a new session
    >>> manager = StorageManager()
    >>> store = manager.create_session("sycophancy", {"model": "gpt-4o"})
    >>>
    >>> # Append events during extraction
    >>> store.append_event(LLMRequestEvent(...), source="extractor")
    >>> store.append_event(LLMResponseEvent(...), source="extractor")
    >>>
    >>> # Later, replay the session
    >>> replayer = SessionReplayer(store)
    >>> state = replayer.reconstruct_state()
    >>> stats = replayer.get_statistics()
"""

from vector_forge.storage.config import StorageConfig
from vector_forge.storage.events import (
    # Enums
    EventCategory,
    # Envelope
    EventEnvelope,
    EventPayload,
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
)
from vector_forge.storage.replay import (
    SessionReplayer,
    ReplayedState,
    ReplayedDatapoint,
    ReplayedVector,
    ReplayedEvaluation,
)
from vector_forge.storage.store import (
    SessionStore,
    StorageManager,
)

__all__ = [
    # Config
    "StorageConfig",
    # Core classes
    "SessionStore",
    "StorageManager",
    "SessionReplayer",
    # Replay data structures
    "ReplayedState",
    "ReplayedDatapoint",
    "ReplayedVector",
    "ReplayedEvaluation",
    # Event envelope
    "EventCategory",
    "EventEnvelope",
    "EventPayload",
    # Session events
    "SessionStartedEvent",
    "SessionCompletedEvent",
    # LLM events
    "LLMRequestEvent",
    "LLMResponseEvent",
    # Tool events
    "ToolCallEvent",
    "ToolResultEvent",
    # Vector events
    "VectorCreatedEvent",
    "VectorComparisonEvent",
    "VectorSelectedEvent",
    # Datapoint events
    "DatapointAddedEvent",
    "DatapointRemovedEvent",
    "DatapointQualityEvent",
    # Evaluation events
    "EvaluationStartedEvent",
    "EvaluationOutputEvent",
    "EvaluationCompletedEvent",
    # Checkpoint events
    "CheckpointCreatedEvent",
    "CheckpointRollbackEvent",
    # State events
    "StateUpdateEvent",
    "IterationStartedEvent",
    "IterationCompletedEvent",
]
