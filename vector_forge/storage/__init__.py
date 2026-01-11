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
    >>> store = manager.create_session("sycophancy", {"model": "claude-opus-4-5"})
    >>>
    >>> # Append events during extraction
    >>> store.append_event(LLMRequestEvent(...), source="generator")
    >>> store.append_event(LLMResponseEvent(...), source="generator")
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
    LLMChunkEvent,
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
    # Contrast pipeline events
    BehaviorAnalyzedEvent,
    ContrastPipelineStartedEvent,
    ContrastPipelineCompletedEvent,
    ContrastPairGeneratedEvent,
    ContrastPairValidatedEvent,
    # Seed events
    SeedGenerationStartedEvent,
    SeedGeneratedEvent,
    SeedGenerationCompletedEvent,
    SeedAssignedEvent,
    # Optimization events
    OptimizationStartedEvent,
    OptimizationProgressEvent,
    OptimizationCompletedEvent,
    AggregationCompletedEvent,
    RankingCompletedEvent,
)
from vector_forge.storage.replay import (
    SessionReplayer,
    ReplayedState,
    ReplayedDatapoint,
    ReplayedVector,
    ReplayedEvaluation,
    ReplayedContrastPair,
    ReplayedSeed,
    ReplayedOptimization,
    ReplayedLogEntry,
)
from vector_forge.storage.store import (
    SessionStore,
    StorageManager,
)
from vector_forge.storage.models import (
    Provider,
    ModelConfig,
    ModelConfigStore,
    ModelConfigManager,
    HFModelConfig,
    HFModelConfigStore,
    HFModelConfigManager,
    COMMON_MODELS,
    API_KEY_ENV_VARS,
)
from vector_forge.storage.preferences import (
    ModelRole,
    SelectedModels,
    Preferences,
    PreferencesManager,
)
from vector_forge.storage.emitter import (
    EventEmitter,
    NullEventEmitter,
    EventEmitterProtocol,
    generate_id,
)
from vector_forge.storage.log_builder import build_log_entry

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
    "ReplayedContrastPair",
    "ReplayedSeed",
    "ReplayedOptimization",
    "ReplayedLogEntry",
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
    "LLMChunkEvent",
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
    # Contrast pipeline events
    "BehaviorAnalyzedEvent",
    "ContrastPipelineStartedEvent",
    "ContrastPipelineCompletedEvent",
    "ContrastPairGeneratedEvent",
    "ContrastPairValidatedEvent",
    # Seed events
    "SeedGenerationStartedEvent",
    "SeedGeneratedEvent",
    "SeedGenerationCompletedEvent",
    "SeedAssignedEvent",
    # Optimization events
    "OptimizationStartedEvent",
    "OptimizationProgressEvent",
    "OptimizationCompletedEvent",
    "AggregationCompletedEvent",
    "RankingCompletedEvent",
    # Model configuration (API models for agents)
    "Provider",
    "ModelConfig",
    "ModelConfigStore",
    "ModelConfigManager",
    "COMMON_MODELS",
    "API_KEY_ENV_VARS",
    # HuggingFace model configuration (target models for steering)
    "HFModelConfig",
    "HFModelConfigStore",
    "HFModelConfigManager",
    # User preferences
    "ModelRole",
    "SelectedModels",
    "Preferences",
    "PreferencesManager",
    # Event emitter
    "EventEmitter",
    "NullEventEmitter",
    "EventEmitterProtocol",
    "generate_id",
    # Log building
    "build_log_entry",
]
