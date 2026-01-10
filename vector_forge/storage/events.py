"""Event types for the event sourcing storage system.

Defines all event types as Pydantic models with discriminated unions
for type-safe serialization and deserialization.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
import uuid

from pydantic import BaseModel, Field


class EventCategory(str, Enum):
    """High-level event categories."""

    SESSION = "session"
    LLM = "llm"
    TOOL = "tool"
    VECTOR = "vector"
    DATAPOINT = "datapoint"
    EVALUATION = "evaluation"
    CHECKPOINT = "checkpoint"
    STATE = "state"


# =============================================================================
# Session Events
# =============================================================================


class SessionStartedEvent(BaseModel):
    """Session initialization event."""

    event_type: Literal["session.started"] = "session.started"
    behavior_name: str
    behavior_description: str
    config: Dict[str, Any]


class SessionCompletedEvent(BaseModel):
    """Session finished event."""

    event_type: Literal["session.completed"] = "session.completed"
    success: bool
    final_vector_ref: Optional[str] = None
    final_layer: Optional[int] = None
    final_score: Optional[float] = None
    total_llm_calls: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None


# =============================================================================
# LLM Events
# =============================================================================


class LLMRequestEvent(BaseModel):
    """Captures full LLM API request."""

    event_type: Literal["llm.request"] = "llm.request"
    request_id: str
    model: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class LLMResponseEvent(BaseModel):
    """Captures full LLM API response with timing."""

    event_type: Literal["llm.response"] = "llm.response"
    request_id: str
    content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: Optional[Dict[str, int]] = None
    latency_ms: float = 0.0
    error: Optional[str] = None


# =============================================================================
# Tool Events
# =============================================================================


class ToolCallEvent(BaseModel):
    """Tool invocation started."""

    event_type: Literal["tool.call"] = "tool.call"
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    agent_id: str = "extractor"


class ToolResultEvent(BaseModel):
    """Tool execution completed."""

    event_type: Literal["tool.result"] = "tool.result"
    call_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0


# =============================================================================
# Vector Events
# =============================================================================


class VectorCreatedEvent(BaseModel):
    """Steering vector created/optimized."""

    event_type: Literal["vector.created"] = "vector.created"
    vector_id: str
    layer: int
    vector_ref: str  # Path to .pt file relative to session dir
    shape: List[int]
    dtype: str
    norm: float
    optimization_metrics: Dict[str, Any] = Field(default_factory=dict)


class VectorComparisonEvent(BaseModel):
    """Vectors compared."""

    event_type: Literal["vector.comparison"] = "vector.comparison"
    vector_ids: List[str]
    similarities: Dict[str, float] = Field(default_factory=dict)


class VectorSelectedEvent(BaseModel):
    """Best vector selected."""

    event_type: Literal["vector.selected"] = "vector.selected"
    vector_id: str
    layer: int
    strength: float
    score: Optional[float] = None
    reason: str = ""


# =============================================================================
# Datapoint Events
# =============================================================================


class DatapointAddedEvent(BaseModel):
    """Training datapoint added."""

    event_type: Literal["datapoint.added"] = "datapoint.added"
    datapoint_id: str
    prompt: str
    positive_completion: str
    negative_completion: Optional[str] = None
    domain: Optional[str] = None
    format_type: Optional[str] = None


class DatapointRemovedEvent(BaseModel):
    """Datapoint removed."""

    event_type: Literal["datapoint.removed"] = "datapoint.removed"
    datapoint_id: str
    reason: str = ""


class DatapointQualityEvent(BaseModel):
    """Quality assessment for datapoint."""

    event_type: Literal["datapoint.quality"] = "datapoint.quality"
    datapoint_id: str
    leave_one_out_influence: Optional[float] = None
    gradient_alignment: float = 0.0
    avg_loss_contribution: float = 0.0
    quality_score: float = 0.0
    recommendation: str = "KEEP"  # KEEP, REVIEW, REMOVE
    is_outlier: bool = False


# =============================================================================
# Evaluation Events
# =============================================================================


class EvaluationStartedEvent(BaseModel):
    """Evaluation session started."""

    event_type: Literal["evaluation.started"] = "evaluation.started"
    evaluation_id: str
    eval_type: str  # "quick" or "thorough"
    vector_id: str
    layer: int
    strength_levels: List[float] = Field(default_factory=list)
    num_prompts: int = 0


class EvaluationOutputEvent(BaseModel):
    """Single evaluation output captured."""

    event_type: Literal["evaluation.output"] = "evaluation.output"
    evaluation_id: str
    prompt: str
    output: str
    strength: Optional[float] = None
    is_baseline: bool = False


class EvaluationCompletedEvent(BaseModel):
    """Evaluation finished with scores."""

    event_type: Literal["evaluation.completed"] = "evaluation.completed"
    evaluation_id: str
    scores: Dict[str, float] = Field(default_factory=dict)
    citations: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    verdict: str = "needs_refinement"
    recommended_strength: float = 1.0
    raw_judge_output: Optional[str] = None


# =============================================================================
# Checkpoint Events
# =============================================================================


class CheckpointCreatedEvent(BaseModel):
    """State checkpoint created."""

    event_type: Literal["checkpoint.created"] = "checkpoint.created"
    checkpoint_id: str
    description: str
    state_ref: Optional[str] = None  # Reference to state snapshot file


class CheckpointRollbackEvent(BaseModel):
    """Rollback to checkpoint."""

    event_type: Literal["checkpoint.rollback"] = "checkpoint.rollback"
    checkpoint_id: str
    previous_sequence: int = 0


# =============================================================================
# State Events (generic state changes)
# =============================================================================


class StateUpdateEvent(BaseModel):
    """Generic state update event."""

    event_type: Literal["state.update"] = "state.update"
    field: str
    old_value: Any = None
    new_value: Any = None


class IterationStartedEvent(BaseModel):
    """Iteration started (outer or inner)."""

    event_type: Literal["state.iteration_started"] = "state.iteration_started"
    iteration_type: str  # "outer" or "inner"
    iteration: int


class IterationCompletedEvent(BaseModel):
    """Iteration completed."""

    event_type: Literal["state.iteration_completed"] = "state.iteration_completed"
    iteration_type: str
    iteration: int
    result: Optional[Dict[str, Any]] = None


# =============================================================================
# Discriminated Union Type
# =============================================================================

EventPayload = Annotated[
    Union[
        # Session
        SessionStartedEvent,
        SessionCompletedEvent,
        # LLM
        LLMRequestEvent,
        LLMResponseEvent,
        # Tool
        ToolCallEvent,
        ToolResultEvent,
        # Vector
        VectorCreatedEvent,
        VectorComparisonEvent,
        VectorSelectedEvent,
        # Datapoint
        DatapointAddedEvent,
        DatapointRemovedEvent,
        DatapointQualityEvent,
        # Evaluation
        EvaluationStartedEvent,
        EvaluationOutputEvent,
        EvaluationCompletedEvent,
        # Checkpoint
        CheckpointCreatedEvent,
        CheckpointRollbackEvent,
        # State
        StateUpdateEvent,
        IterationStartedEvent,
        IterationCompletedEvent,
    ],
    Field(discriminator="event_type"),
]


# =============================================================================
# Event Envelope
# =============================================================================


class EventEnvelope(BaseModel):
    """Immutable event envelope for JSONL storage.

    Every event is wrapped in this envelope which provides:
    - Unique event ID
    - Session association
    - Timestamp
    - Monotonic sequence number
    - Category for filtering
    - Source component identification
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence: int
    category: EventCategory
    event_type: str
    source: str
    payload: Dict[str, Any]

    @classmethod
    def create(
        cls,
        session_id: str,
        sequence: int,
        payload: EventPayload,
        source: str,
    ) -> "EventEnvelope":
        """Create an envelope from a typed payload."""
        event_type = payload.event_type
        category = cls._get_category(event_type)

        return cls(
            session_id=session_id,
            sequence=sequence,
            category=category,
            event_type=event_type,
            source=source,
            payload=payload.model_dump(),
        )

    @staticmethod
    def _get_category(event_type: str) -> EventCategory:
        """Determine category from event type prefix."""
        prefix = event_type.split(".")[0]
        try:
            return EventCategory(prefix)
        except ValueError:
            return EventCategory.STATE

    def get_typed_payload(self) -> EventPayload:
        """Reconstruct the typed payload from dict."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(EventPayload)
        return adapter.validate_python(self.payload)
