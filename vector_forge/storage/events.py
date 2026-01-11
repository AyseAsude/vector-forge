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
    # New categories for complete event sourcing
    CONTRAST = "contrast"
    SEED = "seed"
    OPTIMIZATION = "optimization"


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
    # Usage can contain nested dicts (token details) and floats (cost) from providers
    usage: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    error: Optional[str] = None


class LLMChunkEvent(BaseModel):
    """Captures streaming LLM chunk for real-time display."""

    event_type: Literal["llm.chunk"] = "llm.chunk"
    request_id: str
    chunk: str  # The token(s) in this chunk
    chunk_index: int  # Position in stream (0, 1, 2, ...)
    accumulated: str  # Full content so far (for UI convenience)


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
    eval_type: str  # "quick" or "comprehensive"
    vector_id: str
    layer: int
    sample_idx: Optional[int] = None  # Sample index for UI tracking
    strength_levels: List[float] = Field(default_factory=list)
    num_prompts: int = 0
    dimensions: List[str] = Field(default_factory=list)  # e.g., ["behavior", "specificity", ...]


class EvaluationDimensionStartedEvent(BaseModel):
    """Individual evaluation dimension started (behavior, specificity, etc.)."""

    event_type: Literal["evaluation.dimension_started"] = "evaluation.dimension_started"
    evaluation_id: str
    dimension: str  # "behavior", "specificity", "coherence", "capability", "generalization"
    num_prompts: int
    num_generations: int  # Total generations for this dimension


class EvaluationGenerationEvent(BaseModel):
    """Single model generation during evaluation."""

    event_type: Literal["evaluation.generation"] = "evaluation.generation"
    evaluation_id: str
    dimension: str
    prompt: str
    output: str
    strength: float
    generation_index: int
    is_baseline: bool = False


class EvaluationJudgeCallEvent(BaseModel):
    """Judge LLM call during evaluation."""

    event_type: Literal["evaluation.judge_call"] = "evaluation.judge_call"
    evaluation_id: str
    dimension: str
    prompt: str  # The prompt being judged (or batch description)
    num_outputs: int  # How many outputs being judged in this call
    scores: List[float]  # Scores returned
    latency_ms: float = 0.0


class EvaluationDimensionCompletedEvent(BaseModel):
    """Individual evaluation dimension completed with score."""

    event_type: Literal["evaluation.dimension_completed"] = "evaluation.dimension_completed"
    evaluation_id: str
    dimension: str
    score: float
    max_score: float = 10.0
    details: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = 0.0


class EvaluationProgressEvent(BaseModel):
    """Evaluation progress update."""

    event_type: Literal["evaluation.progress"] = "evaluation.progress"
    evaluation_id: str
    phase: str  # "generating", "judging"
    completed: int
    total: int
    current_dimension: Optional[str] = None


class EvaluationOutputEvent(BaseModel):
    """Single evaluation output captured (legacy, use EvaluationGenerationEvent)."""

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
    sample_idx: Optional[int] = None  # Sample index for UI tracking
    scores: Dict[str, float] = Field(default_factory=dict)
    dimension_scores: Dict[str, float] = Field(default_factory=dict)  # Per-dimension breakdown
    citations: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    verdict: str = "needs_refinement"
    recommended_strength: float = 1.0
    raw_judge_output: Optional[str] = None
    duration_seconds: float = 0.0
    total_generations: int = 0
    total_judge_calls: int = 0


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
# Contrast Pipeline Events
# =============================================================================


class BehaviorAnalyzedEvent(BaseModel):
    """Behavior analysis completed."""

    event_type: Literal["contrast.behavior_analyzed"] = "contrast.behavior_analyzed"
    behavior_name: str
    num_components: int
    components: List[Dict[str, Any]]
    trigger_conditions: List[str]
    contrast_dimensions: List[str]


class ContrastPipelineStartedEvent(BaseModel):
    """Contrast pipeline started."""

    event_type: Literal["contrast.pipeline_started"] = "contrast.pipeline_started"
    behavior_description: str
    num_samples: int
    config: Dict[str, Any]


class ContrastPipelineCompletedEvent(BaseModel):
    """Contrast pipeline completed."""

    event_type: Literal["contrast.pipeline_completed"] = "contrast.pipeline_completed"
    num_samples: int
    total_pairs_generated: int
    total_valid_pairs: int
    avg_quality: float
    duration_seconds: float


class ContrastPairGeneratedEvent(BaseModel):
    """Single contrast pair generated."""

    event_type: Literal["contrast.pair_generated"] = "contrast.pair_generated"
    pair_id: str
    seed_id: str
    prompt: str
    dst_response: str
    src_response: str
    sample_idx: int


class ContrastPairValidatedEvent(BaseModel):
    """Contrast pair validation completed."""

    event_type: Literal["contrast.pair_validated"] = "contrast.pair_validated"
    pair_id: str
    is_valid: bool
    contrast_quality: float
    # Dimension-specific scores (-1 = not evaluated)
    dimension_score: float = -1.0
    marker_score: float = -1.0
    boundary_score: float = -1.0
    intensity_score: float = -1.0
    structural_score: float = -1.0
    semantic_score: float = -1.0
    # Additional info
    weakest_dimension: Optional[str] = None
    rejection_reason: Optional[str] = None


# =============================================================================
# Seed Events
# =============================================================================


class SeedGenerationStartedEvent(BaseModel):
    """Seed generation started."""

    event_type: Literal["seed.generation_started"] = "seed.generation_started"
    num_seeds_requested: int
    behavior_name: str


class SeedGeneratedEvent(BaseModel):
    """Single seed generated."""

    event_type: Literal["seed.generated"] = "seed.generated"
    seed_id: str
    scenario: str
    context: str
    quality_score: float
    is_core: bool


class SeedGenerationCompletedEvent(BaseModel):
    """Seed generation batch completed."""

    event_type: Literal["seed.generation_completed"] = "seed.generation_completed"
    total_generated: int
    avg_quality: float


class SeedAssignedEvent(BaseModel):
    """Seeds assigned to samples."""

    event_type: Literal["seed.assigned"] = "seed.assigned"
    sample_idx: int
    num_core_seeds: int
    num_unique_seeds: int
    seed_ids: List[str]


# =============================================================================
# Optimization Events
# =============================================================================


class OptimizationStartedEvent(BaseModel):
    """Steering vector optimization started."""

    event_type: Literal["optimization.started"] = "optimization.started"
    sample_idx: int
    layer: int
    num_datapoints: int
    config: Dict[str, Any]


class OptimizationProgressEvent(BaseModel):
    """Optimization iteration progress."""

    event_type: Literal["optimization.progress"] = "optimization.progress"
    sample_idx: int
    iteration: int
    loss: float
    norm: float


class OptimizationCompletedEvent(BaseModel):
    """Steering vector optimization completed."""

    event_type: Literal["optimization.completed"] = "optimization.completed"
    sample_idx: int
    layer: int
    final_loss: float
    iterations: int
    loss_history: List[float]
    datapoints_used: int
    duration_seconds: float
    success: bool
    error: Optional[str] = None


class AggregationCompletedEvent(BaseModel):
    """Vector aggregation completed."""

    event_type: Literal["optimization.aggregation_completed"] = "optimization.aggregation_completed"
    strategy: str
    num_vectors: int
    top_k: int
    ensemble_components: List[str]
    final_score: float
    final_layer: int


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
        EvaluationDimensionStartedEvent,
        EvaluationGenerationEvent,
        EvaluationJudgeCallEvent,
        EvaluationDimensionCompletedEvent,
        EvaluationProgressEvent,
        EvaluationOutputEvent,
        EvaluationCompletedEvent,
        # Checkpoint
        CheckpointCreatedEvent,
        CheckpointRollbackEvent,
        # State
        StateUpdateEvent,
        IterationStartedEvent,
        IterationCompletedEvent,
        # Contrast Pipeline
        BehaviorAnalyzedEvent,
        ContrastPipelineStartedEvent,
        ContrastPipelineCompletedEvent,
        ContrastPairGeneratedEvent,
        ContrastPairValidatedEvent,
        # Seed
        SeedGenerationStartedEvent,
        SeedGeneratedEvent,
        SeedGenerationCompletedEvent,
        SeedAssignedEvent,
        # Optimization
        OptimizationStartedEvent,
        OptimizationProgressEvent,
        OptimizationCompletedEvent,
        AggregationCompletedEvent,
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
