"""Event types for the event system."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


class EventType(str, Enum):
    """Event types emitted during extraction."""

    # Pipeline lifecycle
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"

    # Iteration events
    OUTER_ITERATION_STARTED = "iteration.outer.started"
    OUTER_ITERATION_COMPLETED = "iteration.outer.completed"
    INNER_ITERATION_STARTED = "iteration.inner.started"
    INNER_ITERATION_COMPLETED = "iteration.inner.completed"

    # Datapoint events
    DATAPOINT_GENERATION_STARTED = "datapoint.generation.started"
    DATAPOINT_GENERATION_PROGRESS = "datapoint.generation.progress"
    DATAPOINT_GENERATION_COMPLETED = "datapoint.generation.completed"
    DATAPOINT_QUALITY_ANALYZED = "datapoint.quality.analyzed"
    DATAPOINT_REMOVED = "datapoint.removed"

    # Diversity events
    DIVERSITY_CHECK_STARTED = "diversity.check.started"
    DIVERSITY_CHECK_COMPLETED = "diversity.check.completed"
    DIVERSITY_REGENERATION_NEEDED = "diversity.regeneration.needed"

    # Optimization events
    OPTIMIZATION_STARTED = "optimization.started"
    OPTIMIZATION_STEP = "optimization.step"
    OPTIMIZATION_COMPLETED = "optimization.completed"
    LAYER_SEARCH_STARTED = "layer_search.started"
    LAYER_SEARCH_PROGRESS = "layer_search.progress"
    LAYER_SEARCH_COMPLETED = "layer_search.completed"

    # Evaluation events
    QUICK_EVAL_STARTED = "evaluation.quick.started"
    QUICK_EVAL_COMPLETED = "evaluation.quick.completed"
    THOROUGH_EVAL_STARTED = "evaluation.thorough.started"
    THOROUGH_EVAL_PROGRESS = "evaluation.thorough.progress"
    THOROUGH_EVAL_COMPLETED = "evaluation.thorough.completed"

    # Judge events
    JUDGE_STARTED = "judge.started"
    JUDGE_COMPLETED = "judge.completed"
    JUDGE_VERDICT = "judge.verdict"

    # Noise reduction events
    NOISE_REDUCTION_STARTED = "noise_reduction.started"
    NOISE_REDUCTION_COMPLETED = "noise_reduction.completed"

    # Checkpoint events
    CHECKPOINT_CREATED = "checkpoint.created"
    ROLLBACK_PERFORMED = "rollback.performed"

    # Agent events
    AGENT_THINKING = "agent.thinking"
    AGENT_TOOL_CALL = "agent.tool_call"
    AGENT_TOOL_RESULT = "agent.tool_result"

    # Error events
    ERROR = "error"
    WARNING = "warning"


@dataclass
class Event:
    """
    Event emitted during pipeline execution.

    Events are used for UI integration - UIs subscribe to events
    to display progress, logs, and results in real-time.
    """

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    source: str = "pipeline"  # Component that emitted the event

    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()


def create_event(
    event_type: EventType,
    source: str = "pipeline",
    **data: Any,
) -> Event:
    """
    Helper to create an event.

    Args:
        event_type: Type of event.
        source: Component emitting the event.
        **data: Event data as keyword arguments.

    Returns:
        Event instance.
    """
    return Event(type=event_type, data=data, source=source)
