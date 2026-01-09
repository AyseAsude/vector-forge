"""Core types, protocols, and configuration for Vector Forge."""

from vector_forge.core.protocols import (
    LLMClient,
    Tool,
    ToolResult,
    DatapointStrategy,
    NoiseReducer,
    LayerSearchStrategy,
    VectorEvaluator,
    EventEmitter,
    EventHandler,
)
from vector_forge.core.behavior import BehaviorSpec
from vector_forge.core.config import (
    PipelineConfig,
    LLMConfig,
    EvaluationBudget,
    DiversityConfig,
    DatapointStrategyType,
    NoiseReductionType,
)
from vector_forge.core.results import (
    ExtractionResult,
    EvaluationResult,
    DatapointQuality,
    DiversityMetrics,
)
from vector_forge.core.state import ExtractionState, Checkpoint
from vector_forge.core.events import Event, EventType

__all__ = [
    "LLMClient",
    "Tool",
    "ToolResult",
    "DatapointStrategy",
    "NoiseReducer",
    "LayerSearchStrategy",
    "VectorEvaluator",
    "EventEmitter",
    "EventHandler",
    "BehaviorSpec",
    "PipelineConfig",
    "LLMConfig",
    "EvaluationBudget",
    "DiversityConfig",
    "DatapointStrategyType",
    "NoiseReductionType",
    "ExtractionResult",
    "EvaluationResult",
    "DatapointQuality",
    "DiversityMetrics",
    "ExtractionState",
    "Checkpoint",
    "Event",
    "EventType",
]
