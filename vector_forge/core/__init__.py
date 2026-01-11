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
from vector_forge.core.concurrency import (
    ConcurrencyConfig,
    ConcurrencyManager,
    get_concurrency_manager,
    reset_concurrency_manager,
    get_gpu_executor,
    get_evaluation_semaphore,
    get_dimension_semaphore,
    get_llm_semaphore,
    run_in_gpu_executor,
    limit_concurrent_evaluations,
    limit_concurrent_dimensions,
    limit_concurrent_llm_calls,
)

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
    # Concurrency management
    "ConcurrencyConfig",
    "ConcurrencyManager",
    "get_concurrency_manager",
    "reset_concurrency_manager",
    "get_gpu_executor",
    "get_evaluation_semaphore",
    "get_dimension_semaphore",
    "get_llm_semaphore",
    "run_in_gpu_executor",
    "limit_concurrent_evaluations",
    "limit_concurrent_dimensions",
    "limit_concurrent_llm_calls",
]
