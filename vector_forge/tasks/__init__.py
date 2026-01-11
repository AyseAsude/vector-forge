"""Task-based extraction system for Vector Forge.

This module provides a parallel exploration framework for steering vector extraction.
Supports both CAA (Contrastive Activation Addition) and gradient-based optimization.
Runs multiple extraction strategies in parallel and aggregates the best results.

Key Components:
    - ExtractionTask: Defines what to extract and how to evaluate
    - ExtractionSample: A single extraction attempt with specific configuration
    - TaskRunner: Executes tasks with parallelism control
    - BehaviorExpander: Enriches user descriptions into comprehensive specs
    - ContrastToTrainingAdapter: Converts contrast pairs to training datapoints

Example:
    >>> from vector_forge.tasks import ExtractionTask, TaskRunner, TaskConfig
    >>> config = TaskConfig(extraction_method=ExtractionMethod.CAA)
    >>> task = ExtractionTask.from_behavior(behavior, config=config)
    >>> runner = TaskRunner(backend, llm_client)
    >>> result = await runner.run(task, sample_datasets)
"""

from vector_forge.tasks.config import (
    TaskConfig,
    SampleConfig,
    EvaluationConfig,
    OptimizationConfig,
    ContrastConfig,
    ContrastQuality,
    AggregationStrategy,
    LayerStrategy,
    ExtractionMethod,
    TokenPosition,
    CAAConfig,
)
from vector_forge.tasks.sample import (
    ExtractionSample,
    SampleSet,
    SampleGenerator,
)
from vector_forge.tasks.task import (
    ExtractionTask,
    TaskResult,
    SampleResult,
)
from vector_forge.tasks.runner import (
    TaskRunner,
    ExtractionResult,
    RunnerProgress,
    reaggregate_results,
)
from vector_forge.tasks.expander import BehaviorExpander, ExpandedBehavior
from vector_forge.tasks.adapter import (
    ContrastToTrainingAdapter,
    DatapointSerializer,
    SerializedDatapoint,
    OptimizationResultData,
    pairs_to_datapoints,
    dataset_to_datapoints,
)

__all__ = [
    # Configuration
    "TaskConfig",
    "SampleConfig",
    "EvaluationConfig",
    "OptimizationConfig",
    "ContrastConfig",
    "ContrastQuality",
    "AggregationStrategy",
    "LayerStrategy",
    "ExtractionMethod",
    "TokenPosition",
    "CAAConfig",
    # Samples
    "ExtractionSample",
    "SampleSet",
    "SampleGenerator",
    # Tasks and Results
    "ExtractionTask",
    "TaskResult",
    "SampleResult",
    # Runner
    "TaskRunner",
    "ExtractionResult",
    "RunnerProgress",
    "reaggregate_results",
    # Behavior
    "BehaviorExpander",
    "ExpandedBehavior",
    # Adapters
    "ContrastToTrainingAdapter",
    "DatapointSerializer",
    "SerializedDatapoint",
    "OptimizationResultData",
    "pairs_to_datapoints",
    "dataset_to_datapoints",
]
