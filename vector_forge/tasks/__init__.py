"""Task-based extraction system for Vector Forge.

This module provides a parallel exploration framework for steering vector extraction,
inspired by the inspect_ai task model. Instead of sequential refinement, it runs
multiple extraction strategies in parallel and selects the best results.

Key Components:
    - ExtractionTask: Defines what to extract and how to evaluate
    - ExtractionSample: A single extraction attempt with specific configuration
    - TaskRunner: Executes tasks with parallelism control
    - BehaviorExpander: Enriches user descriptions into comprehensive specs

Example:
    >>> from vector_forge.tasks import ExtractionTask, TaskRunner
    >>> task = ExtractionTask.from_description(
    ...     "sycophancy - agreeing with users even when wrong",
    ...     num_samples=16,
    ... )
    >>> runner = TaskRunner(backend, llm_client, max_concurrent=8)
    >>> result = await runner.run(task)
"""

from vector_forge.tasks.config import (
    TaskConfig,
    SampleConfig,
    EvaluationConfig,
    AggregationStrategy,
    LayerStrategy,
)
from vector_forge.tasks.sample import ExtractionSample, SampleGenerator
from vector_forge.tasks.task import ExtractionTask, TaskResult
from vector_forge.tasks.runner import TaskRunner
from vector_forge.tasks.expander import BehaviorExpander

__all__ = [
    # Configuration
    "TaskConfig",
    "SampleConfig",
    "EvaluationConfig",
    "AggregationStrategy",
    "LayerStrategy",
    # Core
    "ExtractionSample",
    "SampleGenerator",
    "ExtractionTask",
    "TaskResult",
    "TaskRunner",
    "BehaviorExpander",
]
