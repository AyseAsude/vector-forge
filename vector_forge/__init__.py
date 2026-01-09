"""
Vector Forge - Autonomous pipeline for extracting clean steering vectors from LLMs.

This package provides:
- Diverse training datapoint generation with quality analysis
- Multi-layer steering vector optimization
- LLM-based evaluation with detailed scoring and citations
- Noise reduction for cleaner vectors
- Full pipeline orchestration with configurable strategies
- Parallel exploration of extraction strategies for optimal results
"""

__version__ = "0.1.0"

from vector_forge.core.behavior import BehaviorSpec
from vector_forge.core.config import PipelineConfig, EvaluationBudget
from vector_forge.tasks.config import TaskConfig
from vector_forge.tasks.task import ExtractionTask, TaskResult
from vector_forge.tasks.runner import TaskRunner
from vector_forge.tasks.expander import BehaviorExpander, ExpandedBehavior

__all__ = [
    # Core
    "BehaviorSpec",
    "PipelineConfig",
    "EvaluationBudget",
    # Tasks
    "TaskConfig",
    "ExtractionTask",
    "TaskResult",
    "TaskRunner",
    "BehaviorExpander",
    "ExpandedBehavior",
]
