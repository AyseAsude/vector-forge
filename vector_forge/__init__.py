"""
Vector Forge - Autonomous pipeline for extracting clean steering vectors from LLMs.

This package provides:
- Diverse training datapoint generation with quality analysis
- Multi-layer steering vector optimization
- LLM-based evaluation with detailed scoring and citations
- Noise reduction for cleaner vectors
- Full pipeline orchestration with configurable strategies
"""

__version__ = "0.1.0"

from vector_forge.core.behavior import BehaviorSpec
from vector_forge.core.config import PipelineConfig, EvaluationBudget

__all__ = [
    "BehaviorSpec",
    "PipelineConfig",
    "EvaluationBudget",
]
