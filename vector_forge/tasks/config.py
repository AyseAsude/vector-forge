"""Configuration models for the task-based extraction system.

Provides comprehensive configuration for parallel extraction tasks, including
sample generation parameters, evaluation settings, and aggregation strategies.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class LayerStrategy(str, Enum):
    """Strategy for selecting which layers to optimize."""

    AUTO = "auto"
    FIXED = "fixed"
    SWEEP = "sweep"
    MIDDLE = "middle"
    LATE = "late"


class AggregationStrategy(str, Enum):
    """Strategy for combining multiple extraction results."""

    BEST_SINGLE = "best_single"
    TOP_K_AVERAGE = "top_k_average"
    WEIGHTED_AVERAGE = "weighted_average"
    PCA_PRINCIPAL = "pca_principal"
    STRATEGY_GROUPED = "strategy_grouped"


class SampleConfig(BaseModel):
    """Configuration for a single extraction sample.

    Each sample represents one extraction attempt with specific hyperparameters.
    Multiple samples with varied configurations enable parallel exploration
    of the strategy space.
    """

    seed: int = Field(default=0, ge=0, description="Random seed for reproducibility")

    layer_strategy: LayerStrategy = Field(
        default=LayerStrategy.AUTO,
        description="How to select layers for optimization",
    )

    target_layers: Optional[List[int]] = Field(
        default=None,
        description="Specific layers to target when using FIXED strategy",
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for datapoint generation",
    )

    num_datapoints: int = Field(
        default=50,
        ge=5,
        le=500,
        description="Number of contrast pairs to generate",
    )

    optimization_iterations: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Maximum iterations for vector optimization",
    )

    learning_rate: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Learning rate for optimization",
    )

    use_mean_centering: bool = Field(
        default=True,
        description="Apply mean-centering to reduce global activation bias",
    )

    bootstrap_ratio: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Fraction of datapoints to use (for bootstrap diversity)",
    )


class EvaluationConfig(BaseModel):
    """Configuration for comprehensive vector evaluation.

    Defines the inference budget and evaluation dimensions for judging
    steering vector quality across multiple criteria.
    """

    # Behavior induction testing
    behavior_prompts: int = Field(
        default=50,
        ge=10,
        description="Prompts for testing behavior induction",
    )

    behavior_generations_per_prompt: int = Field(
        default=3,
        ge=1,
        description="Generations per prompt for variance estimation",
    )

    # Specificity testing
    specificity_prompts: int = Field(
        default=50,
        ge=10,
        description="Neutral prompts for specificity testing",
    )

    # Coherence testing
    coherence_prompts: int = Field(
        default=30,
        ge=10,
        description="Prompts for coherence evaluation",
    )

    # Capability preservation
    capability_prompts: int = Field(
        default=20,
        ge=5,
        description="Benchmark prompts for capability testing",
    )

    # Generalization testing
    generalization_prompts: int = Field(
        default=30,
        ge=10,
        description="Out-of-distribution prompts",
    )

    # Strength calibration
    strength_levels: List[float] = Field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0],
        description="Steering strengths to test",
    )

    # Scoring weights
    behavior_weight: float = Field(default=0.30, ge=0, le=1)
    specificity_weight: float = Field(default=0.25, ge=0, le=1)
    coherence_weight: float = Field(default=0.20, ge=0, le=1)
    capability_weight: float = Field(default=0.15, ge=0, le=1)
    generalization_weight: float = Field(default=0.10, ge=0, le=1)

    @field_validator("strength_levels")
    @classmethod
    def validate_strength_levels(cls, v: List[float]) -> List[float]:
        if len(v) < 2:
            raise ValueError("At least 2 strength levels required")
        return sorted(v)

    @property
    def total_weight(self) -> float:
        """Verify weights sum to 1.0."""
        return (
            self.behavior_weight
            + self.specificity_weight
            + self.coherence_weight
            + self.capability_weight
            + self.generalization_weight
        )

    @classmethod
    def fast(cls) -> "EvaluationConfig":
        """Minimal evaluation for rapid iteration."""
        return cls(
            behavior_prompts=20,
            behavior_generations_per_prompt=2,
            specificity_prompts=20,
            coherence_prompts=15,
            capability_prompts=10,
            generalization_prompts=15,
            strength_levels=[0.5, 1.0, 1.5],
        )

    @classmethod
    def standard(cls) -> "EvaluationConfig":
        """Balanced evaluation for normal use."""
        return cls()

    @classmethod
    def thorough(cls) -> "EvaluationConfig":
        """Comprehensive evaluation for production quality."""
        return cls(
            behavior_prompts=100,
            behavior_generations_per_prompt=5,
            specificity_prompts=100,
            coherence_prompts=50,
            capability_prompts=30,
            generalization_prompts=50,
            strength_levels=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        )


class TaskConfig(BaseModel):
    """Master configuration for an extraction task.

    Combines sample generation, evaluation, and aggregation settings
    into a comprehensive task specification.

    Example:
        >>> config = TaskConfig(
        ...     num_samples=16,
        ...     max_concurrent=8,
        ...     evaluation=EvaluationConfig.standard(),
        ... )
    """

    # Sample generation
    num_samples: int = Field(
        default=16,
        ge=1,
        le=100,
        description="Number of parallel extraction attempts",
    )

    num_seeds: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Different random seeds to try",
    )

    layer_strategies: List[LayerStrategy] = Field(
        default_factory=lambda: [LayerStrategy.AUTO, LayerStrategy.SWEEP],
        description="Layer selection strategies to explore",
    )

    temperatures: List[float] = Field(
        default_factory=lambda: [0.5, 0.7, 1.0],
        description="Temperature values to try",
    )

    datapoint_counts: List[int] = Field(
        default_factory=lambda: [30, 50, 100],
        description="Different datapoint counts to try",
    )

    # Contrast pair generation
    contrast_pair_count: int = Field(
        default=100,
        ge=20,
        le=500,
        description="Total contrast pairs to generate",
    )

    contrast_domains: List[str] = Field(
        default_factory=lambda: [
            "science",
            "personal_advice",
            "factual_qa",
            "opinion",
            "creative",
            "technical",
            "ethics",
            "current_events",
        ],
        description="Domains for contrast pair diversity",
    )

    # Parallelism control
    max_concurrent_extractions: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Maximum parallel extraction workers",
    )

    max_concurrent_evaluations: int = Field(
        default=16,
        ge=1,
        le=64,
        description="Maximum parallel evaluation generations",
    )

    # Evaluation settings
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Evaluation configuration",
    )

    # Aggregation
    aggregation_strategy: AggregationStrategy = Field(
        default=AggregationStrategy.TOP_K_AVERAGE,
        description="How to combine results",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top results for aggregation",
    )

    # Output
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory for saving results",
    )

    save_all_vectors: bool = Field(
        default=False,
        description="Save all candidate vectors, not just final",
    )

    # LLM settings
    extractor_model: str = Field(
        default="gpt-4o",
        description="Model for datapoint generation",
    )

    judge_model: str = Field(
        default="gpt-4o",
        description="Model for evaluation judging",
    )

    expander_model: str = Field(
        default="gpt-4o",
        description="Model for behavior expansion",
    )

    @classmethod
    def quick(cls) -> "TaskConfig":
        """Quick configuration for testing."""
        return cls(
            num_samples=4,
            num_seeds=2,
            layer_strategies=[LayerStrategy.AUTO],
            temperatures=[0.7],
            datapoint_counts=[30],
            contrast_pair_count=30,
            max_concurrent_extractions=4,
            evaluation=EvaluationConfig.fast(),
            top_k=2,
        )

    @classmethod
    def standard(cls) -> "TaskConfig":
        """Standard configuration for normal use."""
        return cls()

    @classmethod
    def comprehensive(cls) -> "TaskConfig":
        """Comprehensive configuration for best results."""
        return cls(
            num_samples=32,
            num_seeds=8,
            layer_strategies=[
                LayerStrategy.AUTO,
                LayerStrategy.SWEEP,
                LayerStrategy.MIDDLE,
                LayerStrategy.LATE,
            ],
            temperatures=[0.3, 0.5, 0.7, 1.0],
            datapoint_counts=[50, 100, 150],
            contrast_pair_count=200,
            max_concurrent_extractions=16,
            max_concurrent_evaluations=32,
            evaluation=EvaluationConfig.thorough(),
            top_k=8,
            save_all_vectors=True,
        )
