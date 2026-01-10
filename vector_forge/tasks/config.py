"""Configuration models for the task-based extraction system.

Provides comprehensive configuration for parallel extraction tasks, including
optimization parameters, contrast generation, evaluation settings, and aggregation strategies.

Architecture follows SOLID principles:
- Single Responsibility: Each config class handles one concern
- Open/Closed: Presets extend base configs without modification
- Interface Segregation: Separate configs for separate concerns
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from vector_forge.constants import DEFAULT_MODEL


class ContrastQuality(str, Enum):
    """Contrast generation quality preset."""

    FAST = "fast"
    STANDARD = "standard"
    THOROUGH = "thorough"


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


# ============================================================================
# Optimization Configuration (NEW - for steering-vectors library)
# ============================================================================


class OptimizationConfig(BaseModel):
    """Configuration for steering vector optimization.

    Controls the gradient-based optimization process that directly
    optimizes log-probabilities of target completions.

    Example:
        >>> config = OptimizationConfig(lr=0.1, max_iters=50)
        >>> # For faster iteration:
        >>> fast_config = OptimizationConfig.fast()
    """

    # Core optimization parameters
    lr: float = Field(
        default=0.1,
        gt=0,
        le=1.0,
        description="Adam learning rate",
    )

    max_iters: int = Field(
        default=50,
        gt=0,
        le=500,
        description="Maximum optimization steps",
    )

    # Temperature control
    coldness: float = Field(
        default=0.7,
        gt=0,
        le=2.0,
        description="Inverse temperature for softmax (higher = sharper)",
    )

    # Regularization via norm constraints
    starting_norm: float = Field(
        default=1.0,
        gt=0,
        description="Initial L2 norm of steering vector",
    )

    max_norm: Optional[float] = Field(
        default=2.0,
        gt=0,
        description="Clip vector norm after each step (None = no clipping)",
    )

    # Loss computation
    normalize_by_length: bool = Field(
        default=True,
        description="Divide loss by completion length for fair comparison",
    )

    use_one_minus: bool = Field(
        default=True,
        description="Use log(1-p) for suppression (vs -log(p))",
    )

    # Early stopping
    target_loss: Optional[float] = Field(
        default=None,
        description="Stop when loss <= this value",
    )

    convergence_eps: float = Field(
        default=1e-5,
        gt=0,
        description="Stop when loss change < eps",
    )

    convergence_patience: int = Field(
        default=3,
        ge=1,
        description="Consecutive steps below eps before stopping",
    )

    # Batched optimization (performance)
    use_batched: bool = Field(
        default=True,
        description="Use batched forward passes for 10-50x faster optimization",
    )

    batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Batch size for batched optimization (larger = faster but more memory)",
    )

    @classmethod
    def fast(cls) -> "OptimizationConfig":
        """Fast configuration for testing."""
        return cls(
            lr=0.15,
            max_iters=30,
            max_norm=3.0,
            normalize_by_length=True,
            use_batched=True,
            batch_size=16,
        )

    @classmethod
    def standard(cls) -> "OptimizationConfig":
        """Standard configuration for normal use."""
        return cls()

    @classmethod
    def thorough(cls) -> "OptimizationConfig":
        """Thorough configuration for production quality."""
        return cls(
            lr=0.08,
            max_iters=100,
            max_norm=1.5,
            starting_norm=0.5,
            convergence_eps=1e-6,
            convergence_patience=5,
            use_batched=True,
            batch_size=8,  # Smaller batch for more iterations
        )

    def to_steering_config(self) -> dict:
        """Convert to steering-vectors OptimizationConfig kwargs."""
        return {
            "lr": self.lr,
            "max_iters": self.max_iters,
            "coldness": self.coldness,
            "starting_norm": self.starting_norm,
            "max_norm": self.max_norm,
            "normalize_by_length": self.normalize_by_length,
            "use_one_minus": self.use_one_minus,
            "use_batched": self.use_batched,
            "batch_size": self.batch_size,
        }


# ============================================================================
# Contrast Generation Configuration
# ============================================================================


class ContrastConfig(BaseModel):
    """Configuration for contrast pair generation pipeline.

    Controls the quality and validation settings for generating
    high-quality contrast pairs used in steering vector extraction.
    """

    # Pool settings
    core_pool_size: int = Field(
        default=80,
        ge=20,
        le=500,
        description="Number of pairs in the shared core pool",
    )

    core_seeds_per_sample: int = Field(
        default=40,
        ge=10,
        le=200,
        description="How many core seeds each sample uses",
    )

    unique_seeds_per_sample: int = Field(
        default=10,
        ge=0,
        le=50,
        description="How many unique seeds each sample generates",
    )

    # Validation thresholds
    min_semantic_distance: float = Field(
        default=0.3,
        ge=0.1,
        le=0.9,
        description="Minimum semantic distance between dst and src",
    )

    min_dst_score: float = Field(
        default=7.0,
        ge=1.0,
        le=10.0,
        description="Minimum behavior score for dst (exhibits behavior)",
    )

    max_src_score: float = Field(
        default=3.0,
        ge=0.0,
        le=9.0,
        description="Maximum behavior score for src (no behavior)",
    )

    min_contrast_quality: float = Field(
        default=6.0,
        ge=1.0,
        le=10.0,
        description="Minimum overall contrast quality score",
    )

    # Regeneration settings
    max_regeneration_attempts: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum attempts to regenerate a failed pair",
    )

    # Seed quality
    min_seed_quality: float = Field(
        default=6.0,
        ge=1.0,
        le=10.0,
        description="Minimum quality score for seeds",
    )

    # Generation settings
    generation_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for pair generation",
    )

    # Parallelism
    max_concurrent_generations: int = Field(
        default=16,
        ge=1,
        le=64,
        description="Maximum concurrent LLM API calls for pair generation",
    )

    @property
    def pairs_per_sample(self) -> int:
        """Total pairs per sample (core + unique)."""
        return self.core_seeds_per_sample + self.unique_seeds_per_sample

    @classmethod
    def fast(cls) -> "ContrastConfig":
        """Fast configuration for testing and iteration."""
        return cls(
            core_pool_size=30,
            core_seeds_per_sample=20,
            unique_seeds_per_sample=5,
            max_regeneration_attempts=1,
            min_seed_quality=5.0,
            min_dst_score=6.0,
            max_src_score=4.0,
            min_contrast_quality=5.0,
        )

    @classmethod
    def standard(cls) -> "ContrastConfig":
        """Standard configuration for normal use."""
        return cls()

    @classmethod
    def thorough(cls) -> "ContrastConfig":
        """Thorough configuration for production quality."""
        return cls(
            core_pool_size=120,
            core_seeds_per_sample=50,
            unique_seeds_per_sample=15,
            max_regeneration_attempts=3,
            min_seed_quality=7.0,
            min_dst_score=8.0,
            max_src_score=2.0,
            min_contrast_quality=7.0,
            max_concurrent_generations=8,
        )

    @classmethod
    def from_preset(cls, preset: ContrastQuality) -> "ContrastConfig":
        """Create configuration from preset."""
        if preset == ContrastQuality.FAST:
            return cls.fast()
        elif preset == ContrastQuality.THOROUGH:
            return cls.thorough()
        return cls.standard()


# ============================================================================
# Sample Configuration
# ============================================================================


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

    # Optimization settings (per-sample overrides)
    lr: Optional[float] = Field(
        default=None,
        gt=0,
        le=1.0,
        description="Override learning rate for this sample",
    )

    max_iters: Optional[int] = Field(
        default=None,
        gt=0,
        le=500,
        description="Override max iterations for this sample",
    )

    # Data settings
    bootstrap_ratio: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Fraction of datapoints to use (for bootstrap diversity)",
    )

    def get_lr(self, default: float) -> float:
        """Get learning rate with fallback to default."""
        return self.lr if self.lr is not None else default

    def get_max_iters(self, default: int) -> int:
        """Get max iterations with fallback to default."""
        return self.max_iters if self.max_iters is not None else default


# ============================================================================
# Evaluation Configuration
# ============================================================================


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


# ============================================================================
# Master Task Configuration
# ============================================================================


class TaskConfig(BaseModel):
    """Master configuration for an extraction task.

    Combines sample generation, optimization, evaluation, and aggregation settings
    into a comprehensive task specification.

    Example:
        >>> config = TaskConfig(
        ...     num_samples=16,
        ...     max_concurrent_extractions=8,
        ...     optimization=OptimizationConfig.standard(),
        ...     contrast=ContrastConfig.standard(),
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

    # Optimization settings (NEW - replaces legacy CAA)
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Steering vector optimization configuration",
    )

    # Contrast pair generation
    contrast: ContrastConfig = Field(
        default_factory=ContrastConfig,
        description="Contrast pair generation configuration",
    )

    # Datapoints per sample (explicit control)
    datapoints_per_sample: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of contrast pairs used per sample for optimization",
    )

    # Parallelism control
    # NOTE: MemoryEstimator automatically limits concurrency based on GPU memory
    # and batch size. Higher values here allow more parallelism on larger GPUs.
    max_concurrent_extractions: int = Field(
        default=16,
        ge=1,
        le=32,
        description="Maximum parallel extraction workers (auto-limited by GPU memory)",
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

    # Target model (HuggingFace model for steering vector extraction)
    target_model: Optional[str] = Field(
        default=None,
        description="HuggingFace model ID or path for steering vector extraction",
    )

    # LLM settings (for agents)
    extractor_model: str = Field(
        default=DEFAULT_MODEL,
        description="Model for datapoint generation",
    )

    judge_model: str = Field(
        default=DEFAULT_MODEL,
        description="Model for evaluation judging",
    )

    expander_model: str = Field(
        default=DEFAULT_MODEL,
        description="Model for behavior expansion",
    )

    @model_validator(mode="after")
    def validate_top_k(self) -> "TaskConfig":
        """Ensure top_k doesn't exceed num_samples."""
        if self.top_k > self.num_samples:
            self.top_k = self.num_samples
        return self

    @classmethod
    def quick(cls) -> "TaskConfig":
        """Quick configuration for testing."""
        return cls(
            num_samples=4,
            num_seeds=2,
            layer_strategies=[LayerStrategy.AUTO],
            optimization=OptimizationConfig.fast(),
            contrast=ContrastConfig.fast(),
            datapoints_per_sample=25,
            max_concurrent_extractions=16,  # Auto-limited by MemoryEstimator
            evaluation=EvaluationConfig.fast(),
            top_k=2,
        )

    @classmethod
    def standard(cls) -> "TaskConfig":
        """Standard configuration for normal use."""
        return cls(
            optimization=OptimizationConfig.standard(),
            contrast=ContrastConfig.standard(),
            datapoints_per_sample=50,
            max_concurrent_extractions=16,  # Auto-limited by MemoryEstimator
        )

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
            optimization=OptimizationConfig.thorough(),
            contrast=ContrastConfig.thorough(),
            datapoints_per_sample=80,
            max_concurrent_extractions=16,  # Auto-limited by MemoryEstimator
            max_concurrent_evaluations=32,
            evaluation=EvaluationConfig.thorough(),
            top_k=8,
        )
