"""Configuration models for the task-based extraction system.

Provides comprehensive configuration for parallel extraction tasks, including
optimization parameters, contrast generation, evaluation settings, and aggregation strategies.

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


class IntensityProfile(str, Enum):
    """Intensity distribution profile for contrast pairs.

    - EXTREME: Heavy on extreme/high intensity for quick direction finding
    - BALANCED: Default balanced distribution across all intensities
    - NATURAL: Heavy on natural/medium for production quality generalization
    """

    EXTREME = "extreme"
    BALANCED = "balanced"
    NATURAL = "natural"


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


class ExtractionMethod(str, Enum):
    """Method for extracting steering vectors."""

    CAA = "caa"
    GRADIENT = "gradient"
    HYBRID = "hybrid"


class TokenPosition(str, Enum):
    """Which token position to extract activations from for CAA."""

    MEAN = "mean"  # Mean of all response tokens (default, recommended)
    LAST = "last"  # Last token of response only
    LAST_PROMPT = "last_prompt_token"  # Last token before response


class EvalDepth(str, Enum):
    """Evaluation depth for tournament rounds.

    Each depth level defines a percentage of the full evaluation budget.
    Quick uses 10% of prompts/inferences, Medium uses 40%, Full uses 100%.
    """

    QUICK = "quick"  # 10% budget - fast elimination signal
    MEDIUM = "medium"  # 40% budget - moderate confidence
    FULL = "full"  # 100% budget - comprehensive evaluation


# Budget percentages for each evaluation depth
EVAL_DEPTH_BUDGET: dict[EvalDepth, float] = {
    EvalDepth.QUICK: 0.10,  # 10% of full evaluation
    EvalDepth.MEDIUM: 0.40,  # 40% of full evaluation
    EvalDepth.FULL: 1.0,  # 100% of full evaluation
}

# Prompt selection strategy: what percentage comes from "best" vs "random"
# For quick: 50% best prompts + 50% random (balance signal with diversity)
# For medium: 60% best + 40% random
# For full: 100% (all prompts, no selection needed)
EVAL_DEPTH_BEST_RATIO: dict[EvalDepth, float] = {
    EvalDepth.QUICK: 0.50,  # Half best, half random
    EvalDepth.MEDIUM: 0.60,  # 60% best, 40% random
    EvalDepth.FULL: 1.0,  # N/A - use all
}


# ============================================================================
# Tournament Configuration
# ============================================================================


class TournamentConfig(BaseModel):
    """Configuration for tournament/elimination-based extraction.

    Instead of running all samples with equal resources, the tournament
    system starts with many samples and progressively eliminates weak
    performers, focusing resources on promising candidates.

    User provides:
        - elimination_rounds: Number of elimination phases
        - final_survivors: Samples entering finals

    System calculates:
        - initial_samples: Based on 75% elimination per round
        - datapoints per round: Graduated from min to max
        - eval depth per round: Quick → Medium → Full

    Example flows:
        rounds=1, survivors=8:  32 → 8 (finals)
        rounds=2, survivors=16: 256 → 64 → 16 (finals)
        rounds=3, survivors=32: 2048 → 512 → 128 → 32 (finals)
    """

    enabled: bool = Field(
        default=True,
        description="Enable tournament mode",
    )

    elimination_rounds: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of elimination phases before finals",
    )

    final_survivors: int = Field(
        default=16,
        ge=4,
        le=128,
        description="Number of samples entering finals",
    )

    elimination_rate: float = Field(
        default=0.75,
        ge=0.5,
        le=0.9,
        description="Fraction eliminated per round (0.75 = keep top 25%)",
    )

    # Datapoint progression
    min_datapoints: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Datapoints per sample in first round",
    )

    max_datapoints: int = Field(
        default=80,
        ge=20,
        le=200,
        description="Datapoints per sample in final round",
    )

    @property
    def keep_rate(self) -> float:
        """Fraction of samples kept per round."""
        return 1.0 - self.elimination_rate

    @property
    def initial_samples(self) -> int:
        """Calculate initial sample count from elimination rounds and survivors."""
        # Each round keeps (1 - elimination_rate) of samples
        # So initial = survivors / keep_rate^rounds
        return int(self.final_survivors / (self.keep_rate ** self.elimination_rounds))

    @property
    def total_rounds(self) -> int:
        """Total rounds including finals."""
        return self.elimination_rounds + 1

    def samples_at_round(self, round_idx: int) -> int:
        """Get sample count at start of a round (0-indexed).

        Args:
            round_idx: Round index (0 = first elimination round)

        Returns:
            Number of samples at start of that round.
        """
        return int(self.initial_samples * (self.keep_rate ** round_idx))

    def survivors_after_round(self, round_idx: int) -> int:
        """Get sample count after elimination in a round.

        Args:
            round_idx: Round index (0 = first elimination round)

        Returns:
            Number of survivors after elimination.
        """
        if round_idx >= self.elimination_rounds:
            # Finals - no elimination
            return self.final_survivors
        return self.samples_at_round(round_idx + 1)

    def datapoints_at_round(self, round_idx: int) -> int:
        """Get datapoints per sample for a round (graduated).

        Args:
            round_idx: Round index (0 = first round)

        Returns:
            Number of datapoints for samples in this round.
        """
        # Linear interpolation from min to max
        if self.elimination_rounds == 0:
            return self.max_datapoints
        progress = round_idx / self.elimination_rounds
        return int(self.min_datapoints + progress * (self.max_datapoints - self.min_datapoints))

    def eval_depth_at_round(self, round_idx: int) -> EvalDepth:
        """Get evaluation depth for a round.

        Args:
            round_idx: Round index (0 = first round)

        Returns:
            EvalDepth for this round.
        """
        if round_idx == 0:
            return EvalDepth.QUICK
        elif round_idx < self.elimination_rounds:
            return EvalDepth.MEDIUM
        else:
            return EvalDepth.FULL

    def get_round_summary(self) -> list[dict]:
        """Get summary of all rounds for display/logging."""
        rounds = []
        for i in range(self.total_rounds):
            is_finals = i >= self.elimination_rounds
            rounds.append({
                "round": i + 1,
                "samples": self.samples_at_round(i),
                "survivors": self.survivors_after_round(i),
                "datapoints": self.datapoints_at_round(i),
                "eval_depth": self.eval_depth_at_round(i).value,
                "is_finals": is_finals,
            })
        return rounds

    @classmethod
    def quick(cls) -> "TournamentConfig":
        """Quick tournament: 32→4 (1 round, 87.5% elimination)."""
        return cls(
            enabled=True,
            elimination_rounds=1,
            final_survivors=4,
            elimination_rate=0.875,
            min_datapoints=15,
            max_datapoints=40,
        )

    @classmethod
    def standard(cls) -> "TournamentConfig":
        """Standard tournament: 256→16 (2 rounds, 75% elimination)."""
        return cls(
            enabled=True,
            elimination_rounds=2,
            final_survivors=16,
            elimination_rate=0.75,
            min_datapoints=15,
            max_datapoints=60,
        )

    @classmethod
    def comprehensive(cls) -> "TournamentConfig":
        """Comprehensive tournament: 1024→32 (3 rounds, 68.5% elimination)."""
        return cls(
            enabled=True,
            elimination_rounds=3,
            final_survivors=32,
            elimination_rate=0.685,
            min_datapoints=15,
            max_datapoints=80,
        )


# ============================================================================
# Signal Filtering Configuration
# ============================================================================


class SignalFilterMode(str, Enum):
    """How to filter pairs for extraction based on signal quality."""

    OFF = "off"              # No filtering (legacy behavior)
    THRESHOLD = "threshold"  # Filter by minimum behavioral signal score
    TOP_K = "top_k"          # Keep only top K pairs by behavioral signal


class ExtractionIntensity(str, Enum):
    """Which intensity levels to use for extraction (not evaluation)."""

    ALL = "all"                    # Use all intensity levels
    HIGH_SIGNAL = "high_signal"    # Only extreme + high intensity
    MAXIMUM = "maximum"            # Only extreme intensity


# ============================================================================
# CAA Configuration
# ============================================================================


class CAAConfig(BaseModel):
    """Configuration for CAA-specific extraction settings.

    Includes signal quality filtering to maximize vector quality by:
    1. Filtering pairs by behavioral signal strength
    2. Filtering pairs by confound control
    3. Using only high-intensity pairs for extraction
    """

    # -------------------------------------------------------------------------
    # Outlier Removal (activation-space)
    # -------------------------------------------------------------------------

    remove_extreme_outliers: bool = Field(
        default=True,
        description="Remove extreme outlier pairs (> outlier_std_threshold std devs)",
    )

    outlier_std_threshold: float = Field(
        default=3.0,
        ge=2.0,
        le=5.0,
        description="Std dev threshold for extreme outlier removal",
    )

    # -------------------------------------------------------------------------
    # Behavioral Signal Filtering (NEW)
    # -------------------------------------------------------------------------

    signal_filter_mode: SignalFilterMode = Field(
        default=SignalFilterMode.OFF,
        description="How to filter pairs by behavioral signal strength",
    )

    min_behavioral_signal: float = Field(
        default=6.0,
        ge=1.0,
        le=10.0,
        description="Minimum behavioral signal score (when mode=THRESHOLD)",
    )

    top_k_pairs: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Number of top pairs to keep (when mode=TOP_K)",
    )

    # -------------------------------------------------------------------------
    # Confound Control Filtering (NEW)
    # -------------------------------------------------------------------------

    min_confound_score: float = Field(
        default=5.0,
        ge=1.0,
        le=10.0,
        description="Minimum confound control score (0 = disabled)",
    )

    # -------------------------------------------------------------------------
    # Intensity Filtering for Extraction (NEW)
    # -------------------------------------------------------------------------

    extraction_intensity: ExtractionIntensity = Field(
        default=ExtractionIntensity.ALL,
        description="Which intensity levels to use for extraction",
    )

    # -------------------------------------------------------------------------
    # Diversity Enforcement (NEW - anti-overfitting)
    # -------------------------------------------------------------------------

    min_scenarios: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Minimum distinct scenarios in filtered set (anti-overfitting)",
    )

    require_intensity_diversity: bool = Field(
        default=False,
        description="Require at least 2 intensity levels in filtered set",
    )

    # -------------------------------------------------------------------------
    # Presets
    # -------------------------------------------------------------------------

    @classmethod
    def maximum_signal(cls) -> "CAAConfig":
        """Config optimized for maximum behavioral signal.

        Uses aggressive filtering to get cleanest possible vector.
        May reduce diversity - use for well-understood behaviors.
        """
        return cls(
            signal_filter_mode=SignalFilterMode.TOP_K,
            top_k_pairs=25,
            min_confound_score=6.0,
            extraction_intensity=ExtractionIntensity.MAXIMUM,
            remove_extreme_outliers=True,
            outlier_std_threshold=2.5,
            min_scenarios=3,
        )

    @classmethod
    def high_signal(cls) -> "CAAConfig":
        """Config for high signal with moderate filtering.

        Good balance between signal quality and diversity.
        """
        return cls(
            signal_filter_mode=SignalFilterMode.THRESHOLD,
            min_behavioral_signal=6.0,
            min_confound_score=5.0,
            extraction_intensity=ExtractionIntensity.HIGH_SIGNAL,
            remove_extreme_outliers=True,
            outlier_std_threshold=3.0,
            min_scenarios=5,
            require_intensity_diversity=True,
        )

    @classmethod
    def balanced(cls) -> "CAAConfig":
        """Default balanced config (legacy behavior with outlier removal)."""
        return cls(
            signal_filter_mode=SignalFilterMode.OFF,
            remove_extreme_outliers=True,
            outlier_std_threshold=3.0,
        )


# ============================================================================
# Optimization Configuration (for gradient-based extraction)
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
        ge=1,
        le=1000,
        description="Number of pairs in the shared core pool",
    )

    core_seeds_per_sample: int = Field(
        default=40,
        ge=1,
        le=500,
        description="How many core seeds each sample uses",
    )

    unique_seeds_per_sample: int = Field(
        default=10,
        ge=0,
        le=200,
        description="How many unique seeds each sample generates",
    )

    # Validation thresholds
    min_semantic_score: float = Field(
        default=4.0,
        ge=1.0,
        le=10.0,
        description="Minimum semantic distance score (0-10 scale)",
    )

    min_dimension_score: float = Field(
        default=6.0,
        ge=1.0,
        le=10.0,
        description="Minimum dimension check score (contrast on right variable)",
    )

    min_structural_score: float = Field(
        default=7.0,
        ge=1.0,
        le=10.0,
        description="Minimum structural check score (well-formed responses)",
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
        le=50,
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
        default=64,
        ge=1,
        le=512,
        description="Maximum concurrent LLM API calls for pair generation",
    )

    # Intensity distribution for contrast pairs
    intensity_extreme: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Proportion of extreme intensity pairs (0.0-1.0)",
    )

    intensity_high: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Proportion of high intensity pairs (0.0-1.0)",
    )

    intensity_medium: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Proportion of medium intensity pairs (0.0-1.0)",
    )

    intensity_natural: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Proportion of natural intensity pairs (0.0-1.0)",
    )

    @property
    def pairs_per_sample(self) -> int:
        """Total pairs per sample (core + unique)."""
        return self.core_seeds_per_sample + self.unique_seeds_per_sample

    @property
    def intensity_distribution(self) -> dict:
        """Get intensity distribution as a dictionary."""
        from vector_forge.contrast.protocols import SignalIntensity
        return {
            SignalIntensity.EXTREME: self.intensity_extreme,
            SignalIntensity.HIGH: self.intensity_high,
            SignalIntensity.MEDIUM: self.intensity_medium,
            SignalIntensity.NATURAL: self.intensity_natural,
        }

    @model_validator(mode='after')
    def validate_intensity_sum(self) -> 'ContrastConfig':
        """Validate that intensity proportions sum to approximately 1.0."""
        total = self.intensity_extreme + self.intensity_high + self.intensity_medium + self.intensity_natural
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Intensity proportions must sum to ~1.0, got {total:.2f}")
        return self

    @classmethod
    def fast(cls) -> "ContrastConfig":
        """Fast configuration for testing and iteration."""
        return cls(
            core_pool_size=30,
            core_seeds_per_sample=20,
            unique_seeds_per_sample=5,
            max_regeneration_attempts=1,
            min_seed_quality=5.0,
            min_dimension_score=5.0,
            min_structural_score=6.0,
            min_semantic_score=3.0,
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
            min_dimension_score=7.0,
            min_structural_score=8.0,
            min_semantic_score=5.0,
            min_contrast_quality=7.0,
        )

    @classmethod
    def from_preset(cls, preset: ContrastQuality) -> "ContrastConfig":
        """Create configuration from preset."""
        if preset == ContrastQuality.FAST:
            return cls.fast()
        elif preset == ContrastQuality.THOROUGH:
            return cls.thorough()
        return cls.standard()

    # -------------------------------------------------------------------------
    # Intensity Distribution Presets
    # -------------------------------------------------------------------------

    @staticmethod
    def get_intensity_preset(profile: IntensityProfile) -> dict[str, float]:
        """Get intensity distribution values for a profile.

        Args:
            profile: The intensity profile to get values for.

        Returns:
            Dictionary with intensity_extreme, intensity_high,
            intensity_medium, intensity_natural values.
        """
        presets = {
            IntensityProfile.EXTREME: {
                "intensity_extreme": 0.40,
                "intensity_high": 0.30,
                "intensity_medium": 0.20,
                "intensity_natural": 0.10,
            },
            IntensityProfile.BALANCED: {
                "intensity_extreme": 0.40,
                "intensity_high": 0.30,
                "intensity_medium": 0.20,
                "intensity_natural": 0.10,
            },
            IntensityProfile.NATURAL: {
                "intensity_extreme": 0.40,
                "intensity_high": 0.30,
                "intensity_medium": 0.20,
                "intensity_natural": 0.10,
            },
        }
        return presets.get(profile, presets[IntensityProfile.BALANCED])

    def with_intensity_profile(self, profile: IntensityProfile) -> "ContrastConfig":
        """Return a copy of this config with a different intensity profile.

        Args:
            profile: The intensity profile to apply.

        Returns:
            New ContrastConfig with updated intensity values.
        """
        intensity_values = self.get_intensity_preset(profile)
        return self.model_copy(update=intensity_values)


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

    # CAA token position (assigned by generator, cycles through all positions)
    token_position: TokenPosition = Field(
        default=TokenPosition.MEAN,
        description="Token position for CAA extraction (auto-assigned by generator)",
    )

    # Optimization settings (per-sample overrides, for gradient method)
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
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5],
        description="Steering strengths to test",
    )

    # Generation temperature for steered outputs during evaluation
    generation_temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Temperature for generating steered outputs during evaluation",
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
            strength_levels=[0.5, 1.5, 2.5],
            generation_temperature=1.0,
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
            strength_levels=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
            generation_temperature=1.0,
        )

    def scale_to_depth(self, depth: EvalDepth) -> "EvaluationConfig":
        """Create a scaled copy of this config for a given evaluation depth.

        Scales all prompt counts and strength levels proportionally based on
        the depth's budget percentage. Used by tournament mode to create
        lightweight evaluations for early elimination rounds.

        Args:
            depth: The evaluation depth (QUICK=10%, MEDIUM=40%, FULL=100%).

        Returns:
            New EvaluationConfig with scaled parameters.

        Example:
            >>> full_config = EvaluationConfig(behavior_prompts=50)
            >>> quick_config = full_config.scale_to_depth(EvalDepth.QUICK)
            >>> quick_config.behavior_prompts  # ~5 (10% of 50)
        """
        budget = EVAL_DEPTH_BUDGET[depth]

        if budget >= 1.0:
            return self.model_copy()

        # Scale prompt counts (with minimums to ensure meaningful evaluation)
        scaled_behavior = max(5, int(self.behavior_prompts * budget))
        scaled_specificity = max(5, int(self.specificity_prompts * budget)) if budget >= 0.4 else 0
        scaled_coherence = max(5, int(self.coherence_prompts * budget)) if budget >= 1.0 else 0
        scaled_capability = max(3, int(self.capability_prompts * budget)) if budget >= 1.0 else 0
        scaled_generalization = max(5, int(self.generalization_prompts * budget)) if budget >= 0.4 else 0

        # Scale strength levels - keep subset spread across range
        scaled_strengths = self._scale_strength_levels(budget)

        # For quick/medium, always use 1 generation per prompt (reduce variance estimation)
        scaled_generations = 1 if budget < 1.0 else self.behavior_generations_per_prompt

        # Adjust weights for active dimensions only
        # Quick: behavior only → 100% behavior weight
        # Medium: behavior + specificity + generalization → rebalance
        if budget < 0.4:
            # Quick: behavior only
            weights = {
                "behavior_weight": 1.0,
                "specificity_weight": 0.0,
                "coherence_weight": 0.0,
                "capability_weight": 0.0,
                "generalization_weight": 0.0,
            }
        elif budget < 1.0:
            # Medium: behavior + specificity + generalization
            weights = {
                "behavior_weight": 0.50,
                "specificity_weight": 0.30,
                "coherence_weight": 0.0,
                "capability_weight": 0.0,
                "generalization_weight": 0.20,
            }
        else:
            # Full: use original weights
            weights = {}

        return self.model_copy(update={
            "behavior_prompts": scaled_behavior,
            "behavior_generations_per_prompt": scaled_generations,
            "specificity_prompts": scaled_specificity,
            "coherence_prompts": scaled_coherence,
            "capability_prompts": scaled_capability,
            "generalization_prompts": scaled_generalization,
            "strength_levels": scaled_strengths,
            **weights,
        })

    def _scale_strength_levels(self, budget: float) -> List[float]:
        """Scale strength levels based on budget.

        Selects a subset of strength levels that are spread across the range.
        Always includes 1.0 as the baseline strength.

        Args:
            budget: The budget fraction (0.0 to 1.0).

        Returns:
            List of strength levels to use.
        """
        if budget >= 1.0:
            return self.strength_levels

        # Determine how many strengths to use
        target_count = max(1, int(len(self.strength_levels) * budget))

        if target_count == 1:
            # Just use 1.0 (or closest to it)
            return [min(self.strength_levels, key=lambda x: abs(x - 1.0))]

        if target_count >= len(self.strength_levels):
            return self.strength_levels

        # Select evenly spaced strengths including endpoints
        sorted_strengths = sorted(self.strength_levels)
        indices = [
            int(i * (len(sorted_strengths) - 1) / (target_count - 1))
            for i in range(target_count)
        ]
        selected = [sorted_strengths[i] for i in indices]

        # Ensure 1.0 is included if it exists in original
        if 1.0 in self.strength_levels and 1.0 not in selected:
            # Replace the closest one with 1.0
            closest_idx = min(range(len(selected)), key=lambda i: abs(selected[i] - 1.0))
            selected[closest_idx] = 1.0

        return sorted(selected)

    @property
    def estimated_inferences(self) -> int:
        """Estimate total number of model inferences for this config.

        Useful for comparing evaluation budgets across depths.
        """
        behavior_inferences = (
            self.behavior_prompts *
            len(self.strength_levels) *
            self.behavior_generations_per_prompt
        )
        specificity_inferences = self.specificity_prompts
        coherence_inferences = self.coherence_prompts * len(self.strength_levels)
        capability_inferences = self.capability_prompts * 2  # baseline + steered
        generalization_inferences = self.generalization_prompts

        return (
            behavior_inferences +
            specificity_inferences +
            coherence_inferences +
            capability_inferences +
            generalization_inferences
        )


# ============================================================================
# Master Task Configuration
# ============================================================================


class TaskConfig(BaseModel):
    """Master configuration for an extraction task.

    Uses CAA (Contrastive Activation Addition) for steering vector extraction.
    CAA is fast, reliable, and requires no hyperparameter tuning.

    Example:
        >>> config = TaskConfig(
        ...     target_layers=[15, 16, 17],
        ...     contrast=ContrastConfig.standard(),
        ... )
    """

    # Extraction method
    extraction_method: ExtractionMethod = Field(
        default=ExtractionMethod.CAA,
        description="Extraction method (CAA is recommended and default)",
    )

    # Sample generation
    # For CAA: multiple samples explore different layers and token positions
    # For Gradient: multiple samples explore different random seeds
    num_samples: int = Field(
        default=16,
        ge=1,
        le=100,
        description="Number of extraction samples to run in parallel",
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

    target_layers: Optional[List[int]] = Field(
        default=None,
        description="Explicit target layers (overrides layer_strategies when set)",
    )

    # CAA-specific settings (exploration space for CAA extraction)
    caa: CAAConfig = Field(
        default_factory=CAAConfig,
        description="CAA extraction exploration configuration",
    )

    # Optimization settings (for gradient-based extraction)
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
    # and extraction method. CAA allows much higher concurrency than gradient.
    # For CAA: max 128 concurrent extractions (forward-only, light memory)
    # For Gradient: max 32 concurrent extractions (forward+backward, heavy memory)
    max_concurrent_extractions: int = Field(
        default=128,
        ge=1,
        le=256,
        description="Maximum parallel extraction workers (auto-limited by GPU memory)",
    )

    max_concurrent_evaluations: int = Field(
        default=64,
        ge=1,
        le=128,
        description="Maximum parallel LLM API calls for evaluation",
    )

    # Evaluation settings
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Evaluation configuration",
    )

    # Tournament/elimination settings
    tournament: TournamentConfig = Field(
        default_factory=TournamentConfig,
        description="Tournament elimination configuration (disabled by default)",
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
    generator_model: str = Field(
        default=DEFAULT_MODEL,
        description="Model for contrast pair generation",
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
        """Ensure top_k doesn't exceed effective sample count."""
        effective = self.effective_samples
        if self.top_k > effective:
            self.top_k = effective
        return self

    @property
    def effective_samples(self) -> int:
        """Get effective sample count (tournament initial or num_samples)."""
        if self.tournament.enabled:
            return self.tournament.initial_samples
        return self.num_samples

    @property
    def is_tournament_mode(self) -> bool:
        """Check if tournament mode is enabled."""
        return self.tournament.enabled

    # -------------------------------------------------------------------------
    # Standard Presets (no tournament)
    # -------------------------------------------------------------------------

    @classmethod
    def quick(cls) -> "TaskConfig":
        """Quick configuration for testing (no tournament)."""
        return cls(
            num_samples=4,
            num_seeds=2,
            layer_strategies=[LayerStrategy.AUTO],
            optimization=OptimizationConfig.fast(),
            contrast=ContrastConfig.fast(),
            datapoints_per_sample=25,
            evaluation=EvaluationConfig.fast(),
            top_k=2,
        )

    @classmethod
    def standard(cls) -> "TaskConfig":
        """Standard configuration for normal use (no tournament)."""
        return cls(
            num_samples=16,
            optimization=OptimizationConfig.standard(),
            contrast=ContrastConfig.standard(),
            datapoints_per_sample=50,
        )

    @classmethod
    def comprehensive(cls) -> "TaskConfig":
        """Comprehensive configuration for best results (no tournament)."""
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
            evaluation=EvaluationConfig.thorough(),
            top_k=8,
        )

    # -------------------------------------------------------------------------
    # Tournament Presets (with progressive elimination)
    # -------------------------------------------------------------------------

    @classmethod
    def quick_tournament(cls) -> "TaskConfig":
        """Quick tournament: 32→4 (1 round, 87.5% elimination).

        ~8x more exploration than quick() for similar compute.
        """
        return cls(
            num_samples=4,  # Final survivors
            num_seeds=2,
            layer_strategies=[LayerStrategy.AUTO, LayerStrategy.SWEEP],
            optimization=OptimizationConfig.fast(),
            contrast=ContrastConfig.fast(),
            datapoints_per_sample=40,  # Max for finals
            evaluation=EvaluationConfig.fast(),
            tournament=TournamentConfig.quick(),
            top_k=2,
        )

    @classmethod
    def standard_tournament(cls) -> "TaskConfig":
        """Standard tournament: 256→16 (2 rounds, 75% elimination).

        ~16x more exploration than standard() for similar compute.
        """
        return cls(
            num_samples=16,  # Final survivors
            num_seeds=4,
            layer_strategies=[LayerStrategy.AUTO, LayerStrategy.SWEEP, LayerStrategy.MIDDLE],
            optimization=OptimizationConfig.standard(),
            contrast=ContrastConfig.standard(),
            datapoints_per_sample=60,  # Max for finals
            evaluation=EvaluationConfig.standard(),
            tournament=TournamentConfig.standard(),
            top_k=8,
        )

    @classmethod
    def comprehensive_tournament(cls) -> "TaskConfig":
        """Comprehensive tournament: 1024→32 (3 rounds, 68.5% elimination).

        ~32x more exploration than comprehensive() for similar compute.
        Best quality with thorough elimination.
        """
        return cls(
            num_samples=32,  # Final survivors
            num_seeds=8,
            layer_strategies=[
                LayerStrategy.AUTO,
                LayerStrategy.SWEEP,
                LayerStrategy.MIDDLE,
                LayerStrategy.LATE,
            ],
            optimization=OptimizationConfig.thorough(),
            contrast=ContrastConfig.thorough(),
            datapoints_per_sample=80,  # Max for finals
            evaluation=EvaluationConfig.thorough(),
            tournament=TournamentConfig.comprehensive(),
            top_k=16,
        )

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def with_tournament(
        self,
        elimination_rounds: int = 2,
        final_survivors: Optional[int] = None,
    ) -> "TaskConfig":
        """Return a copy with tournament mode enabled.

        Args:
            elimination_rounds: Number of elimination phases.
            final_survivors: Survivors entering finals (defaults to num_samples).

        Returns:
            New TaskConfig with tournament enabled.
        """
        survivors = final_survivors or self.num_samples
        return self.model_copy(update={
            "tournament": TournamentConfig(
                enabled=True,
                elimination_rounds=elimination_rounds,
                final_survivors=survivors,
            ),
        })

    def without_tournament(self) -> "TaskConfig":
        """Return a copy with tournament mode disabled."""
        return self.model_copy(update={
            "tournament": TournamentConfig(enabled=False),
        })
