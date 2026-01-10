"""Configuration classes for Vector Forge."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from vector_forge.constants import DEFAULT_MODEL


class DatapointStrategyType(str, Enum):
    """Available strategies for generating training datapoints."""

    CONTRASTIVE = "contrastive"  # Generate both dst and src completions
    SINGLE_DIRECTION = "single_direction"  # Generate dst completions only
    ACTIVATION_DIFF = "activation_diff"  # Use activation differences
    LLM_GENERATED = "llm_generated"  # Let LLM decide completions dynamically


class NoiseReductionType(str, Enum):
    """Available strategies for noise reduction."""

    NONE = "none"
    AVERAGING = "averaging"  # Average multiple vectors from different seeds
    PCA = "pca"  # Project to principal components
    ADVERSARIAL = "adversarial"  # Remove anti-behavior projection


class EvaluationDimension(str, Enum):
    """Dimensions for evaluating steering vector quality."""

    BEHAVIOR_STRENGTH = "behavior_strength"  # Does it induce the target behavior?
    COHERENCE = "coherence"  # Is output grammatical and sensible?
    SPECIFICITY = "specificity"  # Does it NOT affect unrelated behaviors?
    ROBUSTNESS = "robustness"  # Consistent across diverse prompts?
    STRENGTH_CALIBRATION = "strength_calibration"  # Does effect scale with strength?


class LLMConfig(BaseModel):
    """
    Configuration for an LLM client.

    Compatible with litellm model strings for easy provider switching.

    Example:
        >>> config = LLMConfig(model="claude-opus-4-5", temperature=0.7)
        >>> config = LLMConfig(model="gpt-5.2")
        >>> config = LLMConfig(model="ollama/llama3", api_base="http://localhost:11434")
    """

    model: str = Field(default=DEFAULT_MODEL, description="Model identifier (litellm format)")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Max tokens (None = provider default)")
    api_base: Optional[str] = Field(default=None, description="Custom API endpoint")
    api_key: Optional[str] = Field(default=None, description="API key (overrides env)")
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class EvaluationBudget(BaseModel):
    """
    Configuration for inference budget during evaluation.

    Controls the tradeoff between evaluation thoroughness and compute cost.
    """

    # Quick evaluation (used during inner iteration loop)
    quick_eval_prompts: int = Field(default=5, ge=1)
    quick_eval_generations_per_prompt: int = Field(default=3, ge=1)
    quick_eval_strength_levels: List[float] = Field(default_factory=lambda: [1.0, 1.5])

    # Thorough evaluation (used by judge)
    thorough_eval_prompts: int = Field(default=30, ge=5)
    thorough_eval_generations_per_prompt: int = Field(default=5, ge=1)
    thorough_eval_strength_levels: List[float] = Field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0]
    )
    baseline_generations_per_prompt: int = Field(default=3, ge=1)

    @classmethod
    def fast(cls) -> "EvaluationBudget":
        """Minimal budget for quick experiments."""
        return cls(
            quick_eval_prompts=3,
            quick_eval_generations_per_prompt=2,
            quick_eval_strength_levels=[1.0],
            thorough_eval_prompts=10,
            thorough_eval_generations_per_prompt=3,
            thorough_eval_strength_levels=[0.5, 1.0, 1.5],
            baseline_generations_per_prompt=2,
        )

    @classmethod
    def standard(cls) -> "EvaluationBudget":
        """Standard budget for normal use."""
        return cls()

    @classmethod
    def thorough(cls) -> "EvaluationBudget":
        """Full budget for production quality."""
        return cls(
            quick_eval_prompts=10,
            quick_eval_generations_per_prompt=5,
            quick_eval_strength_levels=[0.5, 1.0, 1.5],
            thorough_eval_prompts=50,
            thorough_eval_generations_per_prompt=10,
            thorough_eval_strength_levels=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            baseline_generations_per_prompt=5,
        )


class DiversityConfig(BaseModel):
    """Configuration for datapoint diversity verification and enforcement."""

    # Structured sampling
    use_structured_sampling: bool = Field(default=True)
    domains: List[str] = Field(
        default_factory=lambda: ["science", "math", "history", "personal", "technical", "creative"]
    )
    formats: List[str] = Field(
        default_factory=lambda: ["question", "statement", "dialogue", "instruction"]
    )

    # MMR (Maximal Marginal Relevance) selection
    use_mmr_selection: bool = Field(default=True)
    mmr_lambda: float = Field(default=0.5, ge=0, le=1, description="Balance relevance vs diversity")
    candidate_multiplier: int = Field(default=3, ge=1, description="Generate N*multiplier candidates")

    # Contrastive prompting
    use_contrastive_prompting: bool = Field(default=True)
    show_previous_n: int = Field(default=5, ge=0)

    # Temperature variation
    use_temperature_variation: bool = Field(default=True)
    temperatures: List[float] = Field(default_factory=lambda: [0.3, 0.7, 1.0])

    # Verification thresholds
    min_embedding_distance: float = Field(default=0.3, ge=0, le=1)
    max_avg_similarity: float = Field(default=0.7, ge=0, le=1)


class PipelineConfig(BaseModel):
    """
    Master configuration for the extraction pipeline.

    Example:
        >>> config = PipelineConfig(
        ...     extractor_llm=LLMConfig(model="claude-opus-4-5"),
        ...     judge_llm=LLMConfig(model="claude-sonnet-4-5"),
        ...     datapoint_strategy=DatapointStrategyType.CONTRASTIVE,
        ...     num_prompts=20,
        ...     max_iterations=5,
        ... )
    """

    # LLM Configuration
    extractor_llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM for the extractor agent",
    )
    judge_llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM for the judge",
    )

    # Datapoint Generation
    datapoint_strategy: DatapointStrategyType = Field(default=DatapointStrategyType.CONTRASTIVE)
    num_prompts: int = Field(default=10, ge=1)
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)

    # Layer Search
    layers_to_try: Optional[List[int]] = Field(
        default=None,
        description="Specific layers to try. None = auto-select based on model",
    )
    layer_search_iterations: int = Field(default=2, ge=1)

    # Optimization (passed to steering_vectors optimizer)
    optimization_lr: float = Field(default=0.1, gt=0)
    optimization_max_iters: int = Field(default=50, gt=0)
    optimization_coldness: float = Field(default=0.7, gt=0)

    # Evaluation
    evaluation_budget: EvaluationBudget = Field(default_factory=EvaluationBudget)
    evaluation_dimensions: List[EvaluationDimension] = Field(
        default_factory=lambda: [
            EvaluationDimension.BEHAVIOR_STRENGTH,
            EvaluationDimension.COHERENCE,
            EvaluationDimension.SPECIFICITY,
        ]
    )

    # Iteration Control
    max_outer_iterations: int = Field(default=3, ge=1, description="Max judge-driven refinements")
    max_inner_iterations: int = Field(default=5, ge=1, description="Max extractor iterations")
    quality_threshold: float = Field(default=0.7, ge=0, le=1)

    # Noise Reduction
    noise_reduction: NoiseReductionType = Field(default=NoiseReductionType.AVERAGING)
    num_seeds_for_noise: int = Field(default=3, ge=1)

    # Behavior
    auto_mode: bool = Field(default=True, description="Run autonomously vs pause for approval")
    save_checkpoints: bool = Field(default=True)
    checkpoint_dir: Optional[str] = Field(default=None)
