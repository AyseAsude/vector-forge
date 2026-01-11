"""Contrast generation module for Vector Forge.

This module provides a complete system for generating high-quality
contrast pairs for steering vector training. It includes:

- Behavior analysis: Extract components, triggers, and confounds
- Seed generation: Create quality scenarios based on behavior
- Pair generation: Generate contrast pairs with confound control
- Validation: Verify contrast quality using embedding and LLM judges
- Distribution: Manage core pool and sample-specific pairs

Example:
    >>> from vector_forge.contrast import ContrastPipeline, ContrastPipelineConfig
    >>>
    >>> pipeline = ContrastPipeline(
    ...     llm_client=expander_llm,
    ...     generator_llm_client=generator_llm,
    ...     judge_llm_client=judge_llm,
    ...     config=ContrastPipelineConfig.default(),
    ... )
    >>>
    >>> result = await pipeline.run(
    ...     behavior_description="sycophancy - excessive agreement and validation seeking",
    ...     num_samples=16,
    ... )
    >>>
    >>> for sample_id, dataset in result.sample_datasets.items():
    ...     print(f"Sample {sample_id}: {len(dataset.valid_pairs)} valid pairs")
"""

from vector_forge.contrast.protocols import (
    # Data classes
    BehaviorComponent,
    BehaviorAnalysis,
    Seed,
    ValidationResult,
    ContrastPair,
    ValidatedPair,
    SampleDataset,
    # Protocols
    BehaviorAnalyzerProtocol,
    SeedGeneratorProtocol,
    ContrastValidatorProtocol,
    PairGeneratorProtocol,
    PairRegeneratorProtocol,
    TerritoryAssignerProtocol,
)

from vector_forge.contrast.pipeline import (
    ContrastPipeline,
    ContrastPipelineConfig,
    PipelineResult,
)

from vector_forge.contrast.analysis import (
    BehaviorAnalyzer,
    SeedGenerator,
)

from vector_forge.contrast.validation import (
    EmbeddingContrastValidator,
    LLMContrastValidator,
    CompositeContrastValidator,
)

from vector_forge.contrast.generation import (
    ContrastPairGenerator,
    ContrastRegenerator,
)

from vector_forge.contrast.distribution import (
    TerritoryAssigner,
    PoolManager,
)

__all__ = [
    # Main pipeline
    "ContrastPipeline",
    "ContrastPipelineConfig",
    "PipelineResult",
    # Data classes
    "BehaviorComponent",
    "BehaviorAnalysis",
    "Seed",
    "ValidationResult",
    "ContrastPair",
    "ValidatedPair",
    "SampleDataset",
    # Protocols
    "BehaviorAnalyzerProtocol",
    "SeedGeneratorProtocol",
    "ContrastValidatorProtocol",
    "PairGeneratorProtocol",
    "PairRegeneratorProtocol",
    "TerritoryAssignerProtocol",
    # Components
    "BehaviorAnalyzer",
    "SeedGenerator",
    "EmbeddingContrastValidator",
    "LLMContrastValidator",
    "CompositeContrastValidator",
    "ContrastPairGenerator",
    "ContrastRegenerator",
    "TerritoryAssigner",
    "PoolManager",
]
