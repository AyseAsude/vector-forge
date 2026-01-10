"""Main pipeline for contrast generation.

This module provides the ContrastPipeline facade that orchestrates
the entire contrast generation process, from behavior analysis
through to validated pair distribution.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from vector_forge.contrast.protocols import (
    BehaviorAnalysis,
    Seed,
    ContrastPair,
    ValidatedPair,
    SampleDataset,
)
from vector_forge.contrast.analysis.behavior_analyzer import BehaviorAnalyzer
from vector_forge.contrast.analysis.seed_generator import SeedGenerator
from vector_forge.contrast.validation.embedding_validator import EmbeddingContrastValidator
from vector_forge.contrast.validation.llm_validator import LLMContrastValidator
from vector_forge.contrast.validation.composite_validator import CompositeContrastValidator
from vector_forge.contrast.generation.pair_generator import ContrastPairGenerator
from vector_forge.contrast.generation.regenerator import ContrastRegenerator
from vector_forge.contrast.distribution.territory_assigner import TerritoryAssigner
from vector_forge.contrast.distribution.pool_manager import PoolManager, PoolConfig
from vector_forge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass
class ContrastPipelineConfig:
    """Configuration for the contrast generation pipeline."""

    # Pool settings
    core_pool_size: int = 80
    """Number of pairs in the shared core pool."""

    core_seeds_per_sample: int = 40
    """How many core seeds each sample uses."""

    unique_seeds_per_sample: int = 10
    """How many unique seeds each sample generates."""

    # Validation settings
    min_semantic_distance: float = 0.3
    """Minimum semantic distance between dst and src."""

    min_dst_score: float = 7.0
    """Minimum behavior score for dst."""

    max_src_score: float = 3.0
    """Maximum behavior score for src."""

    min_contrast_quality: float = 6.0
    """Minimum overall contrast quality."""

    # Regeneration settings
    max_regeneration_attempts: int = 2
    """Maximum attempts to regenerate a failed pair."""

    # Seed quality
    min_seed_quality: float = 6.0
    """Minimum quality score for seeds."""

    # Generation settings
    generation_temperature: float = 0.7
    """Temperature for pair generation."""

    # Parallelism
    max_concurrent_generations: int = 5
    """Maximum concurrent pair generations."""

    @classmethod
    def default(cls) -> "ContrastPipelineConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def fast(cls) -> "ContrastPipelineConfig":
        """Create fast configuration for testing."""
        return cls(
            core_pool_size=30,
            core_seeds_per_sample=20,
            unique_seeds_per_sample=5,
            max_regeneration_attempts=1,
            min_seed_quality=5.0,
        )

    @classmethod
    def thorough(cls) -> "ContrastPipelineConfig":
        """Create thorough configuration for production."""
        return cls(
            core_pool_size=120,
            core_seeds_per_sample=50,
            unique_seeds_per_sample=15,
            max_regeneration_attempts=3,
            min_seed_quality=7.0,
            min_dst_score=8.0,
            max_src_score=2.0,
            min_contrast_quality=7.0,
        )

    def to_pool_config(self) -> PoolConfig:
        """Convert to PoolConfig."""
        return PoolConfig(
            core_pool_size=self.core_pool_size,
            core_seeds_per_sample=self.core_seeds_per_sample,
            unique_seeds_per_sample=self.unique_seeds_per_sample,
        )


@dataclass
class PipelineResult:
    """Result from running the contrast pipeline."""

    behavior_analysis: BehaviorAnalysis
    """The behavior analysis."""

    sample_datasets: Dict[int, SampleDataset]
    """Datasets for each sample."""

    statistics: Dict[str, any] = field(default_factory=dict)
    """Pipeline statistics."""

    @property
    def num_samples(self) -> int:
        """Number of samples."""
        return len(self.sample_datasets)

    @property
    def total_pairs(self) -> int:
        """Total number of pairs across all samples."""
        return sum(len(d.all_pairs) for d in self.sample_datasets.values())

    @property
    def total_valid_pairs(self) -> int:
        """Total number of valid pairs."""
        return sum(len(d.valid_pairs) for d in self.sample_datasets.values())

    @property
    def avg_quality(self) -> float:
        """Average contrast quality across all samples."""
        qualities = [d.avg_contrast_quality for d in self.sample_datasets.values()]
        return sum(qualities) / len(qualities) if qualities else 0.0


class ContrastPipeline:
    """Main pipeline for generating validated contrast pairs.

    Orchestrates the entire process:
    1. Behavior analysis - understand the target behavior
    2. Seed generation - create quality scenarios
    3. Territory assignment - distribute seeds to samples
    4. Pair generation - create contrast pairs with confound control
    5. Validation - verify contrast quality
    6. Regeneration - fix failed pairs
    7. Distribution - organize into sample datasets

    Example:
        >>> pipeline = ContrastPipeline(
        ...     llm_client=extractor_llm,
        ...     judge_llm_client=judge_llm,
        ... )
        >>> result = await pipeline.run(
        ...     behavior_description="sycophancy",
        ...     num_samples=16,
        ... )
        >>> for sample_id, dataset in result.sample_datasets.items():
        ...     print(f"Sample {sample_id}: {len(dataset.valid_pairs)} valid pairs")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        judge_llm_client: Optional[BaseLLMClient] = None,
        config: Optional[ContrastPipelineConfig] = None,
    ):
        """Initialize the contrast pipeline.

        Args:
            llm_client: LLM client for generation (analysis, seeds, pairs).
            judge_llm_client: LLM client for judging. If None, uses llm_client.
            config: Pipeline configuration.
        """
        self._llm = llm_client
        self._judge_llm = judge_llm_client or llm_client
        self._config = config or ContrastPipelineConfig.default()

        # Initialize components (Dependency Injection)
        self._analyzer = BehaviorAnalyzer(llm_client)
        self._seed_generator = SeedGenerator(
            llm_client,
            min_quality_score=self._config.min_seed_quality,
        )
        self._pair_generator = ContrastPairGenerator(
            llm_client,
            temperature=self._config.generation_temperature,
        )
        self._regenerator = ContrastRegenerator(llm_client)
        self._territory_assigner = TerritoryAssigner()
        self._pool_manager = PoolManager(self._config.to_pool_config())

        # Build validator chain
        self._validator = CompositeContrastValidator([
            EmbeddingContrastValidator(
                min_distance=self._config.min_semantic_distance,
            ),
            LLMContrastValidator(
                self._judge_llm,
                min_dst_score=self._config.min_dst_score,
                max_src_score=self._config.max_src_score,
                min_contrast_quality=self._config.min_contrast_quality,
            ),
        ])

    async def run(
        self,
        behavior_description: str,
        num_samples: int,
    ) -> PipelineResult:
        """Run the complete contrast generation pipeline.

        Args:
            behavior_description: Description of the target behavior.
            num_samples: Number of samples to generate data for.

        Returns:
            PipelineResult with all sample datasets.
        """
        logger.info(
            f"Starting contrast pipeline for {num_samples} samples: "
            f"{behavior_description[:50]}..."
        )

        # Step 1: Analyze behavior
        logger.info("Step 1: Analyzing behavior...")
        analysis = await self._analyzer.analyze(behavior_description)
        logger.info(f"  Found {len(analysis.components)} components")

        # Step 2: Generate quality seeds
        logger.info("Step 2: Generating quality seeds...")
        total_seeds_needed = (
            self._config.core_pool_size +
            self._config.unique_seeds_per_sample * num_samples
        )
        all_seeds = await self._seed_generator.generate(
            analysis,
            count=total_seeds_needed,
        )
        logger.info(f"  Generated {len(all_seeds)} quality seeds")

        # Step 3: Split seeds into core and unique pools
        core_seeds = all_seeds[:self._config.core_pool_size]
        unique_seeds = all_seeds[self._config.core_pool_size:]

        logger.info(
            f"  Core pool: {len(core_seeds)} seeds, "
            f"Unique pool: {len(unique_seeds)} seeds"
        )

        # Step 4: Generate and validate core pool pairs
        logger.info("Step 3: Generating core pool pairs...")
        core_pairs = await self._generate_validated_pairs(
            seeds=core_seeds,
            analysis=analysis,
        )
        self._pool_manager.set_core_pool(core_pairs)
        logger.info(
            f"  Core pool: {len(core_pairs)} pairs, "
            f"{len([p for p in core_pairs if p.is_valid])} valid"
        )

        # Step 5: Assign seeds to samples
        logger.info("Step 4: Assigning seeds to samples...")
        core_assignments = self._territory_assigner.assign(
            seeds=core_seeds,
            num_samples=num_samples,
            seeds_per_sample=self._config.core_seeds_per_sample,
        )
        unique_assignments = self._territory_assigner.assign(
            seeds=unique_seeds,
            num_samples=num_samples,
            seeds_per_sample=self._config.unique_seeds_per_sample,
        )
        self._pool_manager.set_seed_assignments(core_assignments, unique_assignments)

        # Step 6: Generate sample-specific pairs
        logger.info("Step 5: Generating sample-specific pairs...")
        for sample_idx in range(num_samples):
            sample_unique_seeds = unique_assignments.get(sample_idx, [])

            if sample_unique_seeds:
                sample_pairs = await self._generate_validated_pairs(
                    seeds=sample_unique_seeds,
                    analysis=analysis,
                )
                self._pool_manager.add_sample_pairs(sample_idx, sample_pairs)

            logger.debug(f"  Sample {sample_idx}: generated unique pairs")

        # Step 7: Get final datasets
        logger.info("Step 6: Building sample datasets...")
        sample_datasets = self._pool_manager.get_all_datasets()
        statistics = self._pool_manager.get_statistics()

        logger.info(
            f"Pipeline complete: {len(sample_datasets)} samples, "
            f"avg {statistics['per_sample']['avg_valid']:.1f} valid pairs/sample, "
            f"avg quality {statistics['per_sample']['avg_quality']:.1f}"
        )

        return PipelineResult(
            behavior_analysis=analysis,
            sample_datasets=sample_datasets,
            statistics=statistics,
        )

    async def _generate_validated_pairs(
        self,
        seeds: List[Seed],
        analysis: BehaviorAnalysis,
    ) -> List[ValidatedPair]:
        """Generate and validate pairs for a list of seeds.

        Uses semaphore to limit concurrent generations.
        """
        semaphore = asyncio.Semaphore(self._config.max_concurrent_generations)

        async def process_seed(seed: Seed) -> ValidatedPair:
            async with semaphore:
                pair = await self._pair_generator.generate(seed, analysis)
                validated = await self._validate_with_retry(pair, analysis)
                return validated

        tasks = [process_seed(seed) for seed in seeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        validated_pairs = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Pair generation failed: {result}")
            else:
                validated_pairs.append(result)

        return validated_pairs

    async def _validate_with_retry(
        self,
        pair: ContrastPair,
        analysis: BehaviorAnalysis,
    ) -> ValidatedPair:
        """Validate a pair, regenerating if needed."""
        current_pair = pair

        for attempt in range(self._config.max_regeneration_attempts + 1):
            validation = await self._validator.validate(current_pair, analysis)

            if validation.is_valid:
                return ValidatedPair(
                    prompt=current_pair.prompt,
                    dst=current_pair.dst,
                    src=current_pair.src,
                    seed=current_pair.seed,
                    metadata=current_pair.metadata,
                    validation=validation,
                    attempts=attempt + 1,
                )

            # Regenerate if we have attempts left
            if attempt < self._config.max_regeneration_attempts:
                logger.debug(
                    f"Regenerating pair (attempt {attempt + 1}): "
                    f"dst={validation.dst_behavior_score:.1f}, "
                    f"src={validation.src_behavior_score:.1f}, "
                    f"quality={validation.contrast_quality:.1f}"
                )
                current_pair = await self._regenerator.regenerate(
                    current_pair,
                    validation,
                    analysis,
                    attempt + 1,
                )

        # Return best effort even if not fully valid
        return ValidatedPair(
            prompt=current_pair.prompt,
            dst=current_pair.dst,
            src=current_pair.src,
            seed=current_pair.seed,
            metadata=current_pair.metadata,
            validation=validation,
            attempts=self._config.max_regeneration_attempts + 1,
        )

    # Convenience methods for accessing components

    @property
    def analyzer(self) -> BehaviorAnalyzer:
        """Get the behavior analyzer."""
        return self._analyzer

    @property
    def seed_generator(self) -> SeedGenerator:
        """Get the seed generator."""
        return self._seed_generator

    @property
    def validator(self) -> CompositeContrastValidator:
        """Get the validator chain."""
        return self._validator

    @property
    def pool_manager(self) -> PoolManager:
        """Get the pool manager."""
        return self._pool_manager
