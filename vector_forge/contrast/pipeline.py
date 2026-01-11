"""Main pipeline for contrast generation.

This module provides the ContrastPipeline facade that orchestrates
the entire contrast generation process, from behavior analysis
through to validated pair distribution.

Event sourcing: All pipeline operations emit events for full reproducibility.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vector_forge.storage.emitter import EventEmitter

from vector_forge.contrast.protocols import (
    BehaviorAnalysis,
    Seed,
    ContrastPair,
    ValidatedPair,
    SampleDataset,
    SignalIntensity,
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


# ============================================================================
# Constants
# ============================================================================

SEMANTIC_SCORE_TO_DISTANCE_FACTOR = 0.7
"""Factor to convert semantic score (0-10) to cosine distance (0-0.7).

This maps the user-facing semantic score (0-10 scale) to actual cosine distance
used by the embedding validator. The mapping is linear:
- Score 10 → Distance 0.7 (highly different embeddings)
- Score 4 → Distance 0.28 (minimum acceptable difference)
- Score 0 → Distance 0.0 (identical embeddings)

The 0.7 maximum was chosen empirically:
- Cosine distances above 0.7 are rare for coherent text pairs
- Distance of ~0.3 corresponds to "clearly different but related" texts
- Distance of ~0.5 corresponds to "substantially different content"

Formula: distance = (score / 10.0) * SEMANTIC_SCORE_TO_DISTANCE_FACTOR
"""


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
    min_semantic_score: float = 4.0
    """Minimum semantic distance score (0-10, maps from distance 0.3 ~ score 4)."""

    min_dimension_score: float = 6.0
    """Minimum dimension check score."""

    min_structural_score: float = 7.0
    """Minimum structural check score."""

    min_contrast_quality: float = 6.0
    """Minimum overall contrast quality."""

    # Regeneration settings
    max_regeneration_attempts: int = 2
    """Maximum attempts to regenerate a failed pair."""

    # Generation settings
    generation_temperature: float = 0.7
    """Temperature for pair generation."""

    # Parallelism
    max_concurrent_generations: int = 5
    """Maximum concurrent pair generations."""

    # Intensity distribution for contrast pairs
    intensity_extreme: float = 0.10
    """Proportion of extreme intensity pairs (maximum signal)."""

    intensity_high: float = 0.20
    """Proportion of high intensity pairs (clear signal)."""

    intensity_medium: float = 0.30
    """Proportion of medium intensity pairs (balanced)."""

    intensity_natural: float = 0.40
    """Proportion of natural intensity pairs (deployment-realistic)."""

    @property
    def intensity_distribution(self) -> Dict[SignalIntensity, float]:
        """Get intensity distribution as a dictionary."""
        return {
            SignalIntensity.EXTREME: self.intensity_extreme,
            SignalIntensity.HIGH: self.intensity_high,
            SignalIntensity.MEDIUM: self.intensity_medium,
            SignalIntensity.NATURAL: self.intensity_natural,
        }

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
        )

    @classmethod
    def thorough(cls) -> "ContrastPipelineConfig":
        """Create thorough configuration for production."""
        return cls(
            core_pool_size=120,
            core_seeds_per_sample=50,
            unique_seeds_per_sample=15,
            max_regeneration_attempts=3,
            min_dimension_score=7.0,
            min_structural_score=8.0,
            min_semantic_score=5.0,
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
        event_emitter: Optional["EventEmitter"] = None,
    ):
        """Initialize the contrast pipeline.

        Args:
            llm_client: LLM client for generation (analysis, seeds, pairs).
            judge_llm_client: LLM client for judging. If None, uses llm_client.
            config: Pipeline configuration.
            event_emitter: Optional event emitter for event sourcing.
        """
        self._llm = llm_client
        self._judge_llm = judge_llm_client or llm_client
        self._config = config or ContrastPipelineConfig.default()
        self._emitter = event_emitter

        # Initialize components (Dependency Injection)
        self._analyzer = BehaviorAnalyzer(llm_client)
        self._seed_generator = SeedGenerator(llm_client)
        self._pair_generator = ContrastPairGenerator(
            llm_client,
            temperature=self._config.generation_temperature,
            max_concurrency=self._config.max_concurrent_generations,
        )
        self._regenerator = ContrastRegenerator(llm_client)
        self._territory_assigner = TerritoryAssigner()
        self._pool_manager = PoolManager(self._config.to_pool_config())

        # Build validator chain
        # Convert semantic score (0-10) to cosine distance for embedding validator
        min_distance = (
            self._config.min_semantic_score / 10.0
        ) * SEMANTIC_SCORE_TO_DISTANCE_FACTOR
        self._validator = CompositeContrastValidator([
            EmbeddingContrastValidator(min_distance=min_distance),
            LLMContrastValidator(
                self._judge_llm,
                min_dimension_score=self._config.min_dimension_score,
                min_structural_score=self._config.min_structural_score,
            ),
        ])

    def _emit(self, method_name: str, **kwargs) -> None:
        """Emit an event if emitter is available."""
        if self._emitter is not None:
            method = getattr(self._emitter, method_name, None)
            if method:
                method(**kwargs)

    def _generate_id(self, prefix: str = "") -> str:
        """Generate a unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}" if prefix else uuid.uuid4().hex[:12]

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
        start_time = time.time()

        logger.info(
            f"Starting contrast pipeline for {num_samples} samples: "
            f"{behavior_description[:50]}..."
        )

        # Emit pipeline started event
        self._emit(
            "emit_pipeline_started",
            behavior_description=behavior_description,
            num_samples=num_samples,
            config={
                "core_pool_size": self._config.core_pool_size,
                "core_seeds_per_sample": self._config.core_seeds_per_sample,
                "unique_seeds_per_sample": self._config.unique_seeds_per_sample,
                "min_semantic_score": self._config.min_semantic_score,
                "min_dimension_score": self._config.min_dimension_score,
                "min_structural_score": self._config.min_structural_score,
                "min_contrast_quality": self._config.min_contrast_quality,
            },
        )

        # Step 1: Analyze behavior
        logger.info("Step 1: Analyzing behavior...")
        analysis = await self._analyzer.analyze(behavior_description)
        logger.info(f"  Found {len(analysis.components)} components")

        # Emit behavior analyzed event
        self._emit(
            "emit_behavior_analyzed",
            behavior_name=behavior_description[:50],
            num_components=len(analysis.components),
            components=[
                {
                    "name": c.name,
                    "description": c.description,
                    "markers": c.markers[:3] if c.markers else [],
                }
                for c in analysis.components
            ],
            trigger_conditions=analysis.trigger_conditions[:5] if analysis.trigger_conditions else [],
            contrast_dimensions=analysis.contrast_dimensions[:5] if analysis.contrast_dimensions else [],
        )

        # Step 2: Generate quality seeds
        logger.info("Step 2: Generating quality seeds...")
        total_seeds_needed = (
            self._config.core_pool_size +
            self._config.unique_seeds_per_sample * num_samples
        )

        # Emit seed generation started
        self._emit(
            "emit_seed_generation_started",
            num_seeds_requested=total_seeds_needed,
            behavior_name=behavior_description[:50],
        )

        all_seeds = await self._seed_generator.generate(
            analysis,
            count=total_seeds_needed,
        )
        logger.info(f"  Generated {len(all_seeds)} quality seeds")

        # Emit each seed generated
        for i, seed in enumerate(all_seeds):
            is_core = i < self._config.core_pool_size
            self._emit(
                "emit_seed_generated",
                seed_id=self._generate_id("seed"),
                scenario=seed.scenario[:200] if seed.scenario else "",
                context=seed.context[:200] if seed.context else "",
                quality_score=seed.quality_score or 0.0,
                is_core=is_core,
            )

        # Emit seed generation completed
        avg_quality = (
            sum(s.quality_score or 0.0 for s in all_seeds) / len(all_seeds)
            if all_seeds else 0.0
        )
        self._emit(
            "emit_seed_generation_completed",
            total_generated=len(all_seeds),
            avg_quality=avg_quality,
        )

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
            sample_idx=-1,  # -1 indicates core pool
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

        # Emit seed assignments
        for sample_idx in range(num_samples):
            core_count = len(core_assignments.get(sample_idx, []))
            unique_count = len(unique_assignments.get(sample_idx, []))
            self._emit(
                "emit_seeds_assigned",
                sample_idx=sample_idx,
                num_core_seeds=core_count,
                num_unique_seeds=unique_count,
                seed_ids=[],  # Would need to track seed IDs properly
            )

        # Step 6: Generate sample-specific pairs
        logger.info("Step 5: Generating sample-specific pairs...")
        for sample_idx in range(num_samples):
            sample_unique_seeds = unique_assignments.get(sample_idx, [])

            if sample_unique_seeds:
                sample_pairs = await self._generate_validated_pairs(
                    seeds=sample_unique_seeds,
                    analysis=analysis,
                    sample_idx=sample_idx,
                )
                self._pool_manager.add_sample_pairs(sample_idx, sample_pairs)

            logger.debug(f"  Sample {sample_idx}: generated unique pairs")

        # Step 7: Get final datasets
        logger.info("Step 6: Building sample datasets...")
        sample_datasets = self._pool_manager.get_all_datasets()
        statistics = self._pool_manager.get_statistics()

        duration_seconds = time.time() - start_time

        logger.info(
            f"Pipeline complete: {len(sample_datasets)} samples, "
            f"avg {statistics['per_sample']['avg_valid']:.1f} valid pairs/sample, "
            f"avg quality {statistics['per_sample']['avg_quality']:.1f}"
        )

        # Emit pipeline completed
        total_pairs = sum(len(d.all_pairs) for d in sample_datasets.values())
        total_valid = sum(len(d.valid_pairs) for d in sample_datasets.values())
        avg_qual = statistics['per_sample']['avg_quality'] if 'per_sample' in statistics else 0.0

        self._emit(
            "emit_pipeline_completed",
            num_samples=len(sample_datasets),
            total_pairs_generated=total_pairs,
            total_valid_pairs=total_valid,
            avg_quality=avg_qual,
            duration_seconds=duration_seconds,
        )

        # Debug: Log detailed dataset info
        for sample_idx, dataset in sample_datasets.items():
            logger.debug(
                f"Sample {sample_idx}: {len(dataset.core_pairs)} core + "
                f"{len(dataset.unique_pairs)} unique = {len(dataset.all_pairs)} total, "
                f"{len(dataset.valid_pairs)} valid"
            )
            if len(dataset.valid_pairs) == 0 and dataset.all_pairs:
                # Log why pairs are invalid
                for pair in dataset.all_pairs[:3]:  # First 3
                    if pair.validation:
                        logger.warning(
                            f"  Invalid pair: weakest={pair.validation.weakest_dimension}, "
                            f"quality={pair.validation.contrast_quality:.1f}, "
                            f"reason={pair.validation.get_improvement_guidance()}"
                        )

        return PipelineResult(
            behavior_analysis=analysis,
            sample_datasets=sample_datasets,
            statistics=statistics,
        )

    def _assign_intensities(self, seeds: List[Seed]) -> List[SignalIntensity]:
        """Assign intensities to seeds based on distribution config.

        Args:
            seeds: Seeds to assign intensities to.

        Returns:
            List of intensities, one per seed.
        """
        n_seeds = len(seeds)
        if n_seeds == 0:
            return []

        distribution = self._config.intensity_distribution
        intensities: List[SignalIntensity] = []

        # Calculate counts for each intensity
        remaining = n_seeds
        intensity_counts: Dict[SignalIntensity, int] = {}

        for intensity, ratio in distribution.items():
            count = int(n_seeds * ratio)
            intensity_counts[intensity] = count
            remaining -= count

        # Distribute remaining to the highest ratio (typically NATURAL)
        if remaining > 0:
            max_intensity = max(distribution, key=distribution.get)
            intensity_counts[max_intensity] += remaining

        # Build the intensity list
        for intensity, count in intensity_counts.items():
            intensities.extend([intensity] * count)

        return intensities

    async def _generate_validated_pairs(
        self,
        seeds: List[Seed],
        analysis: BehaviorAnalysis,
        sample_idx: int = -1,
    ) -> List[ValidatedPair]:
        """Generate and validate pairs for a list of seeds.

        Uses semaphore to limit concurrent generations.
        Distributes intensities across seeds based on config.

        Args:
            seeds: Seeds to generate pairs from.
            analysis: Behavior analysis.
            sample_idx: Sample index (-1 for core pool).
        """
        semaphore = asyncio.Semaphore(self._config.max_concurrent_generations)

        # Assign intensities to seeds based on distribution
        intensities = self._assign_intensities(seeds)

        async def process_seed(seed: Seed, intensity: SignalIntensity) -> ValidatedPair:
            async with semaphore:
                pair = await self._pair_generator.generate(seed, analysis, intensity)

                # Emit pair generated event
                pair_id = self._generate_id("pair")
                self._emit(
                    "emit_pair_generated",
                    pair_id=pair_id,
                    seed_id=self._generate_id("seed"),  # Would ideally track actual seed ID
                    prompt=pair.prompt[:500] if pair.prompt else "",
                    dst_response=pair.dst[:500] if pair.dst else "",
                    src_response=pair.src[:500] if pair.src else "",
                    sample_idx=sample_idx,
                )

                validated = await self._validate_with_retry(pair, analysis, pair_id)
                return validated

        tasks = [process_seed(seed, intensity) for seed, intensity in zip(seeds, intensities)]
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
        pair_id: str = "",
    ) -> ValidatedPair:
        """Validate a pair, regenerating if needed."""
        current_pair = pair
        pair_id = pair_id or self._generate_id("pair")

        for attempt in range(self._config.max_regeneration_attempts + 1):
            validation = await self._validator.validate(current_pair, analysis)

            # Emit validation event
            rejection_reason = None
            if not validation.is_valid:
                rejection_reason = validation.get_improvement_guidance()

            self._emit(
                "emit_pair_validated",
                pair_id=pair_id,
                is_valid=validation.is_valid,
                contrast_quality=validation.contrast_quality,
                dimension_score=validation.dimension_score,
                marker_score=validation.marker_score,
                boundary_score=validation.boundary_score,
                intensity_score=validation.intensity_score,
                structural_score=validation.structural_score,
                semantic_score=validation.semantic_score,
                weakest_dimension=validation.weakest_dimension if not validation.is_valid else None,
                rejection_reason=rejection_reason,
            )

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
                    f"weakest={validation.weakest_dimension}, "
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
