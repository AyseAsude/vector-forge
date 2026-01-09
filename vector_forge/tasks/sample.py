"""Extraction sample definitions and generators.

Provides the sample abstraction representing a single extraction attempt,
and generators for creating diverse sample sets that explore the strategy space.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Iterator
from itertools import product
import hashlib

from vector_forge.tasks.config import (
    TaskConfig,
    SampleConfig,
    LayerStrategy,
)
from vector_forge.tasks.expander import ExpandedBehavior


@dataclass
class ExtractionSample:
    """A single extraction attempt with specific configuration.

    Represents one point in the strategy space to explore. Multiple samples
    with varied configurations enable finding the optimal approach for
    each behavior.

    Attributes:
        behavior: The expanded behavior specification.
        config: Sample-specific hyperparameters.
        strategy_name: Human-readable name for this strategy combination.
        sample_id: Unique identifier for this sample.
    """

    behavior: ExpandedBehavior
    config: SampleConfig
    strategy_name: str = ""
    sample_id: str = ""

    def __post_init__(self) -> None:
        """Generate strategy name and ID if not provided."""
        if not self.strategy_name:
            self.strategy_name = self._generate_strategy_name()
        if not self.sample_id:
            self.sample_id = self._generate_id()

    def _generate_strategy_name(self) -> str:
        """Create descriptive name from configuration."""
        parts = [
            f"seed{self.config.seed}",
            self.config.layer_strategy.value,
            f"t{self.config.temperature}",
            f"n{self.config.num_datapoints}",
        ]
        if self.config.use_mean_centering:
            parts.append("mc")
        if self.config.bootstrap_ratio < 1.0:
            parts.append(f"bs{int(self.config.bootstrap_ratio * 100)}")
        return "_".join(parts)

    def _generate_id(self) -> str:
        """Create unique identifier from behavior and config."""
        content = f"{self.behavior.name}_{self.strategy_name}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        return f"{self.behavior.name}/{self.strategy_name}"


@dataclass
class SampleSet:
    """Collection of samples for parallel execution.

    Groups samples by strategy type for analysis and provides
    iteration utilities.
    """

    samples: List[ExtractionSample] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[ExtractionSample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> ExtractionSample:
        return self.samples[index]

    def by_strategy(self, strategy: LayerStrategy) -> List[ExtractionSample]:
        """Filter samples by layer strategy."""
        return [
            s for s in self.samples
            if s.config.layer_strategy == strategy
        ]

    def by_seed(self, seed: int) -> List[ExtractionSample]:
        """Filter samples by random seed."""
        return [s for s in self.samples if s.config.seed == seed]

    @property
    def strategies_used(self) -> List[LayerStrategy]:
        """List of unique strategies in this set."""
        return list(set(s.config.layer_strategy for s in self.samples))

    @property
    def seeds_used(self) -> List[int]:
        """List of unique seeds in this set."""
        return sorted(set(s.config.seed for s in self.samples))


class SampleGenerator:
    """Generates diverse extraction samples from task configuration.

    Creates a set of samples that systematically explore the hyperparameter
    space, similar to how Petri generates diverse audit instructions.

    Supports multiple generation strategies:
    - Full grid: Cartesian product of all parameters
    - Smart sampling: Latin hypercube for good coverage with fewer samples
    - Focused: Variations around a base configuration

    Example:
        >>> generator = SampleGenerator(task_config)
        >>> samples = generator.generate_grid(behavior)
        >>> print(f"Generated {len(samples)} samples")
    """

    def __init__(self, config: TaskConfig) -> None:
        """Initialize generator with task configuration.

        Args:
            config: Task configuration with parameter ranges.
        """
        self._config = config

    def generate_grid(self, behavior: ExpandedBehavior) -> SampleSet:
        """Generate samples using full grid search over parameters.

        Creates one sample for each combination of:
        - Seeds (0 to num_seeds-1)
        - Layer strategies
        - Temperatures
        - Datapoint counts

        Args:
            behavior: The expanded behavior to extract.

        Returns:
            SampleSet containing all grid combinations.
        """
        samples = []

        seeds = range(self._config.num_seeds)
        strategies = self._config.layer_strategies
        temps = self._config.temperatures
        counts = self._config.datapoint_counts

        for seed, strategy, temp, count in product(seeds, strategies, temps, counts):
            config = SampleConfig(
                seed=seed,
                layer_strategy=strategy,
                temperature=temp,
                num_datapoints=count,
                use_mean_centering=True,
            )

            sample = ExtractionSample(
                behavior=behavior,
                config=config,
            )
            samples.append(sample)

        return SampleSet(
            samples=samples,
            metadata={
                "generation_method": "grid",
                "behavior_name": behavior.name,
                "total_combinations": len(samples),
            },
        )

    def generate_smart(
        self,
        behavior: ExpandedBehavior,
        n_samples: Optional[int] = None,
    ) -> SampleSet:
        """Generate samples using quasi-random sampling for good coverage.

        Uses Latin Hypercube Sampling to ensure diverse coverage of the
        parameter space with fewer samples than full grid search.

        Args:
            behavior: The expanded behavior to extract.
            n_samples: Number of samples to generate. Defaults to config.num_samples.

        Returns:
            SampleSet with quasi-randomly distributed samples.
        """
        n = n_samples or self._config.num_samples

        try:
            from scipy.stats import qmc

            sampler = qmc.LatinHypercube(d=4)
            points = sampler.random(n=n)
        except ImportError:
            # Fallback to simple random sampling
            import random

            points = [[random.random() for _ in range(4)] for _ in range(n)]

        samples = []
        seeds = list(range(self._config.num_seeds))
        strategies = self._config.layer_strategies
        temps = self._config.temperatures
        counts = self._config.datapoint_counts

        for point in points:
            seed_idx = int(point[0] * len(seeds))
            strat_idx = int(point[1] * len(strategies))
            temp_idx = int(point[2] * len(temps))
            count_idx = int(point[3] * len(counts))

            # Clamp indices to valid range
            seed_idx = min(seed_idx, len(seeds) - 1)
            strat_idx = min(strat_idx, len(strategies) - 1)
            temp_idx = min(temp_idx, len(temps) - 1)
            count_idx = min(count_idx, len(counts) - 1)

            config = SampleConfig(
                seed=seeds[seed_idx],
                layer_strategy=strategies[strat_idx],
                temperature=temps[temp_idx],
                num_datapoints=counts[count_idx],
                use_mean_centering=True,
            )

            sample = ExtractionSample(
                behavior=behavior,
                config=config,
            )
            samples.append(sample)

        return SampleSet(
            samples=samples,
            metadata={
                "generation_method": "smart",
                "behavior_name": behavior.name,
                "requested_samples": n,
            },
        )

    def generate_focused(
        self,
        behavior: ExpandedBehavior,
        base_config: SampleConfig,
        n_variations: int = 8,
    ) -> SampleSet:
        """Generate samples with variations around a base configuration.

        Useful for amplifying a promising strategy by exploring
        nearby configurations.

        Args:
            behavior: The expanded behavior to extract.
            base_config: Base configuration to vary around.
            n_variations: Number of variations to generate.

        Returns:
            SampleSet with variations of the base configuration.
        """
        samples = []

        # Always include base config
        samples.append(ExtractionSample(
            behavior=behavior,
            config=base_config,
        ))

        # Generate variations by changing one parameter at a time
        for i in range(n_variations - 1):
            new_config = SampleConfig(
                seed=base_config.seed + i + 1,
                layer_strategy=base_config.layer_strategy,
                temperature=base_config.temperature,
                num_datapoints=base_config.num_datapoints,
                optimization_iterations=base_config.optimization_iterations,
                learning_rate=base_config.learning_rate,
                use_mean_centering=base_config.use_mean_centering,
                bootstrap_ratio=max(0.5, base_config.bootstrap_ratio - 0.1 * (i % 3)),
            )

            sample = ExtractionSample(
                behavior=behavior,
                config=new_config,
            )
            samples.append(sample)

        return SampleSet(
            samples=samples,
            metadata={
                "generation_method": "focused",
                "behavior_name": behavior.name,
                "base_strategy": base_config.layer_strategy.value,
            },
        )

    def generate_seeded(
        self,
        behavior: ExpandedBehavior,
        strategy: LayerStrategy,
        n_seeds: int = 5,
    ) -> SampleSet:
        """Generate samples for a single strategy with multiple seeds.

        Useful for noise reduction through averaging vectors from
        the same strategy but different random initializations.

        Args:
            behavior: The expanded behavior to extract.
            strategy: The layer strategy to use.
            n_seeds: Number of different seeds to try.

        Returns:
            SampleSet with same strategy, different seeds.
        """
        samples = []

        for seed in range(n_seeds):
            config = SampleConfig(
                seed=seed,
                layer_strategy=strategy,
                temperature=0.7,
                num_datapoints=50,
                use_mean_centering=True,
            )

            sample = ExtractionSample(
                behavior=behavior,
                config=config,
            )
            samples.append(sample)

        return SampleSet(
            samples=samples,
            metadata={
                "generation_method": "seeded",
                "behavior_name": behavior.name,
                "strategy": strategy.value,
                "n_seeds": n_seeds,
            },
        )
