"""Pool manager for core and sample-specific pairs.

This module manages the distribution of contrast pairs between
the shared core pool and sample-specific unique pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vector_forge.contrast.protocols import (
    Seed,
    ValidatedPair,
    SampleDataset,
)

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for the pool manager."""

    core_pool_size: int = 80
    """Number of pairs in the shared core pool."""

    core_seeds_per_sample: int = 40
    """How many core seeds each sample uses."""

    unique_seeds_per_sample: int = 10
    """How many unique seeds each sample generates."""

    @property
    def total_seeds_per_sample(self) -> int:
        """Total seeds used per sample."""
        return self.core_seeds_per_sample + self.unique_seeds_per_sample

    @classmethod
    def for_sample_count(cls, num_samples: int, total_per_sample: int = 50) -> "PoolConfig":
        """Create config based on sample count.

        Args:
            num_samples: Number of samples.
            total_per_sample: Total pairs each sample should have.

        Returns:
            PoolConfig with appropriate settings.
        """
        # More samples = larger core pool, fewer unique per sample
        if num_samples <= 4:
            core_ratio = 0.6
        elif num_samples <= 8:
            core_ratio = 0.7
        elif num_samples <= 16:
            core_ratio = 0.8
        else:
            core_ratio = 0.85

        core_per_sample = int(total_per_sample * core_ratio)
        unique_per_sample = total_per_sample - core_per_sample

        # Core pool should be at least core_per_sample, ideally larger
        core_pool_size = max(core_per_sample, int(core_per_sample * 1.5))

        return cls(
            core_pool_size=core_pool_size,
            core_seeds_per_sample=core_per_sample,
            unique_seeds_per_sample=unique_per_sample,
        )


@dataclass
class Pool:
    """A pool of validated pairs with metadata."""

    pairs: List[ValidatedPair] = field(default_factory=list)
    seeds: List[Seed] = field(default_factory=list)

    def add_pair(self, pair: ValidatedPair) -> None:
        """Add a pair to the pool."""
        self.pairs.append(pair)
        if pair.seed and pair.seed not in self.seeds:
            self.seeds.append(pair.seed)

    def get_pairs_for_seeds(self, seeds: List[Seed]) -> List[ValidatedPair]:
        """Get pairs that were generated from specific seeds."""
        seed_set = set(seeds)
        return [p for p in self.pairs if p.seed in seed_set]

    @property
    def valid_pairs(self) -> List[ValidatedPair]:
        """Get only valid pairs."""
        return [p for p in self.pairs if p.is_valid]

    @property
    def avg_quality(self) -> float:
        """Average contrast quality of valid pairs."""
        valid = self.valid_pairs
        if not valid:
            return 0.0
        return sum(p.contrast_quality for p in valid) / len(valid)

    def get_top_pairs(self, n: int) -> List[ValidatedPair]:
        """Get top N pairs by quality."""
        sorted_pairs = sorted(
            self.valid_pairs,
            key=lambda p: p.contrast_quality,
            reverse=True,
        )
        return sorted_pairs[:n]


class PoolManager:
    """Manages core pool and sample-specific pair distribution.

    Handles:
    - Core pool creation and management
    - Sample-specific pair tracking
    - Quality-based pair selection
    - Statistics and reporting

    Example:
        >>> manager = PoolManager(config)
        >>> manager.set_core_pool(core_pairs)
        >>> manager.add_sample_pairs(0, sample_0_pairs)
        >>> dataset = manager.get_sample_dataset(0)
    """

    def __init__(self, config: Optional[PoolConfig] = None):
        """Initialize the pool manager.

        Args:
            config: Pool configuration.
        """
        self._config = config or PoolConfig()
        self._core_pool = Pool()
        self._sample_pools: Dict[int, Pool] = {}
        self._seed_assignments: Dict[int, List[Seed]] = {}

    @property
    def config(self) -> PoolConfig:
        """Get the pool configuration."""
        return self._config

    @property
    def core_pool(self) -> Pool:
        """Get the core pool."""
        return self._core_pool

    def set_core_pool(self, pairs: List[ValidatedPair]) -> None:
        """Set the core pool pairs.

        Args:
            pairs: Validated pairs for the core pool.
        """
        self._core_pool = Pool(pairs=list(pairs))

        # Extract seeds
        for pair in pairs:
            if pair.seed and pair.seed not in self._core_pool.seeds:
                self._core_pool.seeds.append(pair.seed)

        logger.info(
            f"Core pool set: {len(pairs)} pairs, "
            f"{len(self._core_pool.valid_pairs)} valid, "
            f"avg quality {self._core_pool.avg_quality:.1f}"
        )

    def set_seed_assignments(
        self,
        core_assignments: Dict[int, List[Seed]],
        unique_assignments: Dict[int, List[Seed]],
    ) -> None:
        """Set the seed assignments for all samples.

        Args:
            core_assignments: Core seed assignments per sample.
            unique_assignments: Unique seed assignments per sample.
        """
        for sample_idx in set(core_assignments.keys()) | set(unique_assignments.keys()):
            core = core_assignments.get(sample_idx, [])
            unique = unique_assignments.get(sample_idx, [])
            self._seed_assignments[sample_idx] = core + unique

    def add_sample_pairs(
        self,
        sample_idx: int,
        pairs: List[ValidatedPair],
    ) -> None:
        """Add sample-specific pairs.

        Args:
            sample_idx: Sample index.
            pairs: Validated pairs for this sample.
        """
        if sample_idx not in self._sample_pools:
            self._sample_pools[sample_idx] = Pool()

        for pair in pairs:
            self._sample_pools[sample_idx].add_pair(pair)

        logger.debug(
            f"Sample {sample_idx}: added {len(pairs)} pairs, "
            f"total {len(self._sample_pools[sample_idx].pairs)}"
        )

    def get_sample_dataset(self, sample_idx: int) -> SampleDataset:
        """Get the complete dataset for a sample.

        Args:
            sample_idx: Sample index.

        Returns:
            SampleDataset with core and unique pairs.
        """
        # Get assigned seeds
        assigned_seeds = self._seed_assignments.get(sample_idx, [])

        # Split into core and unique
        core_seeds = set(self._core_pool.seeds)
        sample_core_seeds = [s for s in assigned_seeds if s in core_seeds]
        sample_unique_seeds = [s for s in assigned_seeds if s not in core_seeds]

        # Get core pairs for this sample's seeds
        core_pairs = self._core_pool.get_pairs_for_seeds(sample_core_seeds)

        # Get unique pairs
        sample_pool = self._sample_pools.get(sample_idx, Pool())
        unique_pairs = sample_pool.pairs

        return SampleDataset(
            sample_id=sample_idx,
            core_pairs=core_pairs,
            unique_pairs=unique_pairs,
        )

    def get_all_datasets(self) -> Dict[int, SampleDataset]:
        """Get datasets for all samples.

        Returns:
            Dict mapping sample_id to SampleDataset.
        """
        sample_ids = set(self._seed_assignments.keys()) | set(self._sample_pools.keys())
        return {
            sample_idx: self.get_sample_dataset(sample_idx)
            for sample_idx in sorted(sample_ids)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics.

        Returns:
            Dict with various statistics.
        """
        datasets = self.get_all_datasets()

        total_core = len(self._core_pool.pairs)
        total_unique = sum(len(p.pairs) for p in self._sample_pools.values())

        valid_core = len(self._core_pool.valid_pairs)
        valid_unique = sum(len(p.valid_pairs) for p in self._sample_pools.values())

        avg_qualities = []
        for dataset in datasets.values():
            if dataset.valid_pairs:
                avg_qualities.append(dataset.avg_contrast_quality)

        return {
            "num_samples": len(datasets),
            "core_pool": {
                "total": total_core,
                "valid": valid_core,
                "avg_quality": self._core_pool.avg_quality,
            },
            "unique_pools": {
                "total": total_unique,
                "valid": valid_unique,
            },
            "per_sample": {
                "avg_pairs": sum(len(d.all_pairs) for d in datasets.values()) / len(datasets) if datasets else 0,
                "avg_valid": sum(len(d.valid_pairs) for d in datasets.values()) / len(datasets) if datasets else 0,
                "avg_quality": sum(avg_qualities) / len(avg_qualities) if avg_qualities else 0,
            },
        }
