"""Territory assigner for distributing seeds to samples.

This module assigns seeds to samples ensuring diversity and coverage
across all samples while avoiding duplication.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List, Set

from vector_forge.contrast.protocols import Seed, TerritoryAssignerProtocol

logger = logging.getLogger(__name__)


class TerritoryAssigner(TerritoryAssignerProtocol):
    """Assigns seeds to samples using stratified distribution.

    Ensures that:
    - Each sample gets diverse seeds (different components, attributes)
    - Different samples get different seed combinations
    - High-quality seeds are distributed fairly
    - Coverage of behavior components is maximized

    Example:
        >>> assigner = TerritoryAssigner()
        >>> assignments = assigner.assign(seeds, num_samples=16, seeds_per_sample=10)
        >>> print(len(assignments[0]))  # 10 seeds for sample 0
    """

    def __init__(
        self,
        ensure_component_coverage: bool = True,
        randomize: bool = True,
    ):
        """Initialize the territory assigner.

        Args:
            ensure_component_coverage: If True, try to cover all components in each sample.
            randomize: If True, add some randomization to assignments.
        """
        self._ensure_coverage = ensure_component_coverage
        self._randomize = randomize

    def assign(
        self,
        seeds: List[Seed],
        num_samples: int,
        seeds_per_sample: int,
    ) -> Dict[int, List[Seed]]:
        """Assign seeds to samples with stratification.

        Args:
            seeds: Available seeds to assign.
            num_samples: Number of samples.
            seeds_per_sample: How many seeds each sample gets.

        Returns:
            Dict mapping sample_id to list of seeds.
        """
        if not seeds:
            logger.warning("No seeds to assign")
            return {i: [] for i in range(num_samples)}

        logger.info(
            f"Assigning {len(seeds)} seeds to {num_samples} samples "
            f"({seeds_per_sample} per sample)"
        )

        # Group seeds by target component for stratification
        component_groups = self._group_by_component(seeds)

        # Sort seeds by quality score
        sorted_seeds = sorted(seeds, key=lambda s: s.quality_score, reverse=True)

        # Assign to each sample
        assignments: Dict[int, List[Seed]] = {}

        for sample_idx in range(num_samples):
            sample_seeds = self._select_for_sample(
                all_seeds=sorted_seeds,
                component_groups=component_groups,
                sample_idx=sample_idx,
                count=seeds_per_sample,
                num_samples=num_samples,
            )
            assignments[sample_idx] = sample_seeds

        # Log distribution stats
        self._log_distribution_stats(assignments, component_groups)

        return assignments

    def _group_by_component(self, seeds: List[Seed]) -> Dict[str, List[Seed]]:
        """Group seeds by their target components."""
        groups: Dict[str, List[Seed]] = defaultdict(list)

        for seed in seeds:
            if seed.target_components:
                for component in seed.target_components:
                    groups[component].append(seed)
            else:
                groups["_general"].append(seed)

        return dict(groups)

    def _select_for_sample(
        self,
        all_seeds: List[Seed],
        component_groups: Dict[str, List[Seed]],
        sample_idx: int,
        count: int,
        num_samples: int,
    ) -> List[Seed]:
        """Select seeds for a specific sample with stratification.

        Uses a combination of:
        1. Component-based stratification (ensure coverage)
        2. Quality-based selection (prefer high-quality seeds)
        3. Offset-based diversity (different samples get different slices)
        """
        selected: List[Seed] = []
        used_indices: Set[int] = set()

        # Create RNG with sample-specific seed for reproducibility
        rng = random.Random(sample_idx * 1000 + 42)

        # Phase 1: Ensure component coverage
        if self._ensure_coverage:
            components = list(component_groups.keys())
            if self._randomize:
                rng.shuffle(components)

            for component in components:
                if len(selected) >= count:
                    break

                component_seeds = component_groups[component]
                if not component_seeds:
                    continue

                # Select with offset based on sample index
                offset = sample_idx % len(component_seeds)

                for i in range(len(component_seeds)):
                    idx = (offset + i) % len(component_seeds)
                    seed = component_seeds[idx]

                    # Find global index
                    try:
                        global_idx = all_seeds.index(seed)
                    except ValueError:
                        continue

                    if global_idx not in used_indices:
                        selected.append(seed)
                        used_indices.add(global_idx)
                        break

        # Phase 2: Fill remaining with quality-based selection
        remaining_needed = count - len(selected)

        if remaining_needed > 0:
            # Get unused seeds, preferring high quality
            available = [
                (i, s) for i, s in enumerate(all_seeds)
                if i not in used_indices
            ]

            if self._randomize:
                # Add some randomization while still preferring quality
                # Take from different parts of the sorted list based on sample_idx
                start_offset = (sample_idx * len(available)) // num_samples
                available = available[start_offset:] + available[:start_offset]

            for global_idx, seed in available:
                if len(selected) >= count:
                    break
                if global_idx not in used_indices:
                    selected.append(seed)
                    used_indices.add(global_idx)

        return selected

    def _log_distribution_stats(
        self,
        assignments: Dict[int, List[Seed]],
        component_groups: Dict[str, List[Seed]],
    ) -> None:
        """Log statistics about the distribution."""
        total_seeds = sum(len(seeds) for seeds in assignments.values())
        avg_per_sample = total_seeds / len(assignments) if assignments else 0

        # Check component coverage per sample
        coverage_stats = []
        all_components = set(component_groups.keys())

        for sample_idx, seeds in assignments.items():
            sample_components = set()
            for seed in seeds:
                sample_components.update(seed.target_components)
            coverage = len(sample_components) / len(all_components) if all_components else 1
            coverage_stats.append(coverage)

        avg_coverage = sum(coverage_stats) / len(coverage_stats) if coverage_stats else 0

        logger.info(
            f"Distribution complete: {total_seeds} total seeds, "
            f"{avg_per_sample:.1f} avg per sample, "
            f"{avg_coverage:.1%} avg component coverage"
        )


class StratifiedTerritoryAssigner(TerritoryAssigner):
    """Extended assigner with attribute-based stratification.

    In addition to component coverage, also ensures diversity in
    seed attributes (emotion, formality, domain, etc.).
    """

    def __init__(
        self,
        attribute_keys: List[str] = None,
        ensure_component_coverage: bool = True,
        randomize: bool = True,
    ):
        """Initialize the stratified assigner.

        Args:
            attribute_keys: Attribute keys to stratify on.
            ensure_component_coverage: If True, try to cover all components.
            randomize: If True, add randomization.
        """
        super().__init__(ensure_component_coverage, randomize)
        self._attribute_keys = attribute_keys or ["emotion", "formality", "domain"]

    def _select_for_sample(
        self,
        all_seeds: List[Seed],
        component_groups: Dict[str, List[Seed]],
        sample_idx: int,
        count: int,
        num_samples: int,
    ) -> List[Seed]:
        """Select with both component and attribute stratification."""
        # First do component-based selection
        base_selection = super()._select_for_sample(
            all_seeds, component_groups, sample_idx, count, num_samples
        )

        # If we have enough, check attribute diversity
        if len(base_selection) >= count:
            return self._ensure_attribute_diversity(base_selection, all_seeds, sample_idx)

        return base_selection

    def _ensure_attribute_diversity(
        self,
        selected: List[Seed],
        all_seeds: List[Seed],
        sample_idx: int,
    ) -> List[Seed]:
        """Swap seeds to improve attribute diversity if needed."""
        rng = random.Random(sample_idx * 1000 + 43)

        # Count attribute values in selection
        attr_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for seed in selected:
            for key in self._attribute_keys:
                value = seed.attributes.get(key, "unknown")
                attr_counts[key][value] += 1

        # Check if any attribute is over-represented
        for key, counts in attr_counts.items():
            max_count = max(counts.values()) if counts else 0
            threshold = len(selected) // 2

            if max_count > threshold:
                # Try to swap some seeds for more diversity
                selected = self._swap_for_diversity(
                    selected, all_seeds, key, counts, rng
                )

        return selected

    def _swap_for_diversity(
        self,
        selected: List[Seed],
        all_seeds: List[Seed],
        attribute_key: str,
        current_counts: Dict[str, int],
        rng: random.Random,
    ) -> List[Seed]:
        """Swap seeds to improve diversity on a specific attribute."""
        # Find the over-represented value
        max_value = max(current_counts.items(), key=lambda x: x[1])[0]

        # Find seeds with this value that could be swapped
        swap_candidates = [
            (i, s) for i, s in enumerate(selected)
            if s.attributes.get(attribute_key) == max_value
        ]

        # Find replacement seeds with different values
        selected_set = set(id(s) for s in selected)
        replacements = [
            s for s in all_seeds
            if id(s) not in selected_set
            and s.attributes.get(attribute_key) != max_value
        ]

        if not swap_candidates or not replacements:
            return selected

        # Swap up to 2 seeds
        rng.shuffle(swap_candidates)
        rng.shuffle(replacements)

        result = list(selected)
        for i in range(min(2, len(swap_candidates), len(replacements))):
            idx, _ = swap_candidates[i]
            result[idx] = replacements[i]

        return result
