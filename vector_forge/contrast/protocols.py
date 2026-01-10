"""Protocols and data classes for contrast generation system.

This module defines the interfaces and data structures used throughout
the contrast generation pipeline, following Interface Segregation Principle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, List, Optional, Dict, Any, Tuple, runtime_checkable


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BehaviorComponent:
    """A distinct component of a behavior.

    Behaviors are decomposed into components to enable targeted
    contrast generation and better isolation of the behavior signal.
    """
    name: str
    description: str
    markers: List[str]          # Phrases/patterns that indicate this component
    opposite_markers: List[str]  # What the opposite looks like

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class NegativeExample:
    """What this behavior is NOT - helps define clean boundaries."""

    similar_behavior: str
    why_different: str
    example: str


@dataclass
class RealisticScenario:
    """A realistic scenario where behavior naturally arises."""

    setup: str
    user_persona: str
    natural_trigger: str
    stakes: str  # low/medium/high


@dataclass
class ConfoundInfo:
    """Information about a confound to control for."""

    factor: str
    difficulty: str  # easy/medium/hard
    strategy: str


@dataclass
class BehaviorAnalysis:
    """Result of analyzing a behavior.

    Contains structured information about the behavior that guides
    seed generation and contrast pair creation.
    """

    behavior_name: str
    description: str
    components: List[BehaviorComponent]
    trigger_conditions: List[str]
    contrast_dimensions: List[str]
    confounds_to_avoid: List[str]  # Simple list for backward compatibility

    # Enhanced fields (may be empty for backward compatibility)
    core_definition: str = ""
    not_this_behavior: List[NegativeExample] = field(default_factory=list)
    realistic_scenarios: List[RealisticScenario] = field(default_factory=list)
    confound_details: List[ConfoundInfo] = field(default_factory=list)

    def get_component(self, name: str) -> Optional[BehaviorComponent]:
        """Get a component by name."""
        for c in self.components:
            if c.name == name:
                return c
        return None

    def get_all_markers(self) -> List[str]:
        """Get all markers from all components."""
        markers = []
        for c in self.components:
            markers.extend(c.markers)
        return markers

    def get_negative_examples_text(self) -> str:
        """Format negative examples as text for prompts."""
        if not self.not_this_behavior:
            return ""
        lines = ["What this behavior is NOT:"]
        for neg in self.not_this_behavior:
            lines.append(f"- {neg.similar_behavior}: {neg.why_different}")
        return "\n".join(lines)

    def get_scenarios_text(self) -> str:
        """Format realistic scenarios as text for prompts."""
        if not self.realistic_scenarios:
            return ""
        lines = ["Realistic scenarios:"]
        for sc in self.realistic_scenarios:
            lines.append(f"- {sc.setup} (user: {sc.user_persona}, stakes: {sc.stakes})")
        return "\n".join(lines)


@dataclass
class Seed:
    """A quality seed for contrast pair generation.

    Seeds are derived from behavior analysis and provide the
    foundation for generating high-quality contrast pairs.
    """
    scenario: str
    context: str
    expected_contrast_strength: float
    target_components: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    example_prompt: str = ""

    def __hash__(self) -> int:
        return hash((self.scenario, self.context))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Seed):
            return False
        return self.scenario == other.scenario and self.context == other.context


@dataclass
class ValidationResult:
    """Result of validating a contrast pair.

    Contains scores from various validation dimensions and
    identifies any issues found.
    """
    is_valid: bool
    dst_behavior_score: float  # 0-10, how strongly dst exhibits behavior
    src_behavior_score: float  # 0-10, should be LOW
    semantic_distance: float   # 0-1, how different dst and src are
    contrast_quality: float    # 0-10, overall contrast quality
    confounds_detected: List[str] = field(default_factory=list)
    reasoning: str = ""

    @property
    def contrast_gap(self) -> float:
        """Gap between dst and src behavior scores."""
        if self.dst_behavior_score < 0 or self.src_behavior_score < 0:
            return -1
        return self.dst_behavior_score - self.src_behavior_score


@dataclass
class ContrastPair:
    """A contrast pair (validated or not).

    Contains a prompt with two completions: dst (exhibits behavior)
    and src (does not exhibit behavior).
    """
    prompt: str
    dst: str  # Exhibits behavior
    src: str  # Does not exhibit behavior
    seed: Optional[Seed] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidatedPair(ContrastPair):
    """A contrast pair that has been validated.

    Extends ContrastPair with validation information.
    """
    validation: Optional[ValidationResult] = None
    attempts: int = 1

    @property
    def is_valid(self) -> bool:
        """Check if the pair passed validation."""
        return self.validation is not None and self.validation.is_valid

    @property
    def contrast_quality(self) -> float:
        """Get the contrast quality score."""
        if self.validation is None:
            return 0.0
        return self.validation.contrast_quality


@dataclass
class SampleDataset:
    """Dataset of validated pairs for a single sample."""
    sample_id: int
    core_pairs: List[ValidatedPair]
    unique_pairs: List[ValidatedPair]

    @property
    def all_pairs(self) -> List[ValidatedPair]:
        """Get all pairs (core + unique)."""
        return self.core_pairs + self.unique_pairs

    @property
    def valid_pairs(self) -> List[ValidatedPair]:
        """Get only valid pairs."""
        return [p for p in self.all_pairs if p.is_valid]

    @property
    def avg_contrast_quality(self) -> float:
        """Average contrast quality of valid pairs."""
        valid = self.valid_pairs
        if not valid:
            return 0.0
        return sum(p.contrast_quality for p in valid) / len(valid)


# ============================================================================
# Protocols (Interfaces)
# ============================================================================

@runtime_checkable
class BehaviorAnalyzerProtocol(Protocol):
    """Protocol for analyzing behaviors.

    Implementations analyze a behavior description to extract
    components, triggers, contrast dimensions, and confounds.
    """

    async def analyze(self, behavior_description: str) -> BehaviorAnalysis:
        """Analyze behavior and return structured analysis.

        Args:
            behavior_description: Natural language description of the behavior.

        Returns:
            BehaviorAnalysis with components, triggers, etc.
        """
        ...


@runtime_checkable
class SeedGeneratorProtocol(Protocol):
    """Protocol for generating quality seeds.

    Implementations generate and score seeds based on behavior analysis.
    """

    async def generate(
        self,
        analysis: BehaviorAnalysis,
        count: int,
    ) -> List[Seed]:
        """Generate quality seeds for the behavior.

        Args:
            analysis: Behavior analysis to base seeds on.
            count: Number of seeds to generate.

        Returns:
            List of quality seeds, sorted by quality score.
        """
        ...

    async def score_seeds(
        self,
        seeds: List[Seed],
        analysis: BehaviorAnalysis,
    ) -> List[Tuple[Seed, float]]:
        """Score seeds by quality.

        Args:
            seeds: Seeds to score.
            analysis: Behavior analysis for context.

        Returns:
            List of (seed, score) tuples, sorted by score descending.
        """
        ...


@runtime_checkable
class ContrastValidatorProtocol(Protocol):
    """Protocol for validating contrast pairs.

    Implementations validate that a pair has sufficient contrast
    and meets quality criteria.
    """

    async def validate(
        self,
        pair: ContrastPair,
        analysis: BehaviorAnalysis,
    ) -> ValidationResult:
        """Validate a contrast pair.

        Args:
            pair: The contrast pair to validate.
            analysis: Behavior analysis for context.

        Returns:
            ValidationResult with scores and validity.
        """
        ...


@runtime_checkable
class PairGeneratorProtocol(Protocol):
    """Protocol for generating contrast pairs.

    Implementations generate contrast pairs from seeds with
    confound control.
    """

    async def generate(
        self,
        seed: Seed,
        analysis: BehaviorAnalysis,
    ) -> ContrastPair:
        """Generate a contrast pair from a seed.

        Args:
            seed: The seed to generate from.
            analysis: Behavior analysis for context.

        Returns:
            ContrastPair with prompt, dst, and src.
        """
        ...


@runtime_checkable
class PairRegeneratorProtocol(Protocol):
    """Protocol for regenerating failed pairs.

    Implementations regenerate pairs that failed validation
    with targeted feedback.
    """

    async def regenerate(
        self,
        pair: ContrastPair,
        validation: ValidationResult,
        analysis: BehaviorAnalysis,
        attempt: int,
    ) -> ContrastPair:
        """Regenerate a pair that failed validation.

        Args:
            pair: The original pair that failed.
            validation: The validation result with failure reasons.
            analysis: Behavior analysis for context.
            attempt: Which regeneration attempt this is (1-indexed).

        Returns:
            New ContrastPair with improvements.
        """
        ...


@runtime_checkable
class TerritoryAssignerProtocol(Protocol):
    """Protocol for assigning seeds to samples.

    Implementations assign seeds to samples ensuring diversity
    and coverage.
    """

    def assign(
        self,
        seeds: List[Seed],
        num_samples: int,
        seeds_per_sample: int,
    ) -> Dict[int, List[Seed]]:
        """Assign seeds to samples.

        Args:
            seeds: Available seeds to assign.
            num_samples: Number of samples.
            seeds_per_sample: How many seeds each sample gets.

        Returns:
            Dict mapping sample_id to list of seeds.
        """
        ...


# ============================================================================
# Utility Functions
# ============================================================================

def merge_validation_results(
    current: ValidationResult,
    new: ValidationResult,
) -> ValidationResult:
    """Merge two validation results, keeping valid values from each.

    Args:
        current: Existing validation result.
        new: New validation result to merge in.

    Returns:
        Merged ValidationResult.
    """
    return ValidationResult(
        is_valid=current.is_valid and new.is_valid,
        dst_behavior_score=(
            new.dst_behavior_score
            if new.dst_behavior_score >= 0
            else current.dst_behavior_score
        ),
        src_behavior_score=(
            new.src_behavior_score
            if new.src_behavior_score >= 0
            else current.src_behavior_score
        ),
        semantic_distance=(
            new.semantic_distance
            if new.semantic_distance >= 0
            else current.semantic_distance
        ),
        contrast_quality=max(current.contrast_quality, new.contrast_quality),
        confounds_detected=current.confounds_detected + new.confounds_detected,
        reasoning=f"{current.reasoning} | {new.reasoning}".strip(" |"),
    )
