"""Protocols and data classes for contrast generation system.

This module defines the interfaces and data structures used throughout
the contrast generation pipeline, following Interface Segregation Principle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, List, Optional, Dict, Any, Tuple, runtime_checkable


# ============================================================================
# Enums
# ============================================================================

class SignalIntensity(str, Enum):
    """Signal intensity levels for contrast pairs."""
    EXTREME = "extreme"  # Maximum contrast, establishes direction
    HIGH = "high"        # Clear signal, still plausible
    MEDIUM = "medium"    # Balanced contrast
    NATURAL = "natural"  # Subtle, realistic deployment


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BehavioralTest:
    """The test that reveals a behavior - the core of extraction.

    This defines HOW to test for the behavior, which drives
    all scenario generation and pair creation.
    """
    description: str                    # Full description of the test
    user_action: str                    # What user does to trigger the test
    model_choice: str                   # What decision the model faces
    distinguishing_variable: str        # THE thing that differs (critical!)
    present_response_pattern: str       # What model does if behavior present
    absent_response_pattern: str        # What model does if behavior absent

    def format_for_prompt(self) -> str:
        """Format behavioral test for inclusion in prompts."""
        return f"""The behavioral test for this behavior:
- User action: {self.user_action}
- Model faces choice about: {self.model_choice}
- The distinguishing variable: {self.distinguishing_variable}
- If behavior PRESENT: {self.present_response_pattern}
- If behavior ABSENT: {self.absent_response_pattern}"""


@dataclass
class IntensityCalibration:
    """How this specific behavior manifests at different intensities.

    Each behavior has its own intensity calibration that defines
    what extreme vs natural looks like for THAT behavior.
    """
    extreme_looks_like: str   # Maximum expression of the behavior
    high_looks_like: str      # Clear but not cartoonish
    medium_looks_like: str    # Balanced expression
    natural_looks_like: str   # Subtle, deployment-realistic

    def get_for_intensity(self, intensity: SignalIntensity) -> str:
        """Get calibration for a specific intensity level."""
        mapping = {
            SignalIntensity.EXTREME: self.extreme_looks_like,
            SignalIntensity.HIGH: self.high_looks_like,
            SignalIntensity.MEDIUM: self.medium_looks_like,
            SignalIntensity.NATURAL: self.natural_looks_like,
        }
        return mapping[intensity]


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
    seed generation and contrast pair creation. The behavioral_test
    and intensity_calibration are the key drivers for composition.
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

    # NEW: Behavioral test - the core of extraction
    behavioral_test: Optional[BehavioralTest] = None

    # NEW: Intensity calibration - how behavior manifests at different levels
    intensity_calibration: Optional[IntensityCalibration] = None

    # NEW: Explicit markers for generation/validation
    presence_markers: List[str] = field(default_factory=list)  # Signs behavior IS present
    absence_markers: List[str] = field(default_factory=list)   # Signs behavior is NOT present

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

    def get_all_presence_markers(self) -> List[str]:
        """Get all presence markers (explicit + component markers)."""
        markers = list(self.presence_markers)
        for c in self.components:
            markers.extend(c.markers)
        return markers

    def get_all_absence_markers(self) -> List[str]:
        """Get all absence markers (explicit + component opposite markers)."""
        markers = list(self.absence_markers)
        for c in self.components:
            markers.extend(c.opposite_markers)
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

    def get_behavioral_test_text(self) -> str:
        """Format behavioral test for prompts."""
        if not self.behavioral_test:
            return ""
        return self.behavioral_test.format_for_prompt()

    def get_intensity_guidance(self, intensity: SignalIntensity) -> str:
        """Get intensity-specific guidance for this behavior."""
        if not self.intensity_calibration:
            return f"Generate at {intensity.value} intensity level."
        return self.intensity_calibration.get_for_intensity(intensity)


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
    # NEW: Suggested intensity for this seed (can be overridden)
    suggested_intensity: Optional[SignalIntensity] = None

    def __hash__(self) -> int:
        return hash((self.scenario, self.context))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Seed):
            return False
        return self.scenario == other.scenario and self.context == other.context


@dataclass
class DimensionCheckResult:
    """Result of checking if contrast is on the correct dimension."""
    score: float  # 0-10
    is_correct: bool
    what_differs: str  # What actually differs between dst and src
    expected_variable: str  # The distinguishing variable we expected
    dst_pattern_match: bool  # Does dst match present_response_pattern?
    src_pattern_match: bool  # Does src match absent_response_pattern?
    explanation: str = ""


@dataclass
class MarkerCheckResult:
    """Result of checking if markers appear correctly."""
    score: float  # 0-10
    presence_markers_found: List[str] = field(default_factory=list)  # Found in dst
    presence_markers_missing: List[str] = field(default_factory=list)  # Expected but not in dst
    absence_markers_found: List[str] = field(default_factory=list)  # Found in src
    absence_markers_missing: List[str] = field(default_factory=list)  # Expected but not in src
    explanation: str = ""


@dataclass
class BoundaryCheckResult:
    """Result of checking if we captured the right behavior."""
    score: float  # 0-10
    is_correct_behavior: bool
    confused_with: Optional[str] = None  # Similar behavior it might be confused with
    explanation: str = ""


@dataclass
class IntensityCheckResult:
    """Result of checking if intensity matches target."""
    score: float  # 0-10
    target_intensity: str  # What was requested
    actual_intensity: str  # What was detected
    matches_calibration: bool
    calibration_description: str = ""  # What target intensity should look like
    explanation: str = ""


@dataclass
class ValidationThresholds:
    """Thresholds for validation dimensions.

    This is the single source of truth for validation thresholds,
    used by both LLMContrastValidator and ValidationComposer to ensure
    consistency between code logic and prompt text.

    Threshold Rationale:
    - dimension (6.0): Must be on correct variable. Lower threshold because
      LLM judges tend to be conservative with high scores.
    - structural (7.0): Higher because malformed responses inject noise.
    - marker (5.0): Lower because markers are correlates, not causes.
      The behavioral test is the ground truth.
    - boundary (5.0): Lower because some overlap is acceptable if dimension is correct.
    """
    dimension: float = 6.0
    """Minimum score for dimension check (contrast on correct variable)."""

    structural: float = 7.0
    """Minimum score for structural check (well-formed responses)."""

    marker: float = 5.0
    """Minimum score for marker check (if markers available)."""

    boundary: float = 5.0
    """Minimum score for boundary check (if boundaries available)."""

    def format_for_prompt(self) -> str:
        """Format thresholds for inclusion in validation prompts."""
        return f"""**Validity Criteria:**
- dimension_check.score >= {self.dimension:.0f} (Critical)
- structural_check.score >= {self.structural:.0f} (Critical)
- If markers available: marker_check.score >= {self.marker:.0f}
- If boundaries available: boundary_check.score >= {self.boundary:.0f}
- intensity_check affects quality score but doesn't gate validity"""


# Default thresholds instance
DEFAULT_VALIDATION_THRESHOLDS = ValidationThresholds()


@dataclass
class StructuralCheckResult:
    """Result of checking structural quality."""
    score: float  # 0-10
    dst_wellformed: bool
    src_wellformed: bool
    dst_complete: bool
    src_complete: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class BehavioralSignalResult:
    """Result of measuring behavioral signal strength."""
    score: float  # Overall signal strength 0-10
    dst_strength: float  # How strongly dst exhibits behavior (0-10)
    src_strength: float  # How clearly src doesn't exhibit (0-10)
    reasoning: str = ""


@dataclass
class ConfoundCheckResult:
    """Result of checking confound control."""
    score: float  # Overall confound control score 0-10
    length_ratio: float  # dst_words / src_words (ideal: 0.8-1.2)
    formality_match: bool
    helpfulness_match: bool
    detail_match: bool
    structure_match: bool
    main_confound: str = ""  # Primary confound issue if any
    reasoning: str = ""


@dataclass
class ValidationResult:
    """Result of validating a contrast pair.

    Contains scores from validation dimensions with detailed results.
    All scores use -1.0 to indicate "not evaluated".
    """
    is_valid: bool
    contrast_quality: float  # 0-10, overall contrast quality
    reasoning: str = ""

    # Dimension-specific scores (0-10, -1 = not evaluated)
    dimension_score: float = -1.0  # Is contrast on right variable?
    marker_score: float = -1.0     # Do markers appear correctly?
    boundary_score: float = -1.0   # Is it the right behavior?
    intensity_score: float = -1.0  # Does it match target intensity?
    structural_score: float = -1.0 # Well-formed, complete?
    semantic_score: float = -1.0   # Semantic distance (from embedding)

    # NEW: Signal quality scores
    behavioral_signal_score: float = -1.0  # How strongly pair contrasts on behavior
    confound_score: float = -1.0           # How well confounds are controlled

    # Detailed results from each check (optional)
    dimension_details: Optional[DimensionCheckResult] = None
    marker_details: Optional[MarkerCheckResult] = None
    boundary_details: Optional[BoundaryCheckResult] = None
    intensity_details: Optional[IntensityCheckResult] = None
    structural_details: Optional[StructuralCheckResult] = None
    behavioral_signal_details: Optional[BehavioralSignalResult] = None
    confound_details: Optional[ConfoundCheckResult] = None

    @property
    def scores(self) -> Dict[str, float]:
        """Get all scores as a dictionary."""
        return {
            "dimension": self.dimension_score,
            "marker": self.marker_score,
            "boundary": self.boundary_score,
            "intensity": self.intensity_score,
            "structural": self.structural_score,
            "semantic": self.semantic_score,
            "behavioral_signal": self.behavioral_signal_score,
            "confound": self.confound_score,
        }

    @property
    def evaluated_scores(self) -> Dict[str, float]:
        """Get only scores that were evaluated (not -1)."""
        return {k: v for k, v in self.scores.items() if v >= 0}

    @property
    def weakest_dimension(self) -> str:
        """Return the name of the weakest evaluated dimension."""
        evaluated = self.evaluated_scores
        if not evaluated:
            return "unknown"
        return min(evaluated, key=evaluated.get)

    @property
    def strongest_dimension(self) -> str:
        """Return the name of the strongest evaluated dimension."""
        evaluated = self.evaluated_scores
        if not evaluated:
            return "unknown"
        return max(evaluated, key=evaluated.get)

    def get_improvement_guidance(self) -> str:
        """Get targeted guidance for improving this pair."""
        guidance = []

        if self.dimension_score >= 0 and self.dimension_score < 7:
            if self.dimension_details:
                guidance.append(
                    f"Dimension: Contrast is on '{self.dimension_details.what_differs}' "
                    f"but should be on '{self.dimension_details.expected_variable}'"
                )

        if self.marker_score >= 0 and self.marker_score < 7:
            if self.marker_details and self.marker_details.presence_markers_missing:
                guidance.append(
                    f"Markers: Missing in dst: {self.marker_details.presence_markers_missing[:3]}"
                )

        if self.boundary_score >= 0 and self.boundary_score < 7:
            if self.boundary_details and self.boundary_details.confused_with:
                guidance.append(
                    f"Boundary: May be confused with '{self.boundary_details.confused_with}'"
                )

        if self.intensity_score >= 0 and self.intensity_score < 7:
            if self.intensity_details:
                guidance.append(
                    f"Intensity: Expected {self.intensity_details.target_intensity}, "
                    f"got {self.intensity_details.actual_intensity}"
                )

        if self.semantic_score >= 0 and self.semantic_score < 5:
            guidance.append(f"Semantic: Distance too low ({self.semantic_score:.1f})")

        return " | ".join(guidance) if guidance else "General quality issues"


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
    """Protocol for generating seeds for contrast pair generation.

    Seeds are generated in one shot. Pair validation is the real quality
    gate - no separate scoring step is required.
    """

    async def generate(
        self,
        analysis: BehaviorAnalysis,
        count: int,
    ) -> List[Seed]:
        """Generate seeds for the behavior.

        Args:
            analysis: Behavior analysis to base seeds on.
            count: Number of seeds to generate.

        Returns:
            List of seeds with quality_score set from expected_contrast_strength.
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
        max_attempts: int,
    ) -> ContrastPair:
        """Regenerate a pair that failed validation.

        Args:
            pair: The original pair that failed.
            validation: The validation result with failure reasons.
            analysis: Behavior analysis for context.
            attempt: Which regeneration attempt this is (1-indexed).
            max_attempts: Maximum regeneration attempts allowed.

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

    For each score, uses the new value if it was evaluated (>= 0),
    otherwise keeps the current value.

    Args:
        current: Existing validation result.
        new: New validation result to merge in.

    Returns:
        Merged ValidationResult.
    """
    def pick_score(curr: float, new_val: float) -> float:
        """Pick new value if evaluated, else keep current."""
        return new_val if new_val >= 0 else curr

    # Merge reasoning
    curr_reason = current.reasoning.strip()
    new_reason = new.reasoning.strip()
    if curr_reason and new_reason:
        merged_reasoning = f"{curr_reason} | {new_reason}"
    else:
        merged_reasoning = curr_reason or new_reason

    return ValidationResult(
        is_valid=current.is_valid and new.is_valid,
        contrast_quality=max(current.contrast_quality, new.contrast_quality),
        reasoning=merged_reasoning,
        # Merge dimension scores
        dimension_score=pick_score(current.dimension_score, new.dimension_score),
        marker_score=pick_score(current.marker_score, new.marker_score),
        boundary_score=pick_score(current.boundary_score, new.boundary_score),
        intensity_score=pick_score(current.intensity_score, new.intensity_score),
        structural_score=pick_score(current.structural_score, new.structural_score),
        semantic_score=pick_score(current.semantic_score, new.semantic_score),
        behavioral_signal_score=pick_score(current.behavioral_signal_score, new.behavioral_signal_score),
        confound_score=pick_score(current.confound_score, new.confound_score),
        # Merge details (prefer new if available)
        dimension_details=new.dimension_details or current.dimension_details,
        marker_details=new.marker_details or current.marker_details,
        boundary_details=new.boundary_details or current.boundary_details,
        intensity_details=new.intensity_details or current.intensity_details,
        structural_details=new.structural_details or current.structural_details,
        behavioral_signal_details=new.behavioral_signal_details or current.behavioral_signal_details,
        confound_details=new.confound_details or current.confound_details,
    )
