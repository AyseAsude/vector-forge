"""LLM-based contrast validation with composable prompts.

This module provides thorough validation using an LLM judge to verify
contrast pairs using behavior-specific prompts composed from analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from vector_forge.contrast.protocols import (
    ContrastValidatorProtocol,
    ContrastPair,
    BehaviorAnalysis,
    ValidationResult,
    SignalIntensity,
    DimensionCheckResult,
    MarkerCheckResult,
    BoundaryCheckResult,
    IntensityCheckResult,
    StructuralCheckResult,
)
from vector_forge.contrast.utils import safe_parse_llm_json
from vector_forge.contrast.validation.composer import ValidationComposer
from vector_forge.core.protocols import Message
from vector_forge.llm import JSON_RESPONSE_FORMAT
from vector_forge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class LLMContrastValidator(ContrastValidatorProtocol):
    """Composable LLM-based contrast validation.

    Uses a ValidationComposer to build behavior-specific validation prompts
    from BehaviorAnalysis. This ensures validation checks are tailored to
    the specific behavior being extracted.

    Validation Dimensions:
    1. Dimension Check - Is contrast on the distinguishing variable?
    2. Marker Check - Do presence/absence markers appear correctly?
    3. Boundary Check - Is this the right behavior, not a similar one?
    4. Intensity Check - Does expression match target intensity?
    5. Structural Check - Are responses well-formed and complete?

    Example:
        >>> validator = LLMContrastValidator(judge_llm)
        >>> result = await validator.validate(pair, analysis, SignalIntensity.MEDIUM)
        >>> print(f"Valid: {result.is_valid}, Dimension: {result.dimension_score}")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        # Validation thresholds
        min_dimension_score: float = 6.0,
        min_structural_score: float = 7.0,
        min_marker_score: float = 5.0,
        min_boundary_score: float = 5.0,
    ):
        """Initialize the LLM validator.

        Args:
            llm_client: LLM client for judging.
            temperature: Generation temperature (lower = more consistent).
            max_tokens: Maximum response tokens (None = provider default).
            min_dimension_score: Minimum dimension check score to pass.
            min_structural_score: Minimum structural check score to pass.
            min_marker_score: Minimum marker check score to pass (if markers available).
            min_boundary_score: Minimum boundary check score to pass (if boundaries available).
        """
        self._llm = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._composer = ValidationComposer()

        # Thresholds
        self._min_dimension = min_dimension_score
        self._min_structural = min_structural_score
        self._min_marker = min_marker_score
        self._min_boundary = min_boundary_score

    async def validate(
        self,
        pair: ContrastPair,
        analysis: BehaviorAnalysis,
        intensity: Optional[SignalIntensity] = None,
    ) -> ValidationResult:
        """Validate contrast pair using behavior-specific checks.

        Composes a validation prompt from the analysis and checks:
        1. Dimension - Is contrast on the distinguishing variable?
        2. Markers - Do expected markers appear correctly?
        3. Boundary - Is this the right behavior, not a similar one?
        4. Intensity - Does it match target intensity calibration?
        5. Structural - Are responses well-formed?

        Args:
            pair: The contrast pair to validate.
            analysis: Behavior analysis for composing validation.
            intensity: Target intensity (defaults to MEDIUM or from pair metadata).

        Returns:
            ValidationResult with dimension-specific scores and details.
        """
        # Determine intensity from pair metadata or default
        if intensity is None:
            intensity_str = pair.metadata.get("intensity", "medium")
            try:
                intensity = SignalIntensity(intensity_str.lower())
            except ValueError:
                intensity = SignalIntensity.MEDIUM

        # Compose behavior-specific validation prompt
        prompt = self._composer.compose(pair, analysis, intensity)

        try:
            response = await self._llm.complete(
                messages=[Message(role="user", content=prompt)],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format=JSON_RESPONSE_FORMAT,
            )

            data = safe_parse_llm_json(response.content)

        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return self._create_error_result(str(e))

        # Extract and build detailed results
        return self._build_result(data, analysis, intensity)

    def _build_result(
        self,
        data: Dict[str, Any],
        analysis: BehaviorAnalysis,
        intensity: SignalIntensity,
    ) -> ValidationResult:
        """Build ValidationResult from parsed LLM response."""

        # Extract dimension check
        dim_data = data.get("dimension_check", {})
        dimension_details = DimensionCheckResult(
            score=float(dim_data.get("score", 5.0)),
            is_correct=dim_data.get("is_correct", False),
            what_differs=dim_data.get("what_differs", ""),
            expected_variable=(
                analysis.behavioral_test.distinguishing_variable
                if analysis.behavioral_test else "the target behavior"
            ),
            dst_pattern_match=dim_data.get("dst_pattern_match", False),
            src_pattern_match=dim_data.get("src_pattern_match", False),
            explanation=dim_data.get("explanation", ""),
        )

        # Extract marker check
        marker_data = data.get("marker_check", {})
        marker_details = MarkerCheckResult(
            score=float(marker_data.get("score", -1.0)),
            presence_markers_found=marker_data.get("presence_markers_found", []),
            presence_markers_missing=marker_data.get("presence_markers_missing", []),
            absence_markers_found=marker_data.get("absence_markers_found", []),
            absence_markers_missing=marker_data.get("absence_markers_missing", []),
            explanation=marker_data.get("explanation", ""),
        )

        # Extract boundary check
        boundary_data = data.get("boundary_check", {})
        boundary_details = BoundaryCheckResult(
            score=float(boundary_data.get("score", -1.0)),
            is_correct_behavior=boundary_data.get("is_correct_behavior", True),
            confused_with=boundary_data.get("confused_with"),
            explanation=boundary_data.get("explanation", ""),
        )

        # Extract intensity check
        intensity_data = data.get("intensity_check", {})
        intensity_details = IntensityCheckResult(
            score=float(intensity_data.get("score", 5.0)),
            target_intensity=intensity.value,
            actual_intensity=intensity_data.get("actual_intensity", "unknown"),
            matches_calibration=intensity_data.get("matches_calibration", False),
            calibration_description=(
                analysis.intensity_calibration.get_for_intensity(intensity)
                if analysis.intensity_calibration else ""
            ),
            explanation=intensity_data.get("explanation", ""),
        )

        # Extract structural check
        struct_data = data.get("structural_check", {})
        structural_details = StructuralCheckResult(
            score=float(struct_data.get("score", 10.0)),
            dst_wellformed=struct_data.get("dst_wellformed", True),
            src_wellformed=struct_data.get("src_wellformed", True),
            dst_complete=struct_data.get("dst_complete", True),
            src_complete=struct_data.get("src_complete", True),
            issues=struct_data.get("issues", []),
        )

        # Extract overall
        overall_data = data.get("overall", {})
        contrast_quality = float(overall_data.get("contrast_quality", 5.0))
        primary_issue = overall_data.get("primary_issue", "none")
        reasoning = overall_data.get("reasoning", "")

        # Determine validity based on thresholds
        is_valid = self._check_validity(
            dimension_details,
            marker_details,
            boundary_details,
            structural_details,
            analysis,
        )

        # Build reasoning string
        reasoning_parts = [
            f"Dim:{dimension_details.score:.0f}",
            f"Struct:{structural_details.score:.0f}",
        ]
        if marker_details.score >= 0:
            reasoning_parts.append(f"Mark:{marker_details.score:.0f}")
        if boundary_details.score >= 0:
            reasoning_parts.append(f"Bound:{boundary_details.score:.0f}")
        reasoning_parts.append(f"Int:{intensity_details.score:.0f}")

        if primary_issue and primary_issue != "none":
            reasoning_parts.append(f"Issue: {primary_issue}")

        full_reasoning = " | ".join(reasoning_parts)
        if reasoning:
            full_reasoning += f" | {reasoning}"

        return ValidationResult(
            is_valid=is_valid,
            contrast_quality=contrast_quality,
            reasoning=full_reasoning,
            # Dimension scores
            dimension_score=dimension_details.score,
            marker_score=marker_details.score,
            boundary_score=boundary_details.score,
            intensity_score=intensity_details.score,
            structural_score=structural_details.score,
            # Detailed results
            dimension_details=dimension_details,
            marker_details=marker_details,
            boundary_details=boundary_details,
            intensity_details=intensity_details,
            structural_details=structural_details,
        )

    def _check_validity(
        self,
        dimension: DimensionCheckResult,
        marker: MarkerCheckResult,
        boundary: BoundaryCheckResult,
        structural: StructuralCheckResult,
        analysis: BehaviorAnalysis,
    ) -> bool:
        """Check if pair passes all validity thresholds."""

        # Critical checks (always required)
        if dimension.score < self._min_dimension:
            return False
        if structural.score < self._min_structural:
            return False

        # Optional checks (only if data was available)
        has_markers = (
            analysis.get_all_presence_markers() or
            analysis.get_all_absence_markers()
        )
        if has_markers and marker.score >= 0:
            if marker.score < self._min_marker:
                return False

        has_boundaries = bool(analysis.not_this_behavior)
        if has_boundaries and boundary.score >= 0:
            if boundary.score < self._min_boundary:
                return False

        return True

    def _create_error_result(self, error_msg: str) -> ValidationResult:
        """Create a ValidationResult for error cases."""
        return ValidationResult(
            is_valid=False,
            contrast_quality=0.0,
            reasoning=f"Validation error: {error_msg}",
        )
