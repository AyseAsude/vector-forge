"""Contrast pair regenerator with targeted feedback from rich validation.

This module regenerates pairs that failed validation with
targeted feedback based on dimension-specific issues.
"""

from __future__ import annotations

import logging
from typing import Optional

from vector_forge.contrast.protocols import (
    PairRegeneratorProtocol,
    ContrastPair,
    ValidationResult,
    BehaviorAnalysis,
    SignalIntensity,
)
from vector_forge.contrast.utils import safe_parse_llm_json
from vector_forge.core.protocols import Message
from vector_forge.llm import JSON_RESPONSE_FORMAT
from vector_forge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class RegenerationPromptComposer:
    """Composes regeneration prompts from validation results and analysis.

    Uses the rich validation data to create targeted regeneration prompts
    that focus on the specific issues identified.
    """

    def compose(
        self,
        pair: ContrastPair,
        validation: ValidationResult,
        analysis: BehaviorAnalysis,
        attempt: int,
        intensity: SignalIntensity,
    ) -> str:
        """Compose a targeted regeneration prompt.

        Args:
            pair: The original pair that failed.
            validation: Rich validation result with dimension details.
            analysis: Behavior analysis for context.
            attempt: Which regeneration attempt this is.
            intensity: Target intensity for the pair.

        Returns:
            Composed regeneration prompt.
        """
        sections = [
            self._compose_header(attempt),
            self._compose_behavior_context(analysis),
            self._compose_original_pair(pair),
            self._compose_validation_feedback(validation, analysis),
            self._compose_targeted_fixes(validation, analysis, intensity),
            self._compose_attempt_instructions(attempt),
            self._compose_output_format(),
        ]

        return "\n\n".join(s for s in sections if s)

    def _compose_header(self, attempt: int) -> str:
        """Compose header with attempt context."""
        return f"""The previous contrast pair failed validation. Generate an improved version.
**Attempt {attempt} of 3**"""

    def _compose_behavior_context(self, analysis: BehaviorAnalysis) -> str:
        """Compose behavior context section."""
        lines = ["## BEHAVIOR"]
        lines.append(f"**Name:** {analysis.behavior_name}")
        lines.append(f"**Description:** {analysis.description}")

        if analysis.core_definition:
            lines.append(f"**Core:** {analysis.core_definition}")

        # Include behavioral test if available
        if analysis.behavioral_test:
            bt = analysis.behavioral_test
            lines.append("")
            lines.append("**Behavioral Test:**")
            lines.append(f"- Distinguishing Variable: {bt.distinguishing_variable}")
            lines.append(f"- dst should show: {bt.present_response_pattern}")
            lines.append(f"- src should show: {bt.absent_response_pattern}")

        # Include negative examples
        if analysis.not_this_behavior:
            lines.append("")
            lines.append("**What this is NOT:**")
            for neg in analysis.not_this_behavior[:3]:
                lines.append(f"- NOT {neg.similar_behavior}: {neg.why_different}")

        return "\n".join(lines)

    def _compose_original_pair(self, pair: ContrastPair) -> str:
        """Compose original pair section."""
        lines = ["## ORIGINAL PAIR"]
        lines.append(f'**Prompt:** "{pair.prompt}"')
        lines.append("")
        lines.append(f'**dst (should exhibit behavior):**\n"{pair.dst}"')
        lines.append("")
        lines.append(f'**src (should NOT exhibit behavior):**\n"{pair.src}"')

        return "\n".join(lines)

    def _compose_validation_feedback(
        self,
        validation: ValidationResult,
        analysis: BehaviorAnalysis,
    ) -> str:
        """Compose validation feedback section with dimension-specific issues."""
        lines = ["## VALIDATION RESULTS"]

        # Overall scores
        lines.append("**Dimension Scores:**")
        if validation.dimension_score >= 0:
            status = "PASS" if validation.dimension_score >= 6 else "FAIL"
            lines.append(f"- Dimension: {validation.dimension_score:.0f}/10 [{status}]")
        if validation.structural_score >= 0:
            status = "PASS" if validation.structural_score >= 7 else "FAIL"
            lines.append(f"- Structural: {validation.structural_score:.0f}/10 [{status}]")
        if validation.marker_score >= 0:
            status = "PASS" if validation.marker_score >= 5 else "FAIL"
            lines.append(f"- Markers: {validation.marker_score:.0f}/10 [{status}]")
        if validation.boundary_score >= 0:
            status = "PASS" if validation.boundary_score >= 5 else "FAIL"
            lines.append(f"- Boundary: {validation.boundary_score:.0f}/10 [{status}]")
        if validation.intensity_score >= 0:
            lines.append(f"- Intensity: {validation.intensity_score:.0f}/10")

        lines.append("")
        lines.append(f"**Weakest Dimension:** {validation.weakest_dimension}")
        lines.append("")

        # Dimension-specific feedback
        lines.append("**Specific Issues:**")

        # Dimension check issues
        if validation.dimension_details and validation.dimension_score < 7:
            dd = validation.dimension_details
            lines.append(f"- DIMENSION: Contrast is on '{dd.what_differs}' but should be on '{dd.expected_variable}'")
            if not dd.dst_pattern_match:
                lines.append(f"  - dst does NOT match expected pattern")
            if not dd.src_pattern_match:
                lines.append(f"  - src does NOT match expected pattern")

        # Marker check issues
        if validation.marker_details and validation.marker_score >= 0 and validation.marker_score < 7:
            md = validation.marker_details
            if md.presence_markers_missing:
                lines.append(f"- MARKERS: Missing in dst: {md.presence_markers_missing[:3]}")
            if md.absence_markers_missing:
                lines.append(f"- MARKERS: Missing in src: {md.absence_markers_missing[:3]}")

        # Boundary check issues
        if validation.boundary_details and validation.boundary_score >= 0 and validation.boundary_score < 7:
            bd = validation.boundary_details
            if bd.confused_with:
                lines.append(f"- BOUNDARY: May be confused with '{bd.confused_with}'")

        # Intensity check issues
        if validation.intensity_details and validation.intensity_score < 7:
            id = validation.intensity_details
            lines.append(f"- INTENSITY: Expected {id.target_intensity}, got {id.actual_intensity}")

        # Structural issues
        if validation.structural_details and validation.structural_score < 8:
            sd = validation.structural_details
            if sd.issues:
                for issue in sd.issues[:3]:
                    lines.append(f"- STRUCTURAL: {issue}")

        return "\n".join(lines)

    def _compose_targeted_fixes(
        self,
        validation: ValidationResult,
        analysis: BehaviorAnalysis,
        intensity: SignalIntensity,
    ) -> str:
        """Compose targeted fix instructions based on validation."""
        lines = ["## TARGETED FIXES REQUIRED"]

        # Get the improvement guidance
        guidance = validation.get_improvement_guidance()
        lines.append(f"**Priority:** {guidance}")
        lines.append("")

        # Dimension fixes
        if validation.dimension_score < 7:
            lines.append("### Fix Dimension Issues")
            if analysis.behavioral_test:
                bt = analysis.behavioral_test
                lines.append(f"The contrast MUST be on: **{bt.distinguishing_variable}**")
                lines.append(f"- Make dst clearly show: {bt.present_response_pattern}")
                lines.append(f"- Make src clearly show: {bt.absent_response_pattern}")
            else:
                lines.append("Ensure dst clearly exhibits the behavior while src does not")
            lines.append("")

        # Marker fixes
        if validation.marker_score >= 0 and validation.marker_score < 7:
            lines.append("### Fix Marker Issues")
            if validation.marker_details:
                md = validation.marker_details
                if md.presence_markers_missing:
                    lines.append(f"Include these in dst: {md.presence_markers_missing[:5]}")
                if md.absence_markers_missing:
                    lines.append(f"Include these in src: {md.absence_markers_missing[:5]}")
            lines.append("")

        # Boundary fixes
        if validation.boundary_score >= 0 and validation.boundary_score < 7:
            lines.append("### Fix Boundary Issues")
            if validation.boundary_details and validation.boundary_details.confused_with:
                confused = validation.boundary_details.confused_with
                # Find the negative example for guidance
                for neg in analysis.not_this_behavior:
                    if neg.similar_behavior.lower() == confused.lower():
                        lines.append(f"This looks like '{confused}' instead of the target behavior")
                        lines.append(f"Remember: {neg.why_different}")
                        break
            lines.append("")

        # Intensity fixes
        if validation.intensity_score < 7:
            lines.append("### Fix Intensity Issues")
            lines.append(f"Target intensity: **{intensity.value.upper()}**")
            if analysis.intensity_calibration:
                calibration = analysis.intensity_calibration.get_for_intensity(intensity)
                lines.append(f"What {intensity.value} looks like: {calibration}")
            lines.append("Note: 'Natural' still requires CLEAR contrast, just realistic expression")
            lines.append("")

        return "\n".join(lines)

    def _compose_attempt_instructions(self, attempt: int) -> str:
        """Compose attempt-specific instructions."""
        instructions = {
            1: """## ATTEMPT 1 APPROACH
**Standard improvement**: Make targeted fixes based on the specific issues identified.
- Focus on the weakest dimension first
- Make incremental improvements
- Maintain what was working""",

            2: """## ATTEMPT 2 APPROACH
**Stronger improvement**: Previous attempt wasn't enough. Be more aggressive.
- Make the contrast MORE obvious
- If dimension was wrong, completely rethink the response approach
- Consider restructuring responses to highlight the distinguishing variable""",

            3: """## ATTEMPT 3 APPROACH
**Maximum contrast**: Previous attempts failed. Use extreme measures.
- Make dst the STRONGEST possible example of the behavior
- Make src the CLEAREST possible counter-example
- The difference should be immediately obvious to anyone
- If the scenario itself is problematic, adapt it significantly""",
        }

        return instructions.get(min(attempt, 3), instructions[3])

    def _compose_output_format(self) -> str:
        """Compose output format section."""
        return """## OUTPUT

Return JSON:
{
  "prompt": "The improved prompt (can reuse original if fine)",
  "dst": "The improved dst response",
  "src": "The improved src response",
  "improvements_made": ["list of specific improvements"],
  "expected_dimension_score": <0-10>,
  "fixed_issues": {
    "dimension": "how you fixed dimension issues",
    "markers": "how you fixed marker issues",
    "boundary": "how you fixed boundary issues",
    "intensity": "how you fixed intensity issues"
  }
}

Return ONLY the JSON object."""


class ContrastRegenerator(PairRegeneratorProtocol):
    """Regenerates pairs that failed validation with targeted feedback.

    Uses rich validation data to create targeted regeneration prompts
    that focus on the specific dimension issues identified.

    Example:
        >>> regenerator = ContrastRegenerator(llm_client)
        >>> fixed_pair = await regenerator.regenerate(
        ...     pair, validation_result, analysis, attempt=1
        ... )
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        temperature: float = 0.8,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the regenerator.

        Args:
            llm_client: LLM client for regeneration.
            temperature: Generation temperature.
            max_tokens: Maximum response tokens (None = provider default).
        """
        self._llm = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._composer = RegenerationPromptComposer()

    async def regenerate(
        self,
        pair: ContrastPair,
        validation: ValidationResult,
        analysis: BehaviorAnalysis,
        attempt: int,
    ) -> ContrastPair:
        """Regenerate a pair that failed validation.

        Uses rich validation data to create targeted feedback for
        the LLM to fix specific dimension issues.

        Args:
            pair: The original pair that failed.
            validation: Rich validation result with dimension details.
            analysis: Behavior analysis for context.
            attempt: Which regeneration attempt this is (1-indexed).

        Returns:
            New ContrastPair with improvements.
        """
        # Determine intensity from pair metadata
        intensity_str = pair.metadata.get("intensity", "medium")
        try:
            intensity = SignalIntensity(intensity_str.lower())
        except ValueError:
            intensity = SignalIntensity.MEDIUM

        # Compose targeted regeneration prompt
        prompt = self._composer.compose(
            pair, validation, analysis, attempt, intensity
        )

        # Increase temperature slightly for later attempts
        temperature = min(1.0, self._temperature + (attempt - 1) * 0.1)

        try:
            response = await self._llm.complete(
                messages=[Message(role="user", content=prompt)],
                temperature=temperature,
                max_tokens=self._max_tokens,
                response_format=JSON_RESPONSE_FORMAT,
            )

            data = safe_parse_llm_json(response.content)

        except Exception as e:
            logger.error(f"Regeneration failed: {e}")
            return pair  # Return original if regeneration fails

        # Use improved versions, falling back to originals
        new_prompt = data.get("prompt") or pair.prompt
        new_dst = data.get("dst") or pair.dst
        new_src = data.get("src") or pair.src

        return ContrastPair(
            prompt=new_prompt,
            dst=new_dst,
            src=new_src,
            seed=pair.seed,
            metadata={
                **pair.metadata,
                "regenerated": True,
                "attempt": attempt,
                "improvements_made": data.get("improvements_made", []),
                "expected_dimension_score": data.get("expected_dimension_score"),
                "fixed_issues": data.get("fixed_issues", {}),
                "original_validation": {
                    "dimension_score": validation.dimension_score,
                    "marker_score": validation.marker_score,
                    "boundary_score": validation.boundary_score,
                    "intensity_score": validation.intensity_score,
                    "structural_score": validation.structural_score,
                    "contrast_quality": validation.contrast_quality,
                    "weakest_dimension": validation.weakest_dimension,
                },
            },
        )
