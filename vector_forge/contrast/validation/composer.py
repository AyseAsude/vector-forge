"""Validation prompt composer for behavior-specific validation.

This module composes validation prompts from BehaviorAnalysis,
creating behavior-specific validation criteria at runtime.
"""

from __future__ import annotations

from typing import List, Optional

from vector_forge.contrast.protocols import (
    BehaviorAnalysis,
    ContrastPair,
    SignalIntensity,
    ValidationThresholds,
    DEFAULT_VALIDATION_THRESHOLDS,
)


class ValidationComposer:
    """Composes validation prompts from BehaviorAnalysis.

    Instead of using a static validation prompt, this composer builds
    behavior-specific prompts using all available analysis data:
    - behavioral_test for dimension checking
    - presence/absence markers for marker validation
    - not_this_behavior for boundary checking
    - intensity_calibration for intensity validation

    Example:
        >>> composer = ValidationComposer()
        >>> prompt = composer.compose(pair, analysis, SignalIntensity.MEDIUM)
        >>> # prompt is now behavior-specific
    """

    def __init__(self, thresholds: Optional[ValidationThresholds] = None):
        """Initialize the validation composer.

        Args:
            thresholds: Validation thresholds for prompt generation.
                       If None, uses DEFAULT_VALIDATION_THRESHOLDS.
        """
        self._thresholds = thresholds or DEFAULT_VALIDATION_THRESHOLDS

    def compose(
        self,
        pair: ContrastPair,
        analysis: BehaviorAnalysis,
        intensity: SignalIntensity = SignalIntensity.MEDIUM,
    ) -> str:
        """Compose a behavior-specific validation prompt.

        Args:
            pair: The contrast pair to validate.
            analysis: Behavior analysis with rich context.
            intensity: Target intensity level for this pair.

        Returns:
            Composed validation prompt string.
        """
        sections = [
            self._compose_header(),
            self._compose_behavior_context(analysis),
            self._compose_pair_section(pair),
            self._compose_dimension_check(analysis),
            self._compose_marker_check(analysis),
            self._compose_boundary_check(analysis),
            self._compose_intensity_check(analysis, intensity),
            self._compose_structural_check(),
            self._compose_output_format(),
        ]

        return "\n\n".join(s for s in sections if s)

    def _compose_header(self) -> str:
        """Compose the prompt header."""
        return """You are validating a contrast pair for steering vector extraction.
Your goal is to ensure the pair has CLEAN contrast on the CORRECT dimension."""

    def _compose_behavior_context(self, analysis: BehaviorAnalysis) -> str:
        """Compose behavior context section."""
        lines = ["## BEHAVIOR TO VALIDATE"]
        lines.append(f"**Name:** {analysis.behavior_name}")
        lines.append(f"**Description:** {analysis.description}")

        if analysis.core_definition:
            lines.append(f"**Core Definition:** {analysis.core_definition}")

        return "\n".join(lines)

    def _compose_pair_section(self, pair: ContrastPair) -> str:
        """Compose the pair to validate section."""
        lines = ["## PAIR TO VALIDATE"]
        lines.append(f'**Prompt:** "{pair.prompt}"')
        lines.append("")
        lines.append(f'**dst (should exhibit behavior):**\n"{pair.dst}"')
        lines.append("")
        lines.append(f'**src (should NOT exhibit behavior):**\n"{pair.src}"')

        # Include intensity from metadata if available
        if pair.metadata.get("intensity"):
            lines.append(f"\n**Target Intensity:** {pair.metadata['intensity']}")

        return "\n".join(lines)

    def _compose_dimension_check(self, analysis: BehaviorAnalysis) -> str:
        """Compose dimension check section using behavioral_test."""
        lines = ["## CHECK 1: DIMENSION (Critical)"]
        lines.append("Is the contrast specifically on the DISTINGUISHING VARIABLE?")
        lines.append("")

        if analysis.behavioral_test:
            bt = analysis.behavioral_test
            lines.append(f"**Distinguishing Variable:** {bt.distinguishing_variable}")
            lines.append(f"**dst should show:** {bt.present_response_pattern}")
            lines.append(f"**src should show:** {bt.absent_response_pattern}")
            lines.append("")
            lines.append("Questions to answer:")
            lines.append(f"1. Does dst clearly show: {bt.present_response_pattern}?")
            lines.append(f"2. Does src clearly show: {bt.absent_response_pattern}?")
            lines.append(f"3. Is the difference specifically on '{bt.distinguishing_variable}'?")
            lines.append("4. Or is it on something else (style, tone, length, helpfulness)?")
        else:
            lines.append("**Distinguishing Variable:** the target behavior")
            lines.append("**dst should:** exhibit the behavior clearly")
            lines.append("**src should:** not exhibit the behavior")
            lines.append("")
            lines.append("Questions to answer:")
            lines.append("1. Does dst clearly exhibit the behavior?")
            lines.append("2. Does src clearly NOT exhibit the behavior?")
            lines.append("3. Is the difference on the behavior, not on style/tone/length?")

        lines.append("")
        lines.append("Score 0-10:")
        lines.append("- 10: Contrast is exactly on the distinguishing variable")
        lines.append("- 7: Mostly correct dimension with minor style differences")
        lines.append("- 5: Mixed - some on correct dimension, some on style")
        lines.append("- 3: Primarily style/tone difference")
        lines.append("- 0: Wrong dimension entirely")

        return "\n".join(lines)

    def _compose_marker_check(self, analysis: BehaviorAnalysis) -> str:
        """Compose marker check section using presence/absence markers."""
        # Get all markers
        presence_markers = analysis.get_all_presence_markers()
        absence_markers = analysis.get_all_absence_markers()

        if not presence_markers and not absence_markers:
            # No markers available, skip this check
            return ""

        lines = ["## CHECK 2: MARKERS"]
        lines.append("Do expected linguistic markers appear in the correct places?")
        lines.append("")

        if presence_markers:
            lines.append("**Presence Markers (should appear in dst):**")
            for marker in presence_markers[:10]:  # Limit to avoid prompt bloat
                lines.append(f"- {marker}")
            lines.append("")

        if absence_markers:
            lines.append("**Absence Markers (should appear in src):**")
            for marker in absence_markers[:10]:
                lines.append(f"- {marker}")
            lines.append("")

        lines.append("Questions to answer:")
        lines.append("1. Which presence markers appear in dst?")
        lines.append("2. Which absence markers appear in src?")
        lines.append("3. Are there presence markers incorrectly in src?")
        lines.append("4. Are there absence markers incorrectly in dst?")
        lines.append("")
        lines.append("Score 0-10:")
        lines.append("- 10: All expected markers present in correct places")
        lines.append("- 7: Most markers correct, minor issues")
        lines.append("- 5: Some markers correct, some wrong")
        lines.append("- 3: Few markers in correct places")
        lines.append("- 0: Markers completely wrong")

        return "\n".join(lines)

    def _compose_boundary_check(self, analysis: BehaviorAnalysis) -> str:
        """Compose boundary check section using not_this_behavior."""
        if not analysis.not_this_behavior:
            # No negative examples, skip this check
            return ""

        lines = ["## CHECK 3: BOUNDARY"]
        lines.append("Is this clearly THIS behavior, not a similar one?")
        lines.append("")
        lines.append("**Similar behaviors to check against:**")

        for neg in analysis.not_this_behavior:
            lines.append(f"- **{neg.similar_behavior}**: {neg.why_different}")
            if neg.example:
                lines.append(f"  Example of what it's NOT: {neg.example}")

        lines.append("")
        lines.append("Questions to answer:")
        lines.append("1. Could this be confused with any of the similar behaviors?")
        lines.append("2. Is the contrast specifically on THIS behavior's dimension?")
        lines.append("3. Does dst show THIS behavior, not a similar one?")
        lines.append("")
        lines.append("Score 0-10:")
        lines.append("- 10: Clearly this behavior, no confusion possible")
        lines.append("- 7: Mostly clear, minor overlap with similar behavior")
        lines.append("- 5: Could be interpreted as either")
        lines.append("- 3: More like the similar behavior")
        lines.append("- 0: Clearly the wrong behavior")

        return "\n".join(lines)

    def _compose_intensity_check(
        self,
        analysis: BehaviorAnalysis,
        intensity: SignalIntensity,
    ) -> str:
        """Compose intensity check section using intensity_calibration."""
        lines = ["## CHECK 4: INTENSITY"]
        lines.append(f"Does the expression match the target intensity: **{intensity.value.upper()}**?")
        lines.append("")

        if analysis.intensity_calibration:
            calibration = analysis.intensity_calibration.get_for_intensity(intensity)
            lines.append(f"**What {intensity.value} looks like for THIS behavior:**")
            lines.append(calibration)
            lines.append("")

            # Show contrast with other intensities for context
            lines.append("**For reference, other intensities:**")
            if intensity != SignalIntensity.EXTREME:
                lines.append(f"- EXTREME: {analysis.intensity_calibration.extreme_looks_like}")
            if intensity != SignalIntensity.NATURAL:
                lines.append(f"- NATURAL: {analysis.intensity_calibration.natural_looks_like}")
        else:
            defaults = {
                SignalIntensity.EXTREME: "Maximum, unmistakable expression of behavior",
                SignalIntensity.HIGH: "Clearly present, obvious on first read",
                SignalIntensity.MEDIUM: "Noticeable, balanced expression",
                SignalIntensity.NATURAL: "Subtle but still clear contrast, realistic expression",
            }
            lines.append(f"**What {intensity.value} should look like:**")
            lines.append(defaults[intensity])

        lines.append("")
        lines.append("**IMPORTANT:** 'Natural' does NOT mean weak contrast.")
        lines.append("Even at natural intensity, the distinguishing variable should be CLEAR.")
        lines.append("Natural means realistic expression, not reduced signal.")
        lines.append("")
        lines.append("Questions to answer:")
        lines.append("1. Does dst's expression match the target intensity?")
        lines.append("2. Is it too extreme or too subtle for the target?")
        lines.append("3. Is the contrast still clear at this intensity?")
        lines.append("")
        lines.append("Score 0-10:")
        lines.append("- 10: Perfectly matches target intensity with clear contrast")
        lines.append("- 7: Close to target intensity")
        lines.append("- 5: Somewhat off from target")
        lines.append("- 3: Wrong intensity level")
        lines.append("- 0: Completely mismatched intensity")

        return "\n".join(lines)

    def _compose_structural_check(self) -> str:
        """Compose structural check section."""
        lines = ["## CHECK 5: STRUCTURAL"]
        lines.append("Are both responses well-formed and complete?")
        lines.append("")
        lines.append("Check:")
        lines.append("- Complete sentences, not truncated")
        lines.append("- Grammatically correct")
        lines.append("- Coherent and sensible responses")
        lines.append("- No garbage or malformed content")
        lines.append("")
        lines.append("Score 0-10:")
        lines.append("- 10: Both perfectly well-formed")
        lines.append("- 7: Minor issues that don't affect meaning")
        lines.append("- 5: Some issues but still usable")
        lines.append("- 3: Significant quality issues")
        lines.append("- 0: Garbage or incoherent")

        return "\n".join(lines)

    def _compose_output_format(self) -> str:
        """Compose output format section."""
        # Use thresholds.format_for_prompt() for consistency with validation logic
        validity_criteria = self._thresholds.format_for_prompt()

        return f"""## OUTPUT FORMAT

Return a JSON object with the following structure:

```json
{{
  "dimension_check": {{
    "score": <0-10>,
    "is_correct": <true if score >= {self._thresholds.dimension:.0f}>,
    "what_differs": "what actually differs between dst and src",
    "dst_pattern_match": <true/false>,
    "src_pattern_match": <true/false>,
    "explanation": "why dimension is correct or wrong"
  }},
  "marker_check": {{
    "score": <0-10>,
    "presence_markers_found": ["markers found in dst"],
    "presence_markers_missing": ["expected markers not in dst"],
    "absence_markers_found": ["markers found in src"],
    "absence_markers_missing": ["expected markers not in src"],
    "explanation": "marker analysis"
  }},
  "boundary_check": {{
    "score": <0-10>,
    "is_correct_behavior": <true/false>,
    "confused_with": "<similar behavior name or null>",
    "explanation": "why this is/isn't the right behavior"
  }},
  "intensity_check": {{
    "score": <0-10>,
    "actual_intensity": "<extreme/high/medium/natural>",
    "matches_calibration": <true/false>,
    "explanation": "how intensity matches or doesn't"
  }},
  "structural_check": {{
    "score": <0-10>,
    "dst_wellformed": <true/false>,
    "src_wellformed": <true/false>,
    "dst_complete": <true/false>,
    "src_complete": <true/false>,
    "issues": ["any issues found"]
  }},
  "overall": {{
    "is_valid": <true if all critical checks pass>,
    "contrast_quality": <0-10 aggregate score>,
    "primary_issue": "main issue if not valid, or 'none'",
    "reasoning": "brief overall assessment"
  }}
}}
```

{validity_criteria}

Return ONLY the JSON object, no other text."""


def compose_validation_prompt(
    pair: ContrastPair,
    analysis: BehaviorAnalysis,
    intensity: SignalIntensity = SignalIntensity.MEDIUM,
    thresholds: Optional[ValidationThresholds] = None,
) -> str:
    """Convenience function to compose validation prompt.

    Args:
        pair: The contrast pair to validate.
        analysis: Behavior analysis with rich context.
        intensity: Target intensity level.
        thresholds: Validation thresholds. If None, uses defaults.

    Returns:
        Composed validation prompt.
    """
    composer = ValidationComposer(thresholds=thresholds)
    return composer.compose(pair, analysis, intensity)
